# -*- coding: utf-8 -*-
import time
from flask import Blueprint, jsonify, request
from app.services.search import qa_system, search_queue, rate_limiter
from app.config import AppConfig
from app.constants import HttpStatus, ErrorMessages
from app.utils import logger
from app.exceptions import (
    SearchError, SearchTimeoutError, SearchRateLimitError, 
    SearchQueueFullError, ModelNotLoadedError
)

search_bp = Blueprint('search', __name__)

@search_bp.route('/search', methods=['POST'])
def search_route():
    """문서 검색
    
    성능 로깅: 검색 시간 측정 및 기록
    """
    start_time = time.perf_counter()
    
    # Rate Limit 체크
    client_ip = request.remote_addr or 'unknown'
    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded: {client_ip}")
        return jsonify({
            'success': False, 
            'message': ErrorMessages.SEARCH_RATE_LIMITED,
            'error_code': 'RATE_LIMITED'
        }), HttpStatus.TOO_MANY_REQUESTS
         
    # 검색 큐 슬롯 획득
    if not search_queue.acquire(timeout=5):
        logger.warning("Search queue full, request rejected")
        return jsonify({
            'success': False, 
            'message': ErrorMessages.SEARCH_QUEUE_FULL,
            'error_code': 'QUEUE_FULL'
        }), HttpStatus.SERVICE_UNAVAILABLE
         
    try:
        data = request.json
        if not data:
            return jsonify({
                'success': False, 
                'message': '요청 본문이 비어있습니다',
                'error_code': 'EMPTY_REQUEST'
            }), HttpStatus.BAD_REQUEST
            
        # 'query' 또는 'q' 파라미터 모두 지원 (프론트엔드 호환성)
        query = data.get('query') or data.get('q', '')
        k = data.get('k', 5)
        hybrid = data.get('hybrid', True)
        sort_by = data.get('sort_by', 'relevance')
        filter_file = data.get('filter_file')
        
        # 검색 히스토리 저장
        if getattr(qa_system, '_search_history', None):
             qa_system._search_history.add(query)
        
        # 검색 수행
        res = qa_system.search(query, k, hybrid, sort_by, filter_file)
        
        # 성능 로깅
        duration_ms = (time.perf_counter() - start_time) * 1000
        result_count = len(res.data) if res.data else 0
        logger.info(f"검색 완료: query='{query[:30]}...' results={result_count} duration={duration_ms:.1f}ms")
        
        return jsonify(res.to_dict())
        
    except ModelNotLoadedError as e:
        logger.error(f"모델 미로드 상태에서 검색 시도: {e}")
        return jsonify({
            'success': False, 
            'message': str(e),
            'error_code': 'MODEL_NOT_LOADED'
        }), HttpStatus.SERVICE_UNAVAILABLE
        
    except (SearchError, SearchTimeoutError) as e:
        logger.error(f"검색 오류: {e}")
        return jsonify({
            'success': False, 
            'message': str(e),
            'error_code': 'SEARCH_ERROR'
        }), HttpStatus.INTERNAL_ERROR
        
    except Exception as e:
        logger.error(f"검색 API 예상치 못한 오류: {e}")
        return jsonify({
            'success': False, 
            'message': f'{ErrorMessages.SEARCH_FAILED}: {str(e)}',
            'error_code': 'UNEXPECTED_ERROR'
        }), HttpStatus.INTERNAL_ERROR
    finally:
        search_queue.release()

# 경로 수정: /history -> /search/history (JS 호출 경로와 일치)
@search_bp.route('/search/history', methods=['GET'])
def get_history():
    """검색 기록 반환"""
    history = getattr(qa_system, '_search_history', None)
    if not history:
        return jsonify({'recent': [], 'popular': []})
        
    limit = request.args.get('limit', 10, type=int)
    return jsonify({
        'recent': history.get_recent(limit),
        'popular': history.get_popular(limit)
    })

# 새로 추가: /search/suggest (자동완성)
@search_bp.route('/search/suggest', methods=['GET'])
def get_suggestions():
    """검색어 자동완성 제안"""
    query = request.args.get('q', '')
    limit = request.args.get('limit', 8, type=int)
    
    history = getattr(qa_system, '_search_history', None)
    if not history or not query:
        return jsonify({'suggestions': []})
    
    suggestions = history.suggest(query, limit)
    return jsonify({'suggestions': suggestions})

@search_bp.route('/cache/clear', methods=['POST'])
def clear_cache():
    """검색 캐시 초기화"""
    try:
        if hasattr(qa_system, '_search_cache'):
            qa_system._search_cache.clear()
        return jsonify({'success': True, 'message': '캐시가 초기화되었습니다'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'캐시 초기화 실패: {str(e)}'}), 500


