# -*- coding: utf-8 -*-
import time
from flask import Blueprint, jsonify, request
from app.services.search import qa_system, search_queue, rate_limiter
from app.services.document import TextHighlighter
from app.config import AppConfig
from app.constants import HttpStatus, ErrorMessages
from app.utils import logger
from app.auth import admin_required
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
        filter_file_id = data.get('filter_file_id')
        
        # 검색 수행
        res = qa_system.search(query, k, hybrid, sort_by, filter_file, filter_file_id)

        # 유효한 검색만 히스토리에 저장
        if res.success and getattr(qa_system, '_search_history', None):
            qa_system._search_history.add(query)
        
        # ====================================================================
        # 응답 최적화 v2.6.1: 콘텐츠 길이 제한 및 하이라이트 사전 처리
        # ====================================================================
        max_content_len = getattr(AppConfig, 'MAX_CONTENT_PREVIEW', 1500)
        
        if res.success and res.data:
            for item in res.data:
                # 콘텐츠 길이 제한 (응답 크기 감소)
                content = item.get('content') or ''
                if len(content) > max_content_len:
                    content = content[:max_content_len] + '...'
                    item['content'] = content
                    item['is_truncated'] = True
                
                # 하이라이트 사전 계산 (클라이언트 부담 감소)
                if not item.get('content_highlighted'):
                    item['content_highlighted'] = TextHighlighter.highlight(content, query)
                
                # 점수 소수점 제한 (JSON 크기 감소)
                score = item.get('score')
                if isinstance(score, (int, float)):
                    item['score'] = round(score, 4)
                vec_score = item.get('vec_score')
                if isinstance(vec_score, (int, float)):
                    item['vec_score'] = round(vec_score, 4)
                bm25_score = item.get('bm25_score')
                if isinstance(bm25_score, (int, float)):
                    item['bm25_score'] = round(bm25_score, 4)
        
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
        return jsonify({'success': True, 'recent': [], 'popular': [], 'popular_legacy': []})
        
    limit = request.args.get('limit', 10, type=int)
    popular = history.get_popular(limit)
    normalized_popular = []
    popular_legacy = []

    for item in popular:
        if isinstance(item, dict):
            query = str(item.get('query', ''))
            count = int(item.get('count', 0) or 0)
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            query = str(item[0])
            count = int(item[1] or 0)
        else:
            continue
        normalized_popular.append({'query': query, 'count': count})
        popular_legacy.append([query, count])

    return jsonify({
        'success': True,
        'recent': history.get_recent(limit),
        'popular': normalized_popular,
        'popular_legacy': popular_legacy
    })

# 새로 추가: /search/suggest (자동완성)
@search_bp.route('/search/suggest', methods=['GET'])
def get_suggestions():
    """검색어 자동완성 제안"""
    query = request.args.get('q', '')
    limit = request.args.get('limit', 8, type=int)
    
    history = getattr(qa_system, '_search_history', None)
    if not history or not query:
        return jsonify({'success': True, 'suggestions': []})
    
    suggestions = history.suggest(query, limit)
    return jsonify({'success': True, 'suggestions': suggestions})

@search_bp.route('/cache/clear', methods=['POST'])
@admin_required
def clear_cache():
    """검색 캐시 초기화"""
    try:
        if hasattr(qa_system, '_search_cache'):
            qa_system._search_cache.clear()
        return jsonify({'success': True, 'message': '캐시가 초기화되었습니다'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'캐시 초기화 실패: {str(e)}'}), 500


