# -*- coding: utf-8 -*-
from flask import Blueprint, jsonify, request
from app.services.search import qa_system, search_queue, rate_limiter
from app.config import AppConfig

search_bp = Blueprint('search', __name__)

@search_bp.route('/search', methods=['POST'])
def search_route():
    """문서 검색"""
    # Rate Limit 체크
    if not rate_limiter.is_allowed(request.remote_addr):
         return jsonify({'success': False, 'message': '요청이 너무 많습니다. 잠시 후 다시 시도해주세요.'}), 429
         
    # 검색 큐 슬롯 획득
    if not search_queue.acquire(timeout=5):
         return jsonify({'success': False, 'message': '서버가 혼잡합니다. 잠시 후 다시 시도해주세요.'}), 503
         
    try:
        data = request.json
        query = data.get('query', '')
        k = data.get('k', 5)
        hybrid = data.get('hybrid', True)
        sort_by = data.get('sort_by', 'relevance')
        filter_file = data.get('filter_file')
        
        # 검색 히스토리 저장
        if getattr(qa_system, '_search_history', None):
             qa_system._search_history.add(query)
        
        # 검색 수행
        res = qa_system.search(query, k, hybrid, sort_by, filter_file)
        return jsonify(res.to_dict())
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        search_queue.release()
        
@search_bp.route('/history', methods=['GET'])
def get_history():
    """검색 기록 반환"""
    history = getattr(qa_system, '_search_history', None)
    if not history:
        return jsonify({'recent': [], 'popular': []})
        
    return jsonify({
        'recent': history.get_recent(),
        'popular': history.get_popular()
    })

@search_bp.route('/cache/clear', methods=['POST'])
def clear_cache():
    """검색 캐시 초기화"""
    res = qa_system.clear_cache()
    return jsonify(res.to_dict())
