# -*- coding: utf-8 -*-
import json
import os

from flask import Blueprint, jsonify, request, session
from app.config import AppConfig
from app.services.search import qa_system
from app.utils import logger, get_app_directory

system_bp = Blueprint('system', __name__)

@system_bp.route('/models', methods=['GET'])
def get_models():
    """사용 가능한 모델 목록 반환"""
    return jsonify({
        'success': True,
        'models': list(AppConfig.AVAILABLE_MODELS.keys()),
        'current': getattr(qa_system, 'model_name', AppConfig.DEFAULT_MODEL)
    })

@system_bp.route('/models', methods=['POST'])
def set_model():
    """AI 모델 변경
    
    런타임에 모델을 변경합니다. 기존 인덱스는 유지되며,
    새 검색 시 새 모델이 사용됩니다.
    """
    data = request.json or {}
    model_name = data.get('model')
    
    if not model_name:
        return jsonify({'success': False, 'message': '모델명이 필요합니다'}), 400
    
    if model_name not in AppConfig.AVAILABLE_MODELS:
        available = ', '.join(AppConfig.AVAILABLE_MODELS.keys())
        return jsonify({
            'success': False, 
            'message': f'지원하지 않는 모델입니다. 사용 가능: {available}'
        }), 400
    
    try:
        # 현재 폴더 정보 유지하면서 모델만 변경
        current_folder = getattr(qa_system, 'current_folder', '')
        offline_mode = getattr(qa_system, 'offline_mode', False)
        local_model_path = getattr(qa_system, 'local_model_path', '')
        
        # 모델 로드 (기존 load_model 메서드 활용)
        result = qa_system.load_model(model_name, offline_mode, local_model_path)
        
        if result.success:
            logger.info(f"모델 변경 완료: {model_name}")
            return jsonify({
                'success': True, 
                'message': f'모델이 {model_name}(으)로 변경되었습니다',
                'current': model_name,
                'note': '검색 성능 최적화를 위해 문서 재처리를 권장합니다'
            })
        else:
            return jsonify({'success': False, 'message': result.message}), 500
    except Exception as e:
        logger.error(f"모델 변경 오류: {e}")
        return jsonify({'success': False, 'message': f'모델 변경 실패: {str(e)}'}), 500


@system_bp.route('/verify_password', methods=['POST'])
def verify_password():
    """관리자 비밀번호 확인"""
    data = request.json
    password = data.get('password')
    # TODO: settings.json에서 비밀번호 로드 및 검증
    # Mock 비밀번호 대신 settings.json의 password_hash 사용
    import hashlib
    
    # 설정 파일에서 해시 로드
    config_dir = os.path.join(get_app_directory(), 'config')
    settings_path = os.path.join(config_dir, 'settings.json')
    
    try:
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        stored_hash = settings.get('admin_password_hash', '')
    except Exception:
        stored_hash = ''
    
    # 입력된 비밀번호의 SHA256 해시와 비교
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    if stored_hash and password_hash == stored_hash:
        session['admin_authenticated'] = True
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': '비밀번호가 일치하지 않습니다'})

@system_bp.route('/status', methods=['GET'])
def status():
    """서버 상태 반환"""
    return jsonify({
        'ready': qa_system.is_ready,
        'loading': qa_system.is_loading,
        'load_progress': qa_system.load_progress,
        'load_error': qa_system.load_error,
        'model': qa_system.model_name
    })
    
@system_bp.route('/init', methods=['POST'])
def init_server():
    """검색 경로 초기화 및 인덱싱 시작"""
    data = request.json
    folder_path = data.get('folder_path')
    if not folder_path or not os.path.exists(folder_path):
        return jsonify({'success': False, 'message': '유효하지 않은 경로입니다'})
        
    res = qa_system.initialize(folder_path, force_reindex=data.get('reindex', False))
    return jsonify(res.to_dict())

@system_bp.route('/stats', methods=['GET'])
def stats():
    """문서 및 시스템 통계 반환
    
    Returns:
        - 문서 통계 (파일 수, 청크 수, 크기)
        - 캐시 통계 (히트율, 크기)
        - 검색 큐 통계 (활성, 처리, 거부)
        - Rate limiter 통계
        - 메모리 사용량 (선택)
    """
    from app.services.search import rate_limiter, search_queue
    from app.utils import MemoryMonitor
    
    doc_stats = qa_system.get_stats()
    
    # 캐시 통계
    cache_stats = {}
    if hasattr(qa_system, '_search_cache'):
        cache_stats = qa_system._search_cache.get_stats()
    
    # 검색 큐 통계
    queue_stats = search_queue.get_stats()
    
    # Rate limiter 통계
    rate_stats = rate_limiter.get_stats()
    
    # 메모리 사용량 (include_memory 파라미터가 true일 때만)
    memory_stats = {}
    if request.args.get('include_memory', 'false').lower() == 'true':
        memory_stats = MemoryMonitor.get_memory_usage()
    
    return jsonify({
        'success': True,
        'documents': doc_stats,
        'cache': cache_stats,
        'search_queue': queue_stats,
        'rate_limiter': rate_stats,
        'memory': memory_stats if memory_stats else None
    })
    
@system_bp.route('/health')
def health():
    """헬스 체크 (상세 정보 포함)"""
    from app.services.search import search_queue
    
    queue_stats = search_queue.get_stats()
    
    # 기본 상태
    status_info = {
        'status': 'ok',
        'model_ready': qa_system.is_ready,
        'model_loading': qa_system.is_loading,
        'active_searches': queue_stats.get('active', 0)
    }
    
    # 부하 경고
    if queue_stats.get('active', 0) > 8:
        status_info['warning'] = 'High search load'
    
    return jsonify(status_info)

# ============================================================================
# Sync API (v2.0) - 폴더 동기화 관련 엔드포인트
# ============================================================================

@system_bp.route('/sync/status', methods=['GET'])
def sync_status():
    """동기화 상태 반환"""
    return jsonify({
        'success': True,
        'is_syncing': qa_system.is_loading,
        'current_folder': getattr(qa_system, 'current_folder', ''),
        'progress': qa_system.load_progress,
        'error': qa_system.load_error
    })

@system_bp.route('/sync/start', methods=['POST'])
def sync_start():
    """폴더 동기화 시작 (초기화 및 인덱싱)
    
    Security: Path traversal 공격 방지를 위한 경로 검증 포함
    """
    try:
        data = request.json or {}
        folder_path = data.get('folder')
        
        if not folder_path:
            return jsonify({'success': False, 'message': '폴더 경로가 필요합니다'}), 400
        
        # Path Traversal 공격 방지: 경로 정규화 및 검증
        try:
            # 경로 정규화 (.. 등 해석)
            normalized_path = os.path.normpath(os.path.realpath(folder_path))
            
            # 위험한 경로 패턴 차단
            dangerous_patterns = ['..', '//']
            if any(p in folder_path for p in dangerous_patterns):
                logger.warning(f"의심스러운 경로 패턴 감지: {folder_path}")
                return jsonify({'success': False, 'message': '유효하지 않은 경로 형식입니다'}), 400
            
            folder_path = normalized_path
        except (ValueError, OSError) as e:
            logger.warning(f"경로 정규화 실패: {folder_path} - {e}")
            return jsonify({'success': False, 'message': '유효하지 않은 경로입니다'}), 400
        
        if not os.path.exists(folder_path):
            return jsonify({'success': False, 'message': f'폴더를 찾을 수 없습니다: {folder_path}'}), 404
        
        if not os.path.isdir(folder_path):
            return jsonify({'success': False, 'message': f'디렉토리가 아닙니다: {folder_path}'}), 400
        
        logger.info(f"동기화 시작: {folder_path}")
        res = qa_system.initialize(folder_path, force_reindex=data.get('force', False))
        return jsonify(res.to_dict())
        
    except json.JSONDecodeError:
        return jsonify({'success': False, 'message': '잘못된 JSON 형식입니다'}), 400
    except Exception as e:
        logger.error(f"동기화 시작 오류: {e}")
        return jsonify({'success': False, 'message': f'동기화 오류: {str(e)}'}), 500

@system_bp.route('/sync/stop', methods=['POST'])
def sync_stop():
    """동기화 중지 (현재는 미구현 - graceful shutdown 필요)"""
    # TODO: ThreadPoolExecutor 작업 취소 로직 구현
    return jsonify({
        'success': True,
        'message': '동기화 중지 요청됨 (현재 작업은 완료될 때까지 계속됩니다)'
    })

# ============================================================================
# Process API - 문서 재처리 엔드포인트
# ============================================================================

@system_bp.route('/process', methods=['POST'])
def process_documents():
    """문서 재처리 (현재 폴더 강제 재인덱싱)"""
    try:
        current_folder = getattr(qa_system, 'current_folder', '')
        
        if not current_folder:
            return jsonify({'success': False, 'message': '초기화된 폴더가 없습니다. 먼저 폴더를 선택해주세요.'})
        
        if not os.path.exists(current_folder):
            return jsonify({'success': False, 'message': f'폴더를 찾을 수 없습니다: {current_folder}'})
        
        logger.info(f"문서 재처리 시작: {current_folder}")
        res = qa_system.initialize(current_folder, force_reindex=True)
        return jsonify(res.to_dict())
        
    except Exception as e:
        logger.error(f"문서 재처리 오류: {e}")
        return jsonify({'success': False, 'message': f'재처리 오류: {str(e)}'}), 500

# ============================================================================
# Admin API - 관리자 인증 엔드포인트
# ============================================================================

@system_bp.route('/admin/check', methods=['GET'])
def admin_check():
    """관리자 인증 상태 확인"""
    is_authenticated = session.get('admin_authenticated', False)
    return jsonify({
        'success': True,
        'authenticated': is_authenticated
    })

@system_bp.route('/admin/auth', methods=['POST'])
def admin_auth():
    """관리자 로그인"""
    import hashlib
    
    data = request.json or {}
    password = data.get('password', '')
    
    if not password:
        return jsonify({'success': False, 'message': '비밀번호가 필요합니다'}), 400
    
    # 설정 파일에서 해시 로드
    config_dir = os.path.join(get_app_directory(), 'config')
    settings_path = os.path.join(config_dir, 'settings.json')
    
    try:
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        stored_hash = settings.get('admin_password_hash', '')
    except Exception:
        # 설정 파일이 없거나 해시가 없으면 기본 관리자 비밀번호 허용
        stored_hash = ''
    
    # 입력된 비밀번호의 SHA256 해시와 비교
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    # 저장된 해시가 없으면 기본 비밀번호 'admin' 허용
    if not stored_hash:
        default_hash = hashlib.sha256('admin'.encode()).hexdigest()
        if password_hash == default_hash:
            session['admin_authenticated'] = True
            logger.info("관리자 로그인 성공 (기본 비밀번호)")
            return jsonify({'success': True, 'message': '로그인 성공 (기본 비밀번호 사용 중 - 변경 권장)'})
    
    if stored_hash and password_hash == stored_hash:
        session['admin_authenticated'] = True
        logger.info("관리자 로그인 성공")
        return jsonify({'success': True, 'message': '로그인 성공'})
    
    logger.warning("관리자 로그인 실패")
    return jsonify({'success': False, 'message': '비밀번호가 일치하지 않습니다'}), 401

@system_bp.route('/admin/logout', methods=['POST'])
def admin_logout():
    """관리자 로그아웃"""
    session.pop('admin_authenticated', None)
    logger.info("관리자 로그아웃")
    return jsonify({'success': True, 'message': '로그아웃 완료'})
