# -*- coding: utf-8 -*-
from flask import Blueprint, jsonify, request, session
from app.config import AppConfig
from app.services.search import qa_system
import os

system_bp = Blueprint('system', __name__)

@system_bp.route('/models', methods=['GET'])
def get_models():
    """사용 가능한 모델 목록 반환"""
    return jsonify({
        'success': True,
        'models': list(AppConfig.AVAILABLE_MODELS.keys()),
        'current': getattr(qa_system, 'model_name', AppConfig.DEFAULT_MODEL)
    })

@system_bp.route('/verify_password', methods=['POST'])
def verify_password():
    """관리자 비밀번호 확인"""
    data = request.json
    password = data.get('password')
    # TODO: settings.json에서 비밀번호 로드 및 검증
    # 현재는 간단한 하드코딩 또는 설정이 없는 상태로 가정
    if password == "admin1234": # Mock password
        session['admin_authenticated'] = True
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': '비밀번호가 일치하지 않습니다'})

@system_bp.route('/status', methods=['GET'])
def status():
    """서버 상태 반환"""
    return jsonify({
        'is_ready': qa_system.is_ready,
        'is_loading': qa_system.is_loading,
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
    """문서 통계 반환"""
    return jsonify(qa_system.get_stats())
    
@system_bp.route('/health')
def health():
    """헬스 체크"""
    return jsonify({'status': 'ok'})
