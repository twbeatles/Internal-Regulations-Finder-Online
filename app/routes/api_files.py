# -*- coding: utf-8 -*-
import os
import threading
from flask import Blueprint, jsonify, request, send_file
from werkzeug.utils import secure_filename
from app.services.search import qa_system
from app.utils import logger, FileUtils, TaskResult, get_app_directory
from app.config import AppConfig

files_bp = Blueprint('files', __name__)
file_lock = threading.Lock()

@files_bp.route('/files', methods=['GET'])
def list_files():
    """로드된 파일 목록 반환"""
    return jsonify({
        'success': True,
        'files': qa_system.get_file_infos()
    })

@files_bp.route('/files/<path:filename>', methods=['DELETE'])
def delete_file(filename):
    """파일 삭제"""
    # 실제 파일 삭제 로직은 간단 구현. 운영 환경에서는 보안 주의.
    # 여기서는 filename이 basename이라고 가정하고 current_folder에서 삭제 시도
    # 또는 full path를 인자로 받아야 함. server.py는 list에서 full path를 줬음.
    # 하지만 URL 파라미터로 full path는 어려움.
    # server.py 구현: filename 파라미터 받아서 file_infos에서 찾아서 삭제.
    
    with file_lock:
        target_path = None
        for fp, info in qa_system.file_infos.items():
            if os.path.basename(fp) == filename:
                target_path = fp
                break
        
        if not target_path:
            return jsonify({'success': False, 'message': '파일을 찾을 수 없습니다'}), 404
            
        try:
            os.remove(target_path)
            if target_path in qa_system.file_infos:
                del qa_system.file_infos[target_path]
            # 인덱스에서도 제거해야 하나, 여기서는 재인덱싱 전까지는 남아있을 수 있음
            # qa_system.remove_document(target_path) 기능 필요
            return jsonify({'success': True, 'message': '파일이 삭제되었습니다'})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500

@files_bp.route('/upload', methods=['POST'])
def upload_file():
    """파일 업로드"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': '파일이 없습니다'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': '파일명이 비어있습니다'}), 400
        
    if not FileUtils.allowed_file(file.filename):
        return jsonify({'success': False, 'message': '지원하지 않는 파일 형식입니다'}), 400
        
    upload_folder = qa_system.current_folder or os.path.join(get_app_directory(), AppConfig.UPLOAD_FOLDER)
    os.makedirs(upload_folder, exist_ok=True)
    
    filename = secure_filename(file.filename)
    save_path = os.path.join(upload_folder, filename)
    
    with file_lock:
        try:
            file.save(save_path)
            # 업로드 후 인덱싱 추가 (단일 파일 처리)
            # qa_system.process_single_file(save_path) 필요
            # 일단 메시지만 리턴
            return jsonify({'success': True, 'message': '파일 업로드 성공 (재인덱싱 필요)'})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500

# ============================================================================
# 태그 관리 (v2.0)
# ============================================================================
@files_bp.route('/tags', methods=['GET'])
def get_tags():
    file_path = request.args.get('file')
    if file_path:
        tags = qa_system.tag_manager.get_tags(file_path)
        return jsonify({'success': True, 'tags': tags})
    else:
        # 전체 태그 통계
        return jsonify({'success': True, 'tags': []}) # Not implemented fully

@files_bp.route('/tags', methods=['POST'])
def add_tag():
    data = request.json
    file_path = data.get('file')
    tag = data.get('tag')
    if qa_system.tag_manager.add_tag(file_path, tag):
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': '태그 추가 실패'})

@files_bp.route('/tags', methods=['DELETE'])
def remove_tag():
    data = request.json
    file_path = data.get('file')
    tag = data.get('tag')
    if qa_system.tag_manager.remove_tag(file_path, tag):
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': '태그 삭제 실패'})

# ============================================================================
# 개정 관리 (v2.0)
# ============================================================================
@files_bp.route('/revisions', methods=['GET'])
def get_revisions():
    filename = request.args.get('filename') # Just filename or full path? TagManager uses full path, RevisionTracker uses filename usually
    # RevisionTracker implementation uses filename
    history = qa_system.revision_tracker.get_history(filename)
    return jsonify({'success': True, 'history': history})

@files_bp.route('/revisions', methods=['POST'])
def save_revision():
    data = request.json
    filename = data.get('filename')
    content = data.get('content')
    comment = data.get('comment', '')
    try:
        res = qa_system.revision_tracker.save_revision(filename, content, comment)
        return jsonify({'success': True, 'revision': res})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@files_bp.route('/compare', methods=['POST'])
def compare_docs():
    data = request.json
    # v1 (content) vs v2 (content) or version IDs
    # Simply using DocumentComparator directly here
    from app.services.document import DocumentComparator
    text1 = data.get('text1', '')
    text2 = data.get('text2', '')
    
    comp = DocumentComparator()
    res = comp.compare(text1, text2)
    return jsonify({'success': True, 'comparison': res})

@files_bp.route('/files/<path:filename>/versions', methods=['GET'])
def get_file_versions(filename):
    """파일의 버전 히스토리 조회"""
    history = qa_system.revision_tracker.get_history(filename)
    return jsonify({
        'success': True,
        'revisions': history
    })

@files_bp.route('/files/<path:filename>/versions/<version>', methods=['GET'])
def get_file_version_content(filename, version):
    """특정 버전의 내용 조회"""
    content = qa_system.revision_tracker.get_revision(filename, version)
    if content is None:
        return jsonify({'success': False, 'message': '버전을 찾을 수 없습니다'}), 404
    return jsonify({
        'success': True,
        'version': version,
        'content': content
    })

@files_bp.route('/files/<path:filename>/versions/compare', methods=['GET'])
def compare_file_versions(filename):
    """두 버전 간 비교"""
    v1 = request.args.get('v1')
    v2 = request.args.get('v2')
    
    if not v1 or not v2:
        return jsonify({'success': False, 'message': '버전 파라미터가 필요합니다 (v1, v2)'}), 400
    
    diff_result = qa_system.revision_tracker.compare_versions(filename, v1, v2)
    
    if diff_result is None:
        return jsonify({'success': False, 'message': '버전을 비교할 수 없습니다'}), 404
    
    return jsonify({
        'success': True,
        'diff': diff_result
    })

