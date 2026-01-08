# -*- coding: utf-8 -*-
import os
import threading
from flask import Blueprint, jsonify, request, send_file
from werkzeug.utils import secure_filename
from app.services.search import qa_system
from app.utils import logger, FileUtils, TaskResult, get_app_directory, api_error, api_success
from app.config import AppConfig
from app.constants import ErrorMessages, HttpStatus
from app.exceptions import DocumentNotFoundError, DocumentError

files_bp = Blueprint('files', __name__)
file_lock = threading.Lock()
LOCK_TIMEOUT = 30  # 파일 잠금 타임아웃 (초)

def acquire_file_lock(timeout=LOCK_TIMEOUT):
    """파일 잠금 획득 (타임아웃 지원)
    
    Returns:
        bool: 잠금 획득 성공 여부
    """
    return file_lock.acquire(timeout=timeout)

def _find_file_path(filename: str) -> str:
    """파일명으로 전체 경로 찾기
    
    Args:
        filename: 찾을 파일명 (basename)
        
    Returns:
        전체 경로
        
    Raises:
        DocumentNotFoundError: 파일을 찾을 수 없는 경우
    """
    for fp in qa_system.file_infos.keys():
        if os.path.basename(fp) == filename:
            return fp
    raise DocumentNotFoundError(filename)

@files_bp.route('/files', methods=['GET'])
def list_files():
    """로드된 파일 목록 반환"""
    return jsonify({
        'success': True,
        'files': qa_system.get_file_infos()
    })

@files_bp.route('/files/names', methods=['GET'])
def list_file_names():
    """파일 이름 목록만 반환 (파일 필터용)"""
    try:
        file_names = list(qa_system.file_infos.keys())
        # basename만 반환
        names = [os.path.basename(fp) for fp in file_names]
        return jsonify({
            'success': True,
            'names': names
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e),
            'names': []
        })

@files_bp.route('/files/all', methods=['DELETE'])
def delete_all_files():
    """모든 로드된 파일 일괄 삭제 (인덱스 및 캐시 초기화)"""
    if not acquire_file_lock():
        return jsonify({
            'success': False,
            'message': '서버가 바쁩니다. 잠시 후 다시 시도해주세요.'
        }), HttpStatus.SERVICE_UNAVAILABLE
    
    try:
        deleted_count = len(qa_system.file_infos)
        file_paths = list(qa_system.file_infos.keys())
        
        # 실제 파일 삭제 (uploads 폴더 내 파일만)
        upload_dir = os.path.join(get_app_directory(), 'uploads')
        deleted_files = []
        failed_files = []
        
        for fp in file_paths:
            try:
                # uploads 폴더 내 파일인 경우에만 실제 삭제
                if fp.startswith(upload_dir) and os.path.exists(fp):
                    os.remove(fp)
                    deleted_files.append(os.path.basename(fp))
                else:
                    deleted_files.append(os.path.basename(fp))  # 인덱스에서만 제거
            except Exception as e:
                failed_files.append(f"{os.path.basename(fp)}: {str(e)}")
        
        # 인덱스 및 캐시 초기화
        qa_system.file_infos.clear()
        qa_system.documents = []
        qa_system.vectorstore = None
        if hasattr(qa_system, '_search_cache'):
            qa_system._search_cache.clear()
        if hasattr(qa_system, '_bm25'):
            qa_system._bm25 = None
        
        logger.info(f"일괄 삭제 완료: {deleted_count}개 파일 제거")
        
        result = {
            'success': True,
            'message': f'{deleted_count}개 파일이 삭제되었습니다',
            'deleted_count': deleted_count,
            'deleted_files': deleted_files
        }
        
        if failed_files:
            result['failed_files'] = failed_files
            result['message'] += f' (실패: {len(failed_files)}개)'
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"일괄 삭제 오류: {e}")
        return jsonify({
            'success': False,
            'message': f'일괄 삭제 실패: {str(e)}'
        }), HttpStatus.INTERNAL_ERROR
    finally:
        file_lock.release()

@files_bp.route('/files/<path:filename>', methods=['DELETE'])
def delete_file(filename):
    """파일 삭제"""
    with file_lock:
        try:
            target_path = _find_file_path(filename)
            
            os.remove(target_path)
            if target_path in qa_system.file_infos:
                del qa_system.file_infos[target_path]
            
            logger.info(f"파일 삭제 완료: {filename}")
            return jsonify(api_success("파일이 삭제되었습니다"))
            
        except DocumentNotFoundError:
            return api_error("파일을 찾을 수 없습니다", status_code=HttpStatus.NOT_FOUND)
        except PermissionError as e:
            logger.error(f"파일 삭제 권한 오류: {filename} - {e}")
            return api_error("파일 삭제 권한이 없습니다", status_code=HttpStatus.FORBIDDEN)
        except Exception as e:
            logger.error(f"파일 삭제 오류: {filename} - {e}")
            return api_error(f"파일 삭제 실패: {str(e)}", status_code=HttpStatus.INTERNAL_ERROR)

@files_bp.route('/upload', methods=['POST'])
def upload_file():
    """파일 업로드 및 자동 인덱싱"""
    if 'file' not in request.files:
        return jsonify({
            'success': False, 
            'message': ErrorMessages.FILE_NOT_FOUND
        }), HttpStatus.BAD_REQUEST
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            'success': False, 
            'message': ErrorMessages.FILE_NAME_EMPTY
        }), HttpStatus.BAD_REQUEST
        
    if not FileUtils.allowed_file(file.filename):
        ext = os.path.splitext(file.filename)[1]
        return jsonify({
            'success': False, 
            'message': f'{ErrorMessages.FILE_TYPE_NOT_SUPPORTED}: {ext}'
        }), HttpStatus.BAD_REQUEST
        
    upload_folder = qa_system.current_folder or os.path.join(get_app_directory(), AppConfig.UPLOAD_FOLDER)
    os.makedirs(upload_folder, exist_ok=True)
    
    filename = secure_filename(file.filename)
    # 한글 파일명 보존 (secure_filename이 제거할 수 있음)
    if not filename or filename == '_':
        filename = file.filename
    
    save_path = os.path.join(upload_folder, filename)
    
    with file_lock:
        try:
            file.save(save_path)
            logger.info(f"파일 저장 완료: {save_path}")
            
            # 업로드 후 즉시 인덱싱 처리
            result = qa_system.process_single_file(save_path)
            
            if result.success:
                return jsonify({
                    'success': True, 
                    'message': result.message,
                    'data': result.data
                })
            else:
                return jsonify({
                    'success': True,  # 파일은 저장됨
                    'message': f'파일 저장 완료 (인덱싱 실패: {result.message})',
                    'indexed': False
                })
        except IOError as e:
            logger.error(f"파일 저장 오류: {e}")
            return jsonify({
                'success': False, 
                'message': f'파일 저장 실패: {str(e)}'
            }), HttpStatus.INTERNAL_ERROR
        except Exception as e:
            logger.error(f"파일 업로드 오류: {e}")
            return jsonify({
                'success': False, 
                'message': f'{ErrorMessages.FILE_UPLOAD_FAILED}: {str(e)}'
            }), HttpStatus.INTERNAL_ERROR

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

