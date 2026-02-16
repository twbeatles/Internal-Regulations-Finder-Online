# -*- coding: utf-8 -*-
import os
import threading
from collections import OrderedDict
from pathlib import Path
from flask import Blueprint, jsonify, request, send_file
from werkzeug.utils import secure_filename
from app.services.search import qa_system
from app.services.document import DocumentExtractor
from app.utils import logger, FileUtils, TaskResult, get_app_directory, api_error, api_success
from app.config import AppConfig
from app.constants import ErrorMessages, HttpStatus
from app.exceptions import DocumentNotFoundError, DocumentError
from app.auth import admin_required

files_bp = Blueprint('files', __name__)
file_lock = threading.Lock()
LOCK_TIMEOUT = 30  # 파일 잠금 타임아웃 (초)
_preview_extractor = DocumentExtractor()
_preview_cache_lock = threading.Lock()
_preview_cache: OrderedDict = OrderedDict()
_PREVIEW_CACHE_MAX_SIZE = 128

def acquire_file_lock(timeout=LOCK_TIMEOUT):
    """파일 잠금 획득 (타임아웃 지원)
    
    Returns:
        bool: 잠금 획득 성공 여부
    """
    return file_lock.acquire(timeout=timeout)

def _preview_cache_get(cache_key: str):
    with _preview_cache_lock:
        payload = _preview_cache.get(cache_key)
        if payload is not None:
            _preview_cache.move_to_end(cache_key)
        return payload

def _preview_cache_set(cache_key: str, payload: dict):
    with _preview_cache_lock:
        if cache_key in _preview_cache:
            del _preview_cache[cache_key]
        while len(_preview_cache) >= _PREVIEW_CACHE_MAX_SIZE:
            _preview_cache.popitem(last=False)
        _preview_cache[cache_key] = payload

def _build_preview_payload(filename: str, target_path: str, content: str, length: int) -> dict:
    text = content or ''
    preview = text[:length]
    truncated = len(text) > length
    info = qa_system.file_infos.get(target_path)
    return {
        'success': True,
        'filename': filename,
        'preview': preview,
        'content': preview,  # 프론트엔드 하위호환
        'total_length': len(text),
        'truncated': truncated,
        'is_truncated': truncated,  # 프론트엔드 하위호환
        'status': getattr(getattr(info, 'status', None), 'value', '완료'),
        'chunks': getattr(info, 'chunks', 0),
    }

def _find_file_path(filename: str) -> str:
    """파일명으로 전체 경로 찾기
    
    Args:
        filename: 찾을 파일명 (basename)
        
    Returns:
        전체 경로
        
    Raises:
        DocumentNotFoundError: 파일을 찾을 수 없는 경우
    """
    # Route uses <path:filename>; normalize to basename to avoid path injection.
    filename = os.path.basename((filename or "").replace("\\", "/"))
    if not filename:
        raise DocumentNotFoundError(filename)

    matches = []
    for fp in qa_system.file_infos.keys():
        if os.path.basename(fp) == filename:
            matches.append(fp)
    if not matches:
        raise DocumentNotFoundError(filename)
    if len(matches) > 1:
        raise DocumentError(f"동일한 파일명이 여러 개 존재합니다: {filename}")
    return matches[0]

@files_bp.route('/files', methods=['GET'])
def list_files():
    """로드된 파일 목록 반환"""
    return jsonify(api_success(files=qa_system.get_file_infos()))

@files_bp.route('/files/names', methods=['GET'])
def list_file_names():
    """파일 이름 목록만 반환 (파일 필터용)"""
    try:
        file_names = list(qa_system.file_infos.keys())
        # basename만 반환
        names = [os.path.basename(fp) for fp in file_names]
        return jsonify(api_success(names=names))
    except Exception as e:
        return api_error(f'파일 목록 조회 실패: {str(e)}')

@files_bp.route('/files/all', methods=['DELETE'])
@admin_required
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
                p = Path(fp).resolve()
                if Path(upload_dir).resolve() in p.parents and p.exists():
                    p.unlink()
                    deleted_files.append(os.path.basename(fp))
                else:
                    deleted_files.append(os.path.basename(fp))  # 인덱스에서만 제거
            except Exception as e:
                failed_files.append(f"{os.path.basename(fp)}: {str(e)}")
        
        # 인덱스 및 캐시 초기화
        qa_system.file_infos.clear()
        qa_system.documents = []
        qa_system.doc_meta = []
        qa_system.vector_store = None
        if hasattr(qa_system, '_search_cache'):
            qa_system._search_cache.clear()
        if hasattr(qa_system, 'bm25') and qa_system.bm25:
            qa_system.bm25.clear()
            qa_system.bm25 = None
        
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
@admin_required
def delete_file(filename):
    """파일 삭제 및 인덱스 정리"""
    with file_lock:
        try:
            target_path = _find_file_path(filename)
            
            # 1. 실제 파일 삭제
            Path(target_path).unlink()
            
            # 2. file_infos에서 제거
            if target_path in qa_system.file_infos:
                del qa_system.file_infos[target_path]
            
            # 3. documents 및 doc_meta에서 해당 파일 관련 항목 제거
            if qa_system.documents and qa_system.doc_meta:
                indices_to_remove = [
                    i for i, meta in enumerate(qa_system.doc_meta)
                    if meta.get('path') == target_path or meta.get('source') == filename
                ]
                # 역순으로 삭제 (인덱스 밀림 방지)
                for idx in reversed(indices_to_remove):
                    if idx < len(qa_system.documents):
                        del qa_system.documents[idx]
                    if idx < len(qa_system.doc_meta):
                        del qa_system.doc_meta[idx]
                
                # BM25 인덱스 재구축 필요
                if indices_to_remove and hasattr(qa_system, '_build_bm25'):
                    qa_system._build_bm25()
            
            # 4. 캐시 무효화
            if hasattr(qa_system, '_search_cache'):
                qa_system._search_cache.invalidate_by_file(filename)
            
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
    
    original_name = file.filename
    filename = FileUtils.sanitize_upload_filename(original_name)

    # Ensure allowed extension after sanitization (defense-in-depth).
    if not FileUtils.allowed_file(filename):
        ext = os.path.splitext(filename)[1]
        return jsonify({'success': False, 'message': f'{ErrorMessages.FILE_TYPE_NOT_SUPPORTED}: {ext}'}), HttpStatus.BAD_REQUEST

    upload_root = Path(upload_folder).resolve()
    save_path = (upload_root / filename).resolve()
    if upload_root not in save_path.parents:
        return jsonify({'success': False, 'message': '유효하지 않은 파일명입니다'}), HttpStatus.BAD_REQUEST

    # Avoid overwriting existing file.
    if save_path.exists():
        stem = save_path.stem
        suffix = save_path.suffix
        for i in range(1, 1000):
            candidate = (upload_root / f"{stem}_{i}{suffix}").resolve()
            if not candidate.exists():
                save_path = candidate
                break
        else:
            return jsonify({'success': False, 'message': '동일 파일명이 너무 많습니다'}), HttpStatus.SERVICE_UNAVAILABLE
    
    with file_lock:
        try:
            file.save(str(save_path))
            logger.info(f"파일 저장 완료: {save_path}")
            
            # 업로드 후 즉시 인덱싱 처리
            result = qa_system.process_single_file(str(save_path))
            
            # 캐시 무효화 (새 파일이 검색에 즉시 반영되도록)
            if hasattr(qa_system, '_search_cache'):
                qa_system._search_cache.clear()
            
            if result.success:
                return jsonify({
                    'success': True, 
                    'message': result.message,
                    'data': result.data,
                    'filename': save_path.name,
                    'original_filename': original_name
                })
            else:
                return jsonify({
                    'success': True,  # 파일은 저장됨
                    'message': f'파일 저장 완료 (인덱싱 실패: {result.message})',
                    'indexed': False,
                    'filename': save_path.name,
                    'original_filename': original_name
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
        # Store tags by basename to avoid leaking absolute paths.
        normalized = os.path.basename((file_path or "").replace("\\", "/"))
        tags = qa_system.tag_manager.get_tags(normalized)
        return jsonify({'success': True, 'tags': tags})
    else:
        # 전체 태그 통계
        stats = qa_system.tag_manager.get_tag_stats()
        return jsonify({'success': True, 'tags': qa_system.tag_manager.get_all_tags(), 'stats': stats})

@files_bp.route('/tags', methods=['POST'])
@admin_required
def add_tag():
    data = request.json or {}
    file_path = data.get('file')
    tag = data.get('tag')
    
    # 입력 검증
    if not file_path or not isinstance(file_path, str):
        return jsonify({'success': False, 'message': '파일 경로가 필요합니다'}), HttpStatus.BAD_REQUEST
    if not tag or not isinstance(tag, str) or not tag.strip():
        return jsonify({'success': False, 'message': '태그가 필요합니다'}), HttpStatus.BAD_REQUEST
    
    normalized = os.path.basename((file_path or "").replace("\\", "/"))
    if qa_system.tag_manager.add_tag(normalized, tag.strip()):
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': '태그 추가 실패'})

@files_bp.route('/tags', methods=['DELETE'])
@admin_required
def remove_tag():
    data = request.json or {}
    file_path = data.get('file')
    tag = data.get('tag')
    
    # 입력 검증
    if not file_path or not isinstance(file_path, str):
        return jsonify({'success': False, 'message': '파일 경로가 필요합니다'}), HttpStatus.BAD_REQUEST
    if not tag or not isinstance(tag, str):
        return jsonify({'success': False, 'message': '태그가 필요합니다'}), HttpStatus.BAD_REQUEST
    
    normalized = os.path.basename((file_path or "").replace("\\", "/"))
    if qa_system.tag_manager.remove_tag(normalized, tag):
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

# ============================================================================
# 추가 엔드포인트 (Frontend 호환용)
# ============================================================================

@files_bp.route('/files/<path:filename>/preview', methods=['GET'])
def get_file_preview(filename):
    """파일 미리보기 (텍스트 일부 반환)"""
    try:
        target_path = _find_file_path(filename)
        length = request.args.get('length', 2000, type=int) or 2000
        length = max(200, min(length, 10000))

        try:
            mtime = os.path.getmtime(target_path)
        except OSError:
            mtime = 0

        cache_key = f"{target_path}|{mtime}|{length}"
        cached = _preview_cache_get(cache_key)
        if cached is not None:
            return jsonify(cached)

        ext = os.path.splitext(target_path)[1].lower()

        # TXT는 앞부분만 빠르게 읽어서 응답 (미리보기 목적).
        if ext == '.txt':
            try:
                with open(target_path, 'r', encoding='utf-8', errors='ignore') as f:
                    head = f.read(length + 1)
                    if len(head) <= length:
                        rest = f.read()
                        content = head + rest
                    else:
                        content = head
            except Exception as e:
                return jsonify({'success': False, 'message': f'텍스트 미리보기 실패: {e}'}), 400

            payload = _build_preview_payload(filename, target_path, content, length)
        else:
            content, error = _preview_extractor.extract(target_path)
            if error:
                return jsonify({'success': False, 'message': error}), 400
            payload = _build_preview_payload(filename, target_path, content, length)

        _preview_cache_set(cache_key, payload)
        return jsonify(payload)
    except DocumentNotFoundError:
        return api_error("파일을 찾을 수 없습니다", status_code=HttpStatus.NOT_FOUND)
    except Exception as e:
        logger.error(f"파일 미리보기 오류: {filename} - {e}")
        return api_error(f"미리보기 실패: {str(e)}", status_code=HttpStatus.INTERNAL_ERROR)

@files_bp.route('/files/structure', methods=['GET'])
def get_file_structure():
    """파일 구조 및 태그 정보 반환"""
    try:
        files = []
        for path, info in qa_system.file_infos.items():
            filename = os.path.basename(path)
            tags = qa_system.tag_manager.get_tags(path)
            files.append({
                'name': filename,
                'path': path,
                'size': info.size,
                'chunks': info.chunks,
                'status': info.status.value if hasattr(info.status, 'value') else str(info.status),
                'tags': tags
            })
        
        return jsonify({
            'success': True,
            'files': files
        })
    except Exception as e:
        logger.error(f"파일 구조 조회 오류: {e}")
        return jsonify({'success': False, 'message': str(e), 'files': []})

@files_bp.route('/tags/set', methods=['POST'])
def set_file_tags():
    """파일 태그 일괄 설정 (덮어쓰기)"""
    data = request.json or {}
    filename = data.get('filename')
    tags = data.get('tags', [])
    
    if not filename or not isinstance(filename, str):
        return jsonify({'success': False, 'message': '파일명이 필요합니다'}), HttpStatus.BAD_REQUEST
    
    if not isinstance(tags, list):
        return jsonify({'success': False, 'message': '태그는 배열이어야 합니다'}), HttpStatus.BAD_REQUEST
    
    try:
        # 파일 경로 찾기
        file_path = _find_file_path(filename)
        qa_system.tag_manager.set_tags(file_path, tags)
        return jsonify({'success': True, 'message': f'{len(tags)}개 태그 설정 완료'})
    except DocumentNotFoundError:
        return jsonify({'success': False, 'message': '파일을 찾을 수 없습니다'}), HttpStatus.NOT_FOUND
    except Exception as e:
        logger.error(f"태그 설정 오류: {e}")
        return jsonify({'success': False, 'message': str(e)})

@files_bp.route('/tags/auto', methods=['POST'])
def auto_tag_file():
    """파일 자동 태그 추천"""
    data = request.json or {}
    filename = data.get('filename')
    
    if not filename or not isinstance(filename, str):
        return jsonify({'success': False, 'message': '파일명이 필요합니다'}), HttpStatus.BAD_REQUEST
    
    try:
        file_path = _find_file_path(filename)
        
        # 문서 내용 추출
        from app.services.document import DocumentExtractor
        extractor = DocumentExtractor()
        content, error = extractor.extract(file_path)
        
        if error:
            return jsonify({'success': False, 'message': f'문서 추출 실패: {error}'})
        
        # 자동 카테고리 추천
        suggested_tags = qa_system.tag_manager.auto_categorize(content, filename)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'suggested_tags': suggested_tags
        })
    except DocumentNotFoundError:
        return jsonify({'success': False, 'message': '파일을 찾을 수 없습니다'}), HttpStatus.NOT_FOUND
    except Exception as e:
        logger.error(f"자동 태그 오류: {e}")
        return jsonify({'success': False, 'message': str(e)})
