# -*- coding: utf-8 -*-
import os
import zipfile
import shutil
from pathlib import Path
from typing import Any, Optional, Tuple
from flask import Blueprint, jsonify, request, send_file
from app.services.search import qa_system
from app.services.files import (
    acquire_file_lock,
    file_lock,
    file_path_resolver,
    is_source_delete_allowed,
    preview_service,
)
from app.utils import logger, FileUtils, get_app_directory, api_error, api_success
from app.config import AppConfig
from app.constants import ErrorMessages, HttpStatus
from app.exceptions import DocumentNotFoundError
from app.auth import admin_required

files_bp = Blueprint('files', __name__)


def _normalize_optional_str(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _get_json_payload() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def _get_request_bool(name: str, default=False) -> bool:
    query_value = request.args.get(name)
    if query_value is not None:
        return _to_bool(query_value, default)
    payload = request.get_json(silent=True) or {}
    if isinstance(payload, dict) and name in payload:
        return _to_bool(payload.get(name), default)
    form_value = request.form.get(name)
    if form_value is not None:
        return _to_bool(form_value, default)
    return default


def _parse_limit(name: str, default: int, minimum: int = 1) -> Tuple[Optional[int], Optional[str]]:
    raw = request.form.get(name)
    if raw is None or str(raw).strip() == "":
        value = default
    else:
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError):
            return None, f"{name}는 정수여야 합니다"
    if value < minimum:
        return None, f"{name}는 {minimum} 이상이어야 합니다"
    return value, None


@files_bp.route('/files', methods=['GET'])
def list_files():
    """로드된 파일 목록 반환"""
    return jsonify(api_success(files=qa_system.get_file_infos()))

@files_bp.route('/files/names', methods=['GET'])
def list_file_names():
    """파일 이름 목록만 반환 (파일 필터용)"""
    try:
        refs = [file_path_resolver.build_file_ref(fp) for fp in qa_system.file_infos.keys()]
        names = [ref['name'] for ref in refs]
        return jsonify(api_success(files=refs, names=names))
    except Exception as e:
        return api_error(f'파일 목록 조회 실패: {str(e)}')

@files_bp.route('/files/all', methods=['DELETE'])
@admin_required
def delete_all_files():
    """모든 로드된 파일 일괄 삭제 (기본: index_only)"""
    if not acquire_file_lock():
        return jsonify({
            'success': False,
            'message': '서버가 바쁩니다. 잠시 후 다시 시도해주세요.'
        }), HttpStatus.SERVICE_UNAVAILABLE
    
    try:
        delete_source = _get_request_bool('delete_source', False)
        deletion_policy = 'delete_source' if delete_source else 'index_only'
        deleted_count = len(qa_system.file_infos)
        file_paths = list(qa_system.file_infos.keys())
        
        # 실제 파일 삭제는 delete_source=true일 때만 수행
        deleted_files = []
        failed_files = []
        deleted_source_count = 0
        
        for fp in file_paths:
            try:
                p = Path(fp).resolve()
                if delete_source:
                    if is_source_delete_allowed(p, qa_system):
                        if p.exists():
                            p.unlink()
                            deleted_source_count += 1
                    else:
                        failed_files.append(f"{os.path.basename(fp)}: 허용된 경로 외 원본 파일 삭제 불가")
                deleted_files.append(os.path.basename(fp))
            except Exception as e:
                failed_files.append(f"{os.path.basename(fp)}: {str(e)}")
        
        qa_system.clear_index(preserve_folder=True)
        
        logger.info(f"일괄 삭제 완료: {deleted_count}개 파일 제거")
        logger.info(
            "파일 일괄 삭제 정책 적용: policy=%s, deleted_from_index=%s, deleted_source=%s",
            deletion_policy, deleted_count, deleted_source_count
        )
        
        result = {
            'success': True,
            'message': f'{deleted_count}개 파일이 삭제되었습니다',
            'deleted_count': deleted_count,
            'deleted_files': deleted_files,
            'deletion_policy': deletion_policy,
            'deleted_from_index': deleted_count,
            'deleted_source': deleted_source_count
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
    return _delete_file_impl(filename=filename)

@files_bp.route('/files/by-id/<file_id>', methods=['DELETE'])
@admin_required
def delete_file_by_id(file_id):
    """file_id 기준 파일 삭제 및 인덱스 정리"""
    return _delete_file_impl(file_id=file_id)

def _delete_file_impl(filename: str | None = None, file_id: str | None = None):
    with file_lock:
        try:
            target_path, resolved_name, resolved_id = file_path_resolver.resolve(file_id=file_id, filename=filename)
            delete_source = _get_request_bool('delete_source', False)
            deletion_policy = 'delete_source' if delete_source else 'index_only'
            deleted_source = False
            source_delete_error = ""

            # 1) 원본 파일 삭제 (옵션)
            if delete_source:
                path_obj = Path(target_path).resolve()
                if is_source_delete_allowed(path_obj, qa_system):
                    try:
                        if path_obj.exists():
                            path_obj.unlink()
                            deleted_source = True
                    except Exception as e:
                        source_delete_error = str(e)
                else:
                    source_delete_error = "허용된 경로 외 원본 파일은 삭제할 수 없습니다"
            
            deleted_from_index = qa_system.remove_file_from_index(target_path, resolved_name, resolved_id)
            
            logger.info(
                "파일 삭제 완료: name=%s file_id=%s policy=%s deleted_source=%s deleted_from_index=%s",
                resolved_name, resolved_id, deletion_policy, deleted_source, deleted_from_index
            )
            payload = api_success(
                "파일 삭제 처리 완료",
                file_id=resolved_id,
                filename=resolved_name,
                deletion_policy=deletion_policy,
                deleted_source=deleted_source,
                deleted_from_index=deleted_from_index
            )
            if source_delete_error:
                payload['source_delete_error'] = source_delete_error
            return jsonify(payload)
            
        except DocumentNotFoundError:
            return api_error("파일을 찾을 수 없습니다", status_code=HttpStatus.NOT_FOUND)
        except PermissionError as e:
            logger.error(f"파일 삭제 권한 오류: {filename or file_id} - {e}")
            return api_error("파일 삭제 권한이 없습니다", status_code=HttpStatus.FORBIDDEN)
        except Exception as e:
            logger.error(f"파일 삭제 오류: {filename or file_id} - {e}")
            return api_error(f"파일 삭제 실패: {str(e)}", status_code=HttpStatus.INTERNAL_ERROR)

@files_bp.route('/upload', methods=['POST'])
@admin_required
def upload_file():
    """파일 업로드 및 자동 인덱싱"""
    if 'file' not in request.files:
        return jsonify({
            'success': False, 
            'message': ErrorMessages.FILE_NOT_FOUND
        }), HttpStatus.BAD_REQUEST
        
    file = request.files['file']
    incoming_filename = _normalize_optional_str(file.filename)
    if incoming_filename is None:
        return jsonify({
            'success': False, 
            'message': ErrorMessages.FILE_NAME_EMPTY
        }), HttpStatus.BAD_REQUEST
        
    if not FileUtils.allowed_file(incoming_filename):
        ext = os.path.splitext(incoming_filename)[1]
        return jsonify({
            'success': False, 
            'message': f'{ErrorMessages.FILE_TYPE_NOT_SUPPORTED}: {ext}'
        }), HttpStatus.BAD_REQUEST
        
    upload_folder = qa_system.current_folder or os.path.join(get_app_directory(), AppConfig.UPLOAD_FOLDER)
    os.makedirs(upload_folder, exist_ok=True)
    
    original_name = incoming_filename
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
                uploaded_file_id = FileUtils.make_file_id(str(save_path))
                return jsonify({
                    'success': True, 
                    'message': result.message,
                    'data': result.data,
                    'file_id': uploaded_file_id,
                    'filename': save_path.name,
                    'original_filename': original_name
                })
            else:
                uploaded_file_id = FileUtils.make_file_id(str(save_path))
                return jsonify({
                    'success': True,  # 파일은 저장됨
                    'message': f'파일 저장 완료 (인덱싱 실패: {result.message})',
                    'indexed': False,
                    'file_id': uploaded_file_id,
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

@files_bp.route('/upload/folder', methods=['POST'])
@admin_required
def upload_folder():
    """ZIP 폴더 업로드 및 파일별 자동 인덱싱"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': ErrorMessages.FILE_NOT_FOUND}), HttpStatus.BAD_REQUEST

    file = request.files['file']
    if not file or not file.filename:
        return jsonify({'success': False, 'message': ErrorMessages.FILE_NAME_EMPTY}), HttpStatus.BAD_REQUEST

    if not str(file.filename).lower().endswith('.zip'):
        return jsonify({'success': False, 'message': 'ZIP 파일만 업로드할 수 있습니다'}), HttpStatus.BAD_REQUEST

    upload_folder = qa_system.current_folder or os.path.join(get_app_directory(), AppConfig.UPLOAD_FOLDER)
    os.makedirs(upload_folder, exist_ok=True)
    upload_root = Path(upload_folder).resolve()

    max_entries, err = _parse_limit('max_entries', AppConfig.ZIP_MAX_ENTRIES, minimum=1)
    if err or max_entries is None:
        return jsonify({'success': False, 'message': err}), HttpStatus.BAD_REQUEST
    max_uncompressed_bytes, err = _parse_limit(
        'max_uncompressed_bytes', AppConfig.ZIP_MAX_UNCOMPRESSED_BYTES, minimum=1
    )
    if err or max_uncompressed_bytes is None:
        return jsonify({'success': False, 'message': err}), HttpStatus.BAD_REQUEST
    max_single_file_bytes, err = _parse_limit(
        'max_single_file_bytes', AppConfig.ZIP_MAX_SINGLE_FILE_BYTES, minimum=1
    )
    if err or max_single_file_bytes is None:
        return jsonify({'success': False, 'message': err}), HttpStatus.BAD_REQUEST

    success_items = []
    failed_items = []
    skipped_items = []

    try:
        with file_lock:
            with zipfile.ZipFile(file.stream) as zf:
                entries = [info for info in zf.infolist() if not info.is_dir()]
                if len(entries) > max_entries:
                    logger.warning(
                        "ZIP 업로드 차단: max_entries 초과 (%s > %s)",
                        len(entries), max_entries
                    )
                    return jsonify({
                        'success': False,
                        'message': f'ZIP 항목 수 제한 초과: {len(entries)} > {max_entries}',
                        'reason': 'max_entries_exceeded',
                        'limits': {
                            'max_entries': max_entries,
                            'max_uncompressed_bytes': max_uncompressed_bytes,
                            'max_single_file_bytes': max_single_file_bytes
                        }
                    }), 413

                total_uncompressed = 0
                for info in entries:
                    file_size = int(getattr(info, 'file_size', 0) or 0)
                    if file_size > max_single_file_bytes:
                        logger.warning(
                            "ZIP 업로드 차단: max_single_file_bytes 초과 (%s > %s) member=%s",
                            file_size, max_single_file_bytes, info.filename
                        )
                        return jsonify({
                            'success': False,
                            'message': f'ZIP 단일 파일 제한 초과: {info.filename}',
                            'reason': 'max_single_file_bytes_exceeded',
                            'limits': {
                                'max_entries': max_entries,
                                'max_uncompressed_bytes': max_uncompressed_bytes,
                                'max_single_file_bytes': max_single_file_bytes
                            }
                        }), 413
                    total_uncompressed += file_size
                    if total_uncompressed > max_uncompressed_bytes:
                        logger.warning(
                            "ZIP 업로드 차단: max_uncompressed_bytes 초과 (%s > %s)",
                            total_uncompressed, max_uncompressed_bytes
                        )
                        return jsonify({
                            'success': False,
                            'message': 'ZIP 전체 압축해제 용량 제한 초과',
                            'reason': 'max_uncompressed_bytes_exceeded',
                            'limits': {
                                'max_entries': max_entries,
                                'max_uncompressed_bytes': max_uncompressed_bytes,
                                'max_single_file_bytes': max_single_file_bytes
                            }
                        }), 413

                actual_total_written = 0
                for info in entries:
                    if info.is_dir():
                        continue

                    member_name = (info.filename or "").replace("\\", "/")
                    member_parts = [p for p in member_name.split('/') if p]

                    # path traversal / absolute path 차단
                    if member_name.startswith('/') or any(part == '..' for part in member_parts):
                        failed_items.append(f"{member_name}: 유효하지 않은 ZIP 경로")
                        continue

                    base_name = os.path.basename(member_name)
                    safe_name = FileUtils.sanitize_upload_filename(base_name)
                    if not FileUtils.allowed_file(safe_name):
                        skipped_items.append(base_name)
                        continue

                    save_path = (upload_root / safe_name).resolve()
                    if upload_root not in save_path.parents:
                        failed_items.append(f"{member_name}: 경로 검증 실패")
                        continue

                    if save_path.exists():
                        stem = save_path.stem
                        suffix = save_path.suffix
                        for i in range(1, 1000):
                            candidate = (upload_root / f"{stem}_{i}{suffix}").resolve()
                            if not candidate.exists():
                                save_path = candidate
                                break
                        else:
                            failed_items.append(f"{base_name}: 동일 파일명이 너무 많습니다")
                            continue

                    try:
                        written = 0
                        with zf.open(info) as src, open(save_path, 'wb') as dst:
                            while True:
                                chunk = src.read(64 * 1024)
                                if not chunk:
                                    break
                                written += len(chunk)
                                if written > max_single_file_bytes:
                                    raise ValueError(
                                        f"실제 해제 크기 제한 초과: {written} > {max_single_file_bytes}"
                                    )
                                actual_total_written += len(chunk)
                                if actual_total_written > max_uncompressed_bytes:
                                    raise ValueError("전체 압축해제 용량 제한 초과")
                                dst.write(chunk)
                    except Exception as e:
                        try:
                            if save_path.exists():
                                save_path.unlink()
                        except Exception:
                            pass
                        failed_items.append(f"{base_name}: 압축 해제 실패")
                        logger.warning(f"ZIP 멤버 해제 실패: {member_name} - {e}")
                        continue

                    result = qa_system.process_single_file(str(save_path))
                    if result.success:
                        success_items.append({
                            'file_id': FileUtils.make_file_id(str(save_path)),
                            'filename': save_path.name
                        })
                    else:
                        failed_items.append(f"{save_path.name}: 인덱싱 실패 ({result.message})")

            if hasattr(qa_system, '_search_cache'):
                qa_system._search_cache.clear()

        return jsonify({
            'success': len(success_items) > 0 and len(failed_items) == 0,
            'message': f"처리 완료: 성공 {len(success_items)}개, 실패 {len(failed_items)}개, 스킵 {len(skipped_items)}개",
            'processed_count': len(success_items) + len(failed_items),
            'success_count': len(success_items),
            'failed_count': len(failed_items),
            'skipped_count': len(skipped_items),
            'limits': {
                'max_entries': max_entries,
                'max_uncompressed_bytes': max_uncompressed_bytes,
                'max_single_file_bytes': max_single_file_bytes
            },
            'success_items': success_items,
            'failed_items': failed_items,
            'skipped_items': skipped_items
        })
    except zipfile.BadZipFile:
        return jsonify({'success': False, 'message': '손상되었거나 유효하지 않은 ZIP 파일입니다'}), HttpStatus.BAD_REQUEST
    except Exception as e:
        logger.error(f"ZIP 업로드 오류: {e}")
        return jsonify({'success': False, 'message': f'ZIP 업로드 실패: {e}'}), HttpStatus.INTERNAL_ERROR

@files_bp.route('/files/<path:filename>/versions', methods=['GET'])
def get_file_versions(filename):
    """파일의 버전 히스토리 조회"""
    return _get_file_versions_impl(filename=filename)

@files_bp.route('/files/by-id/<file_id>/versions', methods=['GET'])
def get_file_versions_by_id(file_id):
    """file_id 기준 파일의 버전 히스토리 조회"""
    return _get_file_versions_impl(file_id=file_id)

def _get_file_versions_impl(filename: str | None = None, file_id: str | None = None):
    try:
        target_path, resolved_name, resolved_id = file_path_resolver.resolve(file_id=file_id, filename=filename)
        history = qa_system.revision_tracker.get_history(resolved_id, legacy_key=resolved_name)
        return jsonify({
            'success': True,
            'history': history,      # compatibility
            'revisions': history,
            'file_id': resolved_id,
            'filename': resolved_name
        })
    except DocumentNotFoundError:
        return jsonify({'success': False, 'message': '파일을 찾을 수 없습니다'}), HttpStatus.NOT_FOUND

@files_bp.route('/files/<path:filename>/versions/<version>', methods=['GET'])
def get_file_version_content(filename, version):
    """특정 버전의 내용 조회"""
    return _get_file_version_content_impl(version, filename=filename)

@files_bp.route('/files/by-id/<file_id>/versions/<version>', methods=['GET'])
def get_file_version_content_by_id(file_id, version):
    """file_id 기준 특정 버전 내용 조회"""
    return _get_file_version_content_impl(version, file_id=file_id)

def _get_file_version_content_impl(version: str, filename: str | None = None, file_id: str | None = None):
    try:
        target_path, resolved_name, resolved_id = file_path_resolver.resolve(file_id=file_id, filename=filename)
        content = qa_system.revision_tracker.get_revision(resolved_id, version, legacy_key=resolved_name)
        if content is None:
            return jsonify({'success': False, 'message': '버전을 찾을 수 없습니다'}), 404
        return jsonify({
            'success': True,
            'file_id': resolved_id,
            'filename': resolved_name,
            'version': version,
            'content': content
        })
    except DocumentNotFoundError:
        return jsonify({'success': False, 'message': '파일을 찾을 수 없습니다'}), HttpStatus.NOT_FOUND

@files_bp.route('/files/<path:filename>/versions/compare', methods=['GET'])
def compare_file_versions(filename):
    """두 버전 간 비교"""
    return _compare_file_versions_impl(filename=filename)

@files_bp.route('/files/by-id/<file_id>/versions/compare', methods=['GET'])
def compare_file_versions_by_id(file_id):
    """file_id 기준 두 버전 간 비교"""
    return _compare_file_versions_impl(file_id=file_id)

def _compare_file_versions_impl(filename: str | None = None, file_id: str | None = None):
    v1 = request.args.get('v1')
    v2 = request.args.get('v2')
    
    if not v1 or not v2:
        return jsonify({'success': False, 'message': '버전 파라미터가 필요합니다 (v1, v2)'}), 400
    
    try:
        target_path, resolved_name, resolved_id = file_path_resolver.resolve(file_id=file_id, filename=filename)
    except DocumentNotFoundError:
        return jsonify({'success': False, 'message': '파일을 찾을 수 없습니다'}), HttpStatus.NOT_FOUND

    diff_result = qa_system.revision_tracker.compare_versions(resolved_id, v1, v2, legacy_key=resolved_name)
    
    if diff_result is None:
        return jsonify({'success': False, 'message': '버전을 비교할 수 없습니다'}), 404
    
    return jsonify({
        'success': True,
        'file_id': resolved_id,
        'diff': diff_result
    })

# ============================================================================
# 추가 엔드포인트 (Frontend 호환용)
# ============================================================================

@files_bp.route('/files/<path:filename>/preview', methods=['GET'])
def get_file_preview(filename):
    """파일 미리보기 (텍스트 일부 반환)"""
    return _get_file_preview_impl(filename=filename)

@files_bp.route('/files/by-id/<file_id>/preview', methods=['GET'])
def get_file_preview_by_id(file_id):
    """file_id 기준 파일 미리보기 (텍스트 일부 반환)"""
    return _get_file_preview_impl(file_id=file_id)

def _get_file_preview_impl(filename: str | None = None, file_id: str | None = None):
    """공용 파일 미리보기 구현"""
    try:
        target_path, resolved_name, resolved_id = file_path_resolver.resolve(file_id=file_id, filename=filename)
        length = request.args.get('length', 2000, type=int) or 2000
        length = max(200, min(length, 10000))

        try:
            mtime = os.path.getmtime(target_path)
        except OSError:
            mtime = 0

        cache_key = f"{target_path}|{mtime}|{length}"
        cached = preview_service.get(cache_key)
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

            cached_details = getattr(qa_system, 'file_details', {}).get(target_path, {})
            metadata = dict(cached_details.get('metadata', {}))
            if not metadata:
                metadata = {
                    'title': os.path.splitext(resolved_name)[0],
                    'file_name': resolved_name,
                    'source_path': target_path,
                    'source_format': 'txt',
                }
            diagnostics = dict(cached_details.get('diagnostics', {})) or {
                'engine_used': 'txt',
                'fallback_used': False,
                'quality_score': 1.0 if content.strip() else 0.0,
                'warnings': [],
            }
            payload = preview_service.build_payload(
                resolved_name,
                target_path,
                content,
                length,
                metadata=metadata,
                tables=list(cached_details.get('tables', [])),
                diagnostics=diagnostics,
            )
        else:
            extracted = preview_service.extractor.extract_with_details(target_path)
            if extracted.error:
                return jsonify({'success': False, 'message': extracted.error}), 400
            payload = preview_service.build_payload(
                resolved_name,
                target_path,
                extracted.text,
                length,
                metadata=extracted.metadata,
                tables=extracted.table_dicts(),
                diagnostics=extracted.diagnostics,
            )

        preview_service.set(cache_key, payload)
        return jsonify(payload)
    except DocumentNotFoundError:
        return api_error("파일을 찾을 수 없습니다", status_code=HttpStatus.NOT_FOUND)
    except Exception as e:
        logger.error(f"파일 미리보기 오류: {filename or file_id} - {e}")
        return api_error(f"미리보기 실패: {str(e)}", status_code=HttpStatus.INTERNAL_ERROR)

@files_bp.route('/files/<path:filename>/download', methods=['GET'])
def download_file(filename):
    """파일 다운로드 (legacy filename route)"""
    return _download_file_impl(filename=filename)

@files_bp.route('/files/by-id/<file_id>/download', methods=['GET'])
def download_file_by_id(file_id):
    """파일 다운로드 (file_id route)"""
    return _download_file_impl(file_id=file_id)

def _download_file_impl(filename: str | None = None, file_id: str | None = None):
    try:
        target_path, resolved_name, resolved_id = file_path_resolver.resolve(file_id=file_id, filename=filename)
        if not os.path.exists(target_path) or not os.path.isfile(target_path):
            return api_error("파일을 찾을 수 없습니다", status_code=HttpStatus.NOT_FOUND)
        return send_file(
            target_path,
            as_attachment=True,
            download_name=resolved_name
        )
    except DocumentNotFoundError:
        return api_error("파일을 찾을 수 없습니다", status_code=HttpStatus.NOT_FOUND)
    except PermissionError:
        return api_error("파일 접근 권한이 없습니다", status_code=HttpStatus.FORBIDDEN)
    except Exception as e:
        logger.error(f"파일 다운로드 오류: {filename or file_id} - {e}")
        return api_error(f"파일 다운로드 실패: {str(e)}", status_code=HttpStatus.INTERNAL_ERROR)

@files_bp.route('/files/structure', methods=['GET'])
def get_file_structure():
    """파일 구조 및 태그 정보 반환"""
    try:
        files = []
        for path, info in qa_system.file_infos.items():
            filename = os.path.basename(path)
            file_id = FileUtils.make_file_id(path)
            tags = qa_system.tag_manager.get_tags(file_id, fallback_keys=[filename])
            files.append({
                'file_id': file_id,
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
