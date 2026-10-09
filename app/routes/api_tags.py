# -*- coding: utf-8 -*-
from flask import Blueprint, jsonify, request

from app.auth import admin_required
from app.constants import HttpStatus
from app.exceptions import DocumentNotFoundError
from app.routes.files_request import get_json_payload, normalize_optional_str
from app.services.files import file_path_resolver
from app.services.search import qa_system
from app.utils import logger

tags_bp = Blueprint('tags', __name__)


@tags_bp.route('/tags', methods=['GET'])
def get_tags():
    file_id = normalize_optional_str(request.args.get('file_id'))
    file_path = normalize_optional_str(request.args.get('file'))
    if file_id or file_path:
        try:
            target_path, resolved_name, resolved_id = file_path_resolver.resolve(
                file_id=file_id, filename=file_path
            )
            tags = qa_system.tag_manager.get_tags(
                resolved_id,
                fallback_keys=[resolved_name]
            )
        except Exception:
            tags = []
        return jsonify({'success': True, 'tags': tags})
    else:
        # 전체 태그 통계
        stats = qa_system.tag_manager.get_tag_stats()
        return jsonify({'success': True, 'tags': qa_system.tag_manager.get_all_tags(), 'stats': stats})


@tags_bp.route('/tags', methods=['POST'])
@admin_required
def add_tag():
    data = get_json_payload()
    file_id = normalize_optional_str(data.get('file_id'))
    file_path = normalize_optional_str(data.get('file'))
    tag = normalize_optional_str(data.get('tag'))

    # 입력 검증
    if not ((file_id and isinstance(file_id, str)) or (file_path and isinstance(file_path, str))):
        return jsonify({'success': False, 'message': 'file_id 또는 파일 경로가 필요합니다'}), HttpStatus.BAD_REQUEST
    if tag is None:
        return jsonify({'success': False, 'message': '태그가 필요합니다'}), HttpStatus.BAD_REQUEST

    try:
        target_path, resolved_name, resolved_id = file_path_resolver.resolve(
            file_id=file_id, filename=file_path
        )
    except Exception:
        return jsonify({'success': False, 'message': '파일을 찾을 수 없습니다'}), HttpStatus.NOT_FOUND

    if qa_system.tag_manager.add_tag(resolved_id, tag):
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': '태그 추가 실패'})


@tags_bp.route('/tags', methods=['DELETE'])
@admin_required
def remove_tag():
    data = get_json_payload()
    file_id = normalize_optional_str(data.get('file_id'))
    file_path = normalize_optional_str(data.get('file'))
    tag = normalize_optional_str(data.get('tag'))

    # 입력 검증
    if not ((file_id and isinstance(file_id, str)) or (file_path and isinstance(file_path, str))):
        return jsonify({'success': False, 'message': 'file_id 또는 파일 경로가 필요합니다'}), HttpStatus.BAD_REQUEST
    if tag is None:
        return jsonify({'success': False, 'message': '태그가 필요합니다'}), HttpStatus.BAD_REQUEST

    try:
        target_path, resolved_name, resolved_id = file_path_resolver.resolve(
            file_id=file_id, filename=file_path
        )
    except Exception:
        return jsonify({'success': False, 'message': '파일을 찾을 수 없습니다'}), HttpStatus.NOT_FOUND

    if qa_system.tag_manager.remove_tag(resolved_id, tag):
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': '태그 삭제 실패'})


@tags_bp.route('/tags/set', methods=['POST'])
@admin_required
def set_file_tags():
    """파일 태그 일괄 설정 (덮어쓰기)"""
    data = get_json_payload()
    file_id = normalize_optional_str(data.get('file_id'))
    filename = normalize_optional_str(data.get('filename'))
    tags = data.get('tags', [])

    if not ((file_id and isinstance(file_id, str)) or (filename and isinstance(filename, str))):
        return jsonify({'success': False, 'message': 'file_id 또는 파일명이 필요합니다'}), HttpStatus.BAD_REQUEST

    if not isinstance(tags, list):
        return jsonify({'success': False, 'message': '태그는 배열이어야 합니다'}), HttpStatus.BAD_REQUEST

    try:
        file_path, resolved_name, resolved_id = file_path_resolver.resolve(
            file_id=file_id, filename=filename
        )
        qa_system.tag_manager.set_tags(resolved_id, tags)
        return jsonify({'success': True, 'message': f'{len(tags)}개 태그 설정 완료'})
    except DocumentNotFoundError:
        return jsonify({'success': False, 'message': '파일을 찾을 수 없습니다'}), HttpStatus.NOT_FOUND
    except Exception as e:
        logger.error(f"태그 설정 오류: {e}")
        return jsonify({'success': False, 'message': str(e)})


@tags_bp.route('/tags/auto', methods=['POST'])
@admin_required
def auto_tag_file():
    """파일 자동 태그 추천"""
    data = get_json_payload()
    file_id = normalize_optional_str(data.get('file_id'))
    filename = normalize_optional_str(data.get('filename'))

    if not ((file_id and isinstance(file_id, str)) or (filename and isinstance(filename, str))):
        return jsonify({'success': False, 'message': 'file_id 또는 파일명이 필요합니다'}), HttpStatus.BAD_REQUEST

    try:
        file_path, resolved_name, resolved_id = file_path_resolver.resolve(
            file_id=file_id, filename=filename
        )

        # 문서 내용 추출
        from app.services.document import DocumentExtractor
        extractor = DocumentExtractor()
        content, error = extractor.extract(file_path)

        if error:
            return jsonify({'success': False, 'message': f'문서 추출 실패: {error}'})

        # 자동 카테고리 추천
        suggested_tags = qa_system.tag_manager.auto_categorize(content, resolved_name)

        return jsonify({
            'success': True,
            'file_id': resolved_id,
            'filename': resolved_name,
            'suggested_tags': suggested_tags,
            'tags': suggested_tags
        })
    except DocumentNotFoundError:
        return jsonify({'success': False, 'message': '파일을 찾을 수 없습니다'}), HttpStatus.NOT_FOUND
    except Exception as e:
        logger.error(f"자동 태그 오류: {e}")
        return jsonify({'success': False, 'message': str(e)})