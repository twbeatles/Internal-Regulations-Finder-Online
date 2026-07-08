# -*- coding: utf-8 -*-
from flask import Blueprint, jsonify, request

from app.auth import admin_required
from app.constants import HttpStatus
from app.routes.files_request import get_json_payload, normalize_optional_str
from app.services.files import file_path_resolver
from app.services.search import qa_system

revisions_bp = Blueprint('revisions', __name__)


@revisions_bp.route('/revisions', methods=['GET'])
def get_revisions():
    file_id = normalize_optional_str(request.args.get('file_id'))
    filename = normalize_optional_str(request.args.get('filename'))
    try:
        target_path, resolved_name, resolved_id = file_path_resolver.resolve(
            file_id=file_id, filename=filename
        )
    except Exception:
        return jsonify({'success': True, 'history': [], 'revisions': []})

    history = qa_system.revision_tracker.get_history(resolved_id, legacy_key=resolved_name)
    return jsonify({'success': True, 'history': history, 'revisions': history, 'file_id': resolved_id})


@revisions_bp.route('/revisions', methods=['POST'])
@admin_required
def save_revision():
    data = get_json_payload()
    file_id = normalize_optional_str(data.get('file_id'))
    filename = normalize_optional_str(data.get('filename'))
    content = data.get('content')
    comment = normalize_optional_str(data.get('comment', '')) or ''
    if not isinstance(content, str) or not content:
        return jsonify({'success': False, 'message': '저장할 내용(content)이 필요합니다'}), HttpStatus.BAD_REQUEST
    try:
        target_path, resolved_name, resolved_id = file_path_resolver.resolve(
            file_id=file_id, filename=filename
        )
        res = qa_system.revision_tracker.save_revision(
            file_key=resolved_id,
            content=content,
            note=comment,
            display_name=resolved_name,
            legacy_key=resolved_name,
        )
        return jsonify({'success': True, 'revision': res})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@revisions_bp.route('/compare', methods=['POST'])
def compare_docs():
    data = get_json_payload()
    # v1 (content) vs v2 (content) or version IDs
    # Simply using DocumentComparator directly here
    from app.services.document import DocumentComparator
    text1 = data.get('text1', '')
    text2 = data.get('text2', '')

    comp = DocumentComparator()
    res = comp.compare(text1, text2)
    return jsonify({'success': True, 'comparison': res})