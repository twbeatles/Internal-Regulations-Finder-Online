# -*- coding: utf-8 -*-
from flask import Blueprint, render_template, session, request, current_app

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """메인 검색 페이지"""
    return render_template('index.html')

@main_bp.route('/admin')
def admin():
    """관리자 페이지"""
    return render_template('admin.html')


@main_bp.route('/setup')
def setup():
    """MCP 클라이언트 연결 안내"""
    from app.services.settings_store import get_settings_store

    settings = get_settings_store().load()
    mcp_cfg = settings.get("mcp") if isinstance(settings.get("mcp"), dict) else {}
    host = str(mcp_cfg.get("host") or "127.0.0.1")
    port = int(mcp_cfg.get("port") or 8081)
    return render_template(
        "setup.html",
        mcp_url=f"http://{host}:{port}/sse",
        mcp_enabled=bool(mcp_cfg.get("enabled")),
    )
