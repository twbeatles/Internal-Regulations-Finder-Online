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
