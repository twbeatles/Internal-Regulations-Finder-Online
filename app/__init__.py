# -*- coding: utf-8 -*-
import os
import json
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from app.config import AppConfig
from app.utils import logger
from app.services.db import db

# NumPy 호환 JSON Provider
try:
    from flask.json.provider import DefaultJSONProvider
    class CustomJSONProvider(DefaultJSONProvider):
        def default(self, obj):
            try:
                import numpy as np
                if isinstance(obj, (np.integer, np.floating, np.bool_)):
                    return obj.item()
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
            except ImportError:
                pass
            return super().default(obj)
except ImportError:
    CustomJSONProvider = None

def create_app():
    # 템플릿과 정적 파일 경로는 프로젝트 루트 기준
    app = Flask(__name__, 
                template_folder='../templates', 
                static_folder='../static')
    
    app.config.from_object(AppConfig)
    app.secret_key = os.urandom(24)
    
    # JSON Provider 설정
    if CustomJSONProvider:
        app.json = CustomJSONProvider(app)
        
    CORS(app, supports_credentials=True)
    
    # DB 초기화
    db.init_db()
    
    # 전역 에러 핸들러
    @app.errorhandler(404)
    def not_found(e):
        if request.path.startswith('/api/'):
            return jsonify({'success': False, 'message': 'API 엔드포인트를 찾을 수 없습니다'}), 404
        return render_template('index.html'), 404

    @app.errorhandler(500)
    def server_error(e):
        logger.error(f"서버 내부 오류: {e}")
        if request.path.startswith('/api/'):
            return jsonify({'success': False, 'message': '서버 내부 오류가 발생했습니다', 'error': str(e)}), 500
        return "서버 오류발생", 500

    # Blueprint 등록 (Lazy Import to avoid circular refs)
    try:
        from app.routes.main_routes import main_bp
        from app.routes.api_search import search_bp
        from app.routes.api_files import files_bp
        from app.routes.api_system import system_bp

        app.register_blueprint(main_bp)
        app.register_blueprint(search_bp, url_prefix='/api')
        app.register_blueprint(files_bp, url_prefix='/api')
        app.register_blueprint(system_bp, url_prefix='/api')
    except ImportError as e:
        logger.warning(f"블루프린트 등록 중 오류 (아직 파일이 없을 수 있음): {e}")

    return app

# 전역 App 객체 (run.py에서 사용)
app = create_app()
