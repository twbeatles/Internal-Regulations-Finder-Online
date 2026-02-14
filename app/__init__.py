# -*- coding: utf-8 -*-
import os
import json
import secrets
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
    # Prefer explicit secret key for stable sessions; fall back to a persisted key file.
    secret = os.environ.get('FLASK_SECRET_KEY') or os.environ.get('SECRET_KEY')
    if not secret:
        config_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config'))
        key_path = os.path.join(config_dir, 'secret_key')
        try:
            os.makedirs(config_dir, exist_ok=True)
            if os.path.exists(key_path):
                with open(key_path, 'r', encoding='utf-8') as f:
                    secret = f.read().strip()
            if not secret:
                secret = secrets.token_hex(32)
                with open(key_path, 'w', encoding='utf-8') as f:
                    f.write(secret)
                logger.info(f"✅ 세션 시크릿 키 생성: {key_path}")
        except Exception as e:
            # Last resort: ephemeral key (sessions reset on restart)
            secret = secrets.token_hex(32)
            logger.warning(f"세션 시크릿 키 파일 생성 실패, 임시 키 사용: {e}")
    app.secret_key = secret

    # Safer default cookie settings for session-based admin auth.
    app.config.setdefault('SESSION_COOKIE_HTTPONLY', True)
    app.config.setdefault('SESSION_COOKIE_SAMESITE', 'Lax')
    
    # JSON Provider 설정
    if CustomJSONProvider:
        app.json = CustomJSONProvider(app)
    
    # ========================================================================
    # 응답 압축 설정 (v2.6.1 성능 최적화)
    # ========================================================================
    try:
        from flask_compress import Compress
        compress = Compress()
        
        # 압축 설정
        app.config['COMPRESS_MIMETYPES'] = [
            'text/html', 'text/css', 'text/javascript',
            'application/javascript', 'application/json',
            'text/xml', 'application/xml'
        ]
        app.config['COMPRESS_LEVEL'] = 6  # 압축 레벨 (1-9, 6이 균형)
        app.config['COMPRESS_MIN_SIZE'] = getattr(AppConfig, 'COMPRESS_MIN_SIZE', 500)
        
        compress.init_app(app)
        logger.info("✅ Gzip 응답 압축 활성화")
    except ImportError:
        logger.warning("flask-compress 미설치 - 응답 압축 비활성화 (pip install flask-compress)")
        
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
