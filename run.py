# -*- coding: utf-8 -*-
import os
import signal
import sys
import threading
from app import app
from app.config import AppConfig
from app.utils import logger, setup_logger
from app.services.search import qa_system
from app.services.db import db
from app.services.settings_store import get_settings_store

# Graceful Shutdown 플래그
_shutdown_event = threading.Event()

def create_default_settings():
    """기본 설정 파일 생성"""
    store = get_settings_store()
    store.ensure_exists()
    return store.paths.settings_json

def graceful_shutdown(signum, frame):
    """서버 종료 시 리소스 정리"""
    logger.info("🛑 종료 신호 수신, Graceful Shutdown 시작...")
    _shutdown_event.set()
    
    # QA System 정리
    try:
        qa_system.cleanup()
        logger.info("✅ QA System 정리 완료")
    except Exception as e:
        logger.warning(f"QA System 정리 중 오류: {e}")
    
    # DB 연결 정리
    try:
        db.close_all()
        logger.info("✅ DB 연결 정리 완료")
    except Exception as e:
        logger.warning(f"DB 정리 중 오류: {e}")
    
    logger.info("👋 서버 종료 완료")
    sys.exit(0)

def initialize_server():
    """서버 초기화 - 모델 로드 및 문서 처리 (백그라운드)"""
    logger.info("🚀 서버 초기화 시작...")
    
    try:
        # 설정 파일 로드 (없으면 생성)
        settings_path = create_default_settings()
        
        folder = ''
        offline_mode = False
        local_model_path = ''
        
        if os.path.exists(settings_path):
            try:
                settings = get_settings_store().load()
                
                folder = settings.get('folder', '')
                offline_mode = settings.get('offline_mode', False)
                local_model_path = settings.get('local_model_path', '')
                
                logger.info(f"📋 설정 로드 완료 - 오프라인 모드: {offline_mode}")
            except Exception as e:
                logger.warning(f"설정 파일 로드 실패: {e}")
        
        # 오프라인 모드 설정
        AppConfig.OFFLINE_MODE = offline_mode
        AppConfig.LOCAL_MODEL_PATH = local_model_path
        
        # 모델 로드 (항상 시도)
        logger.info(f"🤖 AI 모델 로드 시작: {AppConfig.DEFAULT_MODEL}")
        result = qa_system.load_model(
            AppConfig.DEFAULT_MODEL,
            offline_mode=offline_mode,
            local_model_path=local_model_path if local_model_path else None
        )
        
        if result.success:
            logger.info(f"✅ AI 모델 로드 완료: {AppConfig.DEFAULT_MODEL}")
        else:
            logger.warning(f"⚠️ AI 모델 로드 실패: {result.message}")
        
        # 폴더가 설정되어 있으면 문서 처리
        if folder and os.path.exists(folder):
            logger.info(f"📂 문서 폴더 초기화: {folder}")
            qa_system.initialize(folder)
        else:
            logger.info("📁 문서 폴더가 설정되지 않았습니다. 관리자 페이지에서 설정해주세요.")
        
        logger.info("✅ 서버 초기화 완료")
        
    except Exception as e:
        logger.error(f"❌ 서버 초기화 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    setup_logger()
    app_env = os.environ.get("APP_ENV", getattr(AppConfig, "APP_ENV", "development")).strip().lower()
    is_production = app_env == "production"
    
    # 종료 시그널 핸들러 등록
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)
    
    logger.info(f"📚 {AppConfig.APP_NAME} v{AppConfig.APP_VERSION} 시작")
    logger.info(f"🌐 http://localhost:{AppConfig.SERVER_PORT}")
    logger.info(f"⚙️ 관리자: http://localhost:{AppConfig.SERVER_PORT}/admin")
    
    # 백그라운드에서 모델 로드 시작 (서버 시작 지연 방지)
    init_thread = threading.Thread(target=initialize_server, daemon=True)
    init_thread.start()
    
    try:
        from waitress import serve
        logger.info(f"🚀 Waitress 서버 시작 (Port: {AppConfig.SERVER_PORT}, Threads: {AppConfig.SERVER_THREADS})")
        serve(app, host=AppConfig.SERVER_HOST, port=AppConfig.SERVER_PORT, threads=AppConfig.SERVER_THREADS)
    except ImportError:
        if is_production:
            logger.error("❌ APP_ENV=production 환경에서는 waitress가 필수입니다. 서버를 종료합니다.")
            sys.exit(1)

        logger.warning("⚠️ Waitress가 설치되지 않아 개발용 서버로 fallback 실행합니다.")
        app.run(
            host=AppConfig.SERVER_HOST,
            port=AppConfig.SERVER_PORT,
            debug=False,
            use_reloader=False,
            threaded=True
        )
