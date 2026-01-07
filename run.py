# -*- coding: utf-8 -*-
import os
import json
import threading
from app import app
from app.config import AppConfig
from app.utils import logger, setup_logger, get_app_directory
from app.services.search import qa_system

def initialize_server():
    """서버 초기화 - 모델 로드 및 문서 처리 (백그라운드)"""
    logger.info("🚀 서버 초기화 시작...")
    
    try:
        # 설정 파일 로드
        config_dir = os.path.join(get_app_directory(), 'config')
        settings_path = os.path.join(config_dir, 'settings.json')
        
        folder = ''
        offline_mode = False
        local_model_path = ''
        
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
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
    logger.info(f"📚 규정 검색기 서버 v2.2 시작")
    logger.info(f"🌐 http://localhost:{AppConfig.SERVER_PORT}")
    logger.info(f"⚙️ 관리자: http://localhost:{AppConfig.SERVER_PORT}/admin")
    
    # 백그라운드에서 모델 로드 시작 (서버 시작 지연 방지)
    init_thread = threading.Thread(target=initialize_server, daemon=True)
    init_thread.start()
    
    try:
        from waitress import serve
        logger.info(f"🚀 Waitress 서버 시작 (Port: {AppConfig.SERVER_PORT}, Threads: {AppConfig.SERVER_THREADS})")
        serve(app, host='0.0.0.0', port=AppConfig.SERVER_PORT, threads=AppConfig.SERVER_THREADS)
    except ImportError:
        logger.warning("⚠️ Waitress가 설치되지 않았습니다. 개발용 서버로 실행합니다.")
        app.run(
            host='0.0.0.0',
            port=AppConfig.SERVER_PORT,
            debug=True,
            use_reloader=False, 
            threaded=True
        )
