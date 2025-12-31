# -*- coding: utf-8 -*-
from app import app
from app.config import AppConfig
from app.utils import logger, setup_logger

if __name__ == '__main__':
    setup_logger()
    logger.info(f"규정 검색기 서버 v2.0 시작")
    logger.info(f"http://localhost:{AppConfig.SERVER_PORT}")
    
    try:
        from waitress import serve
        logger.info(f"Waitress 서버 시작 (Port: {AppConfig.SERVER_PORT}, Threads: {AppConfig.SERVER_THREADS})")
        serve(app, host='0.0.0.0', port=AppConfig.SERVER_PORT, threads=AppConfig.SERVER_THREADS)
    except ImportError:
        logger.warning("Waitress가 설치되지 않았습니다. 개발용 서버로 실행합니다.")
        app.run(
            host='0.0.0.0',
            port=AppConfig.SERVER_PORT,
            debug=True,
            use_reloader=False, 
            threaded=True
        )
