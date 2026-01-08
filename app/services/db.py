# -*- coding: utf-8 -*-
import sqlite3
import os
import threading
from app.utils import get_app_directory, logger

class DBManager:
    """SQLite 데이터베이스 관리자 (싱글톤, 스레드 안전)
    
    Features:
        - 스레드별 연결 관리 (thread-local)
        - WAL 모드 및 성능 최적화
        - 연결 수 추적
    """
    _instance = None
    _lock = threading.Lock()
    _local = threading.local()
    _connection_count = 0  # 활성 연결 수 추적
    _MAX_CONNECTIONS = 50  # 최대 연결 수 경고 임계값
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DBManager, cls).__new__(cls)
                cls._instance.db_path = os.path.join(get_app_directory(), 'config', 'regulations.db')
                cls._instance._init_db()
        return cls._instance
    
    def _get_conn(self):
        """스레드별 DB 연결 반환 (연결 수 추적 포함)"""
        if not hasattr(self._local, 'conn'):
            with self._lock:
                DBManager._connection_count += 1
                if DBManager._connection_count > self._MAX_CONNECTIONS:
                    logger.warning(f"DB 연결 수 경고: {DBManager._connection_count}개 활성")
            
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
            # 성능 최적화 PRAGMA
            self._local.conn.execute('PRAGMA journal_mode=WAL')
            self._local.conn.execute('PRAGMA synchronous=NORMAL')
            self._local.conn.execute('PRAGMA temp_store=MEMORY')
            self._local.conn.execute('PRAGMA cache_size=-4000')  # 4MB 캐시
            self._local.conn.execute('PRAGMA mmap_size=268435456')  # 256MB mmap
        return self._local.conn
    
    def _init_db(self):
        """DB 테이블 초기화"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                
                # 태그 테이블
                c.execute('''CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(filename, tag)
                )''')
                c.execute('CREATE INDEX IF NOT EXISTS idx_tags_filename ON tags(filename)')
                c.execute('CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag)')
                
                # 리비전 테이블 (메타데이터)
                # 실제 콘텐츠는 파일 시스템이나 별도 저장소에 저장 권장되지만, 편의상 여기에 저장 가능
                # 여기서는 메타데이터와 파일 내용(대용량일 수 있음)을 분리할지 결정해야 함
                # 요구사항: JSON 대체를 원하므로, 텍스트 내용도 DB에 넣는 것이 관리 용이 (규정은 텍스트 위주이므로)
                c.execute('''CREATE TABLE IF NOT EXISTS revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    version TEXT NOT NULL,
                    content TEXT, 
                    comment TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(filename, version)
                )''')
                c.execute('CREATE INDEX IF NOT EXISTS idx_revisions_filename ON revisions(filename)')
                
                # 검색 히스토리
                c.execute('''CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
                c.execute('CREATE INDEX IF NOT EXISTS idx_history_query ON search_history(query)')
                # 복합 인덱스: 최근 검색 조회 최적화
                c.execute('CREATE INDEX IF NOT EXISTS idx_history_query_time ON search_history(query, timestamp DESC)')
                
                conn.commit()
        except Exception as e:
            logger.error(f"DB 초기화 실패: {e}")
    
    def init_db(self):
        """Public wrapper for DB initialization (호환성 유지)
        
        Note: DB는 이미 __new__에서 초기화되지만, 명시적 호출을 위한 메서드
        """
        # 이미 __new__에서 초기화됨, 재초기화가 필요한 경우만 처리
        if not os.path.exists(self.db_path):
            self._init_db()

    def execute(self, query, args=()):
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(query, args)
            conn.commit()
            return cur
        except sqlite3.OperationalError as e:
            # 연결 오류 시 재연결 시도
            logger.warning(f"DB 연결 오류, 재연결 시도: {e}")
            self.close()
            try:
                conn = self._get_conn()
                cur = conn.cursor()
                cur.execute(query, args)
                conn.commit()
                return cur
            except Exception as retry_e:
                logger.error(f"DB 재연결 후 실행 오류 ({query}): {retry_e}")
                raise
        except Exception as e:
            logger.error(f"DB 실행 오류 ({query}): {e}")
            raise

    def fetchall(self, query, args=()):
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(query, args)
            return cur.fetchall()
        except Exception as e:
            logger.error(f"DB 조회 오류 ({query}): {e}")
            return []
            
    def fetchone(self, query, args=()):
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(query, args)
            return cur.fetchone()
        except Exception as e:
            logger.error(f"DB 조회 오류 ({query}): {e}")
            return None
            
    def close(self):
        """현재 스레드의 DB 연결 닫기"""
        if hasattr(self._local, 'conn'):
            try:
                self._local.conn.close()
            except Exception as e:
                logger.debug(f"DB 연결 닫기 실패: {e}")
            finally:
                del self._local.conn
                with self._lock:
                    DBManager._connection_count = max(0, DBManager._connection_count - 1)
    
    @classmethod
    def close_all(cls):
        """모든 DB 연결 정리 (graceful shutdown용)"""
        logger.info(f"DB 연결 정리 시작 (활성: {cls._connection_count}개)")
        cls._connection_count = 0
        # thread-local 연결은 각 스레드에서 정리해야 하지만,
        # 종료 시에는 최소한 카운터 리셋
        logger.info("DB 연결 정리 완료")

db = DBManager()
