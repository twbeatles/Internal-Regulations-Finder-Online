# -*- coding: utf-8 -*-
import sqlite3
import os
import threading
from app.utils import get_app_directory, logger

class DBManager:
    _instance = None
    _lock = threading.Lock()
    _local = threading.local()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DBManager, cls).__new__(cls)
                cls._instance.db_path = os.path.join(get_app_directory(), 'config', 'regulations.db')
                cls._instance._init_db()
        return cls._instance
    
    def _get_conn(self):
        """스레드별 DB 연결 반환"""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
            # 성능 최적화: WAL 모드 등 적용
            self._local.conn.execute('PRAGMA journal_mode=WAL')
            self._local.conn.execute('PRAGMA synchronous=NORMAL')
            self._local.conn.execute('PRAGMA temp_store=MEMORY')
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
                
                conn.commit()
        except Exception as e:
            logger.error(f"DB 초기화 실패: {e}")

    def execute(self, query, args=()):
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(query, args)
            conn.commit()
            return cur
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
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            del self._local.conn

db = DBManager()
