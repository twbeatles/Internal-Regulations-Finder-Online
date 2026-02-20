# -*- coding: utf-8 -*-
import os
import hashlib
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from app.utils import logger, get_app_directory, FileUtils
from app.config import AppConfig
from app.services.db import db
from app.services.document import DocumentComparator

class RevisionTracker:
    """규정 개정 이력 관리 (SQLite 기반)"""
    
    def __init__(self):
        self.revisions_dir = os.path.join(get_app_directory(), 'revisions')
        os.makedirs(self.revisions_dir, exist_ok=True)
        self._revisions_root = Path(self.revisions_dir).resolve()

    def _safe_revision_filename(self, display_name: str, version: str, timestamp: str) -> str:
        safe_base = FileUtils.sanitize_upload_filename(display_name or "document.txt")
        stem = Path(safe_base).stem
        stem = stem or "document"
        return f"{stem}_{version}_{timestamp}.txt"

    def _select_keys(self, primary_key: str, legacy_key: Optional[str] = None) -> List[str]:
        keys = [str(primary_key or "").strip()]
        if legacy_key:
            keys.append(str(legacy_key).strip())
        seen = set()
        ordered = []
        for key in keys:
            if key and key not in seen:
                seen.add(key)
                ordered.append(key)
        return ordered
    
    def save_revision(self, file_key: str, content: str, note: str = "", display_name: str = "") -> Dict:
        """새 버전 저장"""
        try:
            # 다음 버전 번호 결정
            row = db.fetchone("SELECT COUNT(*) as cnt FROM revisions WHERE filename=?", (file_key,))
            next_ver_num = (row['cnt'] if row else 0) + 1
            version = f"v{next_ver_num}"
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 파일 저장
            revision_filename = self._safe_revision_filename(display_name or file_key, version, timestamp)
            revision_path = (self._revisions_root / revision_filename).resolve()
            if self._revisions_root not in revision_path.parents:
                raise ValueError("유효하지 않은 리비전 파일 경로입니다")
            
            with open(revision_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # DB 저장 (content 컬럼에 파일명 저장)
            db.execute("""
                INSERT INTO revisions (filename, version, content, comment) 
                VALUES (?, ?, ?, ?)
            """, (file_key, version, revision_filename, note))
            
            return {
                'version': version,
                'date': datetime.now().isoformat(),
                'note': note,
                'file': revision_filename,
                'display_name': display_name
            }
        except Exception as e:
            logger.error(f"개정 이력 저장 실패: {e}")
            raise e
    
    def get_history(self, file_key: str, legacy_key: Optional[str] = None) -> List[Dict]:
        """버전 히스토리 조회"""
        keys = self._select_keys(file_key, legacy_key)
        if not keys:
            return []
        placeholders = ",".join("?" for _ in keys)
        rows = db.fetchall(
            f"SELECT * FROM revisions WHERE filename IN ({placeholders}) ORDER BY id DESC",
            tuple(keys)
        )
        history = []
        for r in rows:
            history.append({
                'version': r['version'],
                'date': r['created_at'],
                'note': r['comment'],
                'file': r['content'],  # content 컬럼에 파일명이 저장되어 있음
                'key': r['filename']
            })
        return history
    
    def get_revision(self, file_key: str, version: str, legacy_key: Optional[str] = None) -> Optional[str]:
        """특정 버전 내용 조회"""
        keys = self._select_keys(file_key, legacy_key)
        if not keys:
            return None
        placeholders = ",".join("?" for _ in keys)
        row = db.fetchone(
            f"SELECT content FROM revisions WHERE filename IN ({placeholders}) AND version=? ORDER BY id DESC",
            tuple(keys + [version])
        )
        if row:
            revision_filename = row['content']
            revision_path = (self._revisions_root / revision_filename).resolve()
            if self._revisions_root not in revision_path.parents:
                logger.warning(f"리비전 경로 차단됨: {revision_filename}")
                return None
            if revision_path.exists():
                with open(revision_path, 'r', encoding='utf-8') as f:
                    return f.read()
        return None
    
    def compare_versions(self, file_key: str, v1: str, v2: str, legacy_key: Optional[str] = None) -> Optional[Dict]:
        """버전 간 비교"""
        content1 = self.get_revision(file_key, v1, legacy_key=legacy_key)
        content2 = self.get_revision(file_key, v2, legacy_key=legacy_key)
        
        if content1 is None or content2 is None:
            return None
        
        comparator = DocumentComparator()
        return comparator.compare(content1, content2)


class FolderWatcher:
    """watchdog 기반 폴더 변경 감지"""
    
    def __init__(self, callback=None):
        self.observer = None
        self.watching = False
        self.watch_path = ""
        self.callback = callback
        self._watchdog_available = None
    
    @property
    def watchdog_available(self):
        """watchdog 가용성 확인"""
        if self._watchdog_available is None:
            try:
                from watchdog.observers import Observer
                from watchdog.events import FileSystemEventHandler
                self._watchdog_available = True
            except ImportError:
                self._watchdog_available = False
        return self._watchdog_available
    
    def start_watching(self, folder: str) -> bool:
        """모니터링 시작"""
        if not self.watchdog_available:
            logger.warning("watchdog 라이브러리 미설치 (pip install watchdog)")
            return False
        
        if self.watching:
            self.stop_watching()
        
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            
            class RegulationEventHandler(FileSystemEventHandler):
                def __init__(inner_self, callback):
                    inner_self._callback = callback
                
                def on_created(inner_self, event):
                    if not event.is_directory:
                        ext = os.path.splitext(event.src_path)[1].lower()
                        if ext in AppConfig.SUPPORTED_EXTENSIONS:
                            logger.info(f"📁 새 파일 감지: {event.src_path}")
                            if inner_self._callback:
                                inner_self._callback('created', event.src_path)
                
                def on_modified(inner_self, event):
                    if not event.is_directory:
                        ext = os.path.splitext(event.src_path)[1].lower()
                        if ext in AppConfig.SUPPORTED_EXTENSIONS:
                            logger.info(f"📝 파일 수정 감지: {event.src_path}")
                            if inner_self._callback:
                                inner_self._callback('modified', event.src_path)
                                
                def on_deleted(inner_self, event):
                    if not event.is_directory:
                         if inner_self._callback:
                                inner_self._callback('deleted', event.src_path)

            self.event_handler = RegulationEventHandler(self.callback)
            self.observer = Observer()
            self.observer.schedule(self.event_handler, folder, recursive=True)
            self.observer.start()
            self.watching = True
            self.watch_path = folder
            logger.info(f"폴더 모니터링 시작: {folder}")
            return True
            
        except Exception as e:
            logger.error(f"폴더 모니터링 시작 실패: {e}")
            self.watching = False
            return False
    
    def stop_watching(self):
        """모니터링 중지 (타임아웃 포함)"""
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5)  # 최대 5초 대기
            if self.observer.is_alive():
                logger.warning("Observer 종료 타임아웃 - 강제 종료 시도")
            self.observer = None
        self.watching = False
        self.watch_path = ""
        logger.info("폴더 모니터링 중지")
