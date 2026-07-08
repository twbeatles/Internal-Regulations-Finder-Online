# -*- coding: utf-8 -*-
"""검색 히스토리."""
from typing import Any, Dict, List

from app.services.db import db

class SearchHistory:
    def add(self, query: str):
        """검색 히스토리에 쿼리 추가 (5분 내 중복 방지)"""
        query = query.strip()
        if len(query) < 2:
            return
        try:
            # 5분 이내 동일 쿼리 중복 저장 방지
            recent = db.fetchone(
                """SELECT id FROM search_history 
                   WHERE query = ? AND timestamp > datetime('now', '-5 minutes')""",
                (query,)
            )
            if recent:
                return  # 최근 동일 쿼리 존재, 저장 스킵
            db.execute("INSERT INTO search_history (query) VALUES (?)", (query,))
        except Exception:
            pass
            
    def get_recent(self, limit: int = 10) -> List[str]:
        rows = db.fetchall("SELECT query, MAX(timestamp) as ts FROM search_history GROUP BY query ORDER BY ts DESC LIMIT ?", (limit,))
        return [r['query'] for r in rows]
            
    def get_popular(self, limit: int = 10) -> List[Dict[str, Any]]:
        rows = db.fetchall("SELECT query, COUNT(*) as cnt FROM search_history GROUP BY query ORDER BY cnt DESC LIMIT ?", (limit,))
        return [{'query': r['query'], 'count': int(r['cnt'])} for r in rows]
            
    def suggest(self, prefix: str, limit: int = 5) -> List[str]:
        rows = db.fetchall("SELECT DISTINCT query FROM search_history WHERE query LIKE ? ORDER BY timestamp DESC LIMIT ?", (f"{prefix}%", limit))
        return [r['query'] for r in rows]
            
    def clear(self):
        db.execute("DELETE FROM search_history")
