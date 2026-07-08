# -*- coding: utf-8 -*-
"""대화 저장소."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from app.services.db import db


class ConversationStore:
    def create_conversation(self, title: str = "") -> str:
        conv_id = uuid.uuid4().hex
        now = time.time()
        db.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conv_id, title or "새 대화", now, now),
        )
        return conv_id

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        citations: list[dict[str, Any]] | None = None,
    ) -> str:
        msg_id = uuid.uuid4().hex
        now = time.time()
        db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, citations_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                msg_id,
                conversation_id,
                role,
                content,
                json.dumps(citations or [], ensure_ascii=False),
                now,
            ),
        )
        db.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        return msg_id

    def list_conversations(self, limit: int = 30) -> list[dict[str, Any]]:
        rows = db.fetchall(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        row = db.fetchone(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        if not row:
            return None
        messages = db.fetchall(
            "SELECT id, role, content, citations_json, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        )
        msg_list = []
        for m in messages:
            citations = []
            try:
                citations = json.loads(m["citations_json"] or "[]")
            except json.JSONDecodeError:
                citations = []
            msg_list.append(
                {
                    "id": m["id"],
                    "role": m["role"],
                    "content": m["content"],
                    "citations": citations,
                    "created_at": m["created_at"],
                }
            )
        return {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "messages": msg_list,
        }

    def delete_conversation(self, conversation_id: str) -> bool:
        db.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        cur = db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return bool(cur.rowcount)