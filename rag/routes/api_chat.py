# -*- coding: utf-8 -*-
import json
import time

from flask import Blueprint, Response, jsonify, request, session, stream_with_context

from app.auth import admin_required
from app.config import AppConfig
from app.constants import ErrorMessages, HttpStatus
from app.services.search import RateLimiter, qa_system, search_queue
from app.utils import api_error, api_success, logger
from rag.config import (
    RagConfig,
    get_search_mode,
    save_rag_config,
    save_search_mode,
)
from rag.llm.factory import get_llm_health
from rag.pipeline.agentic import run_agentic_rag
from rag.pipeline.orchestrator import RAGPipeline
from rag.store.conversations import ConversationStore

rag_bp = Blueprint("rag", __name__)
_pipeline = RAGPipeline()
_store = ConversationStore()
_rag_rate_limiter = RateLimiter(requests_per_minute=getattr(AppConfig, "RAG_RATE_LIMIT_PER_MINUTE", 60))


def _rag_rate_allowed() -> bool:
    client_ip = request.remote_addr or "unknown"
    return _rag_rate_limiter.is_allowed(client_ip)


def _session_conversation_ids() -> list[str]:
    raw = session.get("rag_conversation_ids") or []
    if not isinstance(raw, list):
        return []
    return [str(x) for x in raw if x]


def _claim_conversation(conversation_id: str) -> None:
    if not conversation_id:
        return
    ids = _session_conversation_ids()
    if conversation_id not in ids:
        ids.append(conversation_id)
        session["rag_conversation_ids"] = ids[-100:]


def _can_access_conversation(conversation_id: str) -> bool:
    if session.get("admin_authenticated"):
        return True
    return str(conversation_id) in set(_session_conversation_ids())


def _sanitize_history(raw: list | None) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    max_items = int(getattr(AppConfig, "MAX_RAG_HISTORY_ITEMS", 6) or 6)
    max_len = int(getattr(AppConfig, "MAX_RAG_HISTORY_ITEM_LENGTH", 2000) or 2000)
    cleaned: list[dict[str, str]] = []
    for item in raw[-max_items:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user"))
        if role not in ("user", "assistant"):
            continue
        content = str(item.get("content", ""))[:max_len]
        cleaned.append({"role": role, "content": content})
    return cleaned


def _redact_llm_settings(cfg: dict) -> dict:
    """응답에서 API 키 등 민감 필드 제거."""
    out = dict(cfg or {})
    for key in list(out.keys()):
        lk = str(key).lower()
        if "key" in lk or "secret" in lk or "token" in lk or "password" in lk:
            out[key] = "***"
    return out


@rag_bp.route("/rag/status", methods=["GET"])
def rag_status():
    config = RagConfig.from_settings()
    llm_health = get_llm_health(config)
    return jsonify(
        api_success(
            "RAG 상태",
            {
                "search_mode": get_search_mode(),
                "ready": _pipeline.is_ready(),
                "index_ready": qa_system.is_ready if hasattr(qa_system, "is_ready") else _pipeline.is_ready(),
                "documents": len(qa_system.documents),
                "llm": llm_health,
                "config": {
                    "retrieval": config.retrieval,
                    "guardrails": config.guardrails,
                },
            },
        )
    )


@rag_bp.route("/settings/search-mode", methods=["GET"])
def get_search_mode_route():
    return jsonify(api_success("검색 모드", {"search_mode": get_search_mode()}))


@rag_bp.route("/settings/search-mode", methods=["POST"])
@admin_required
def set_search_mode_route():
    data = request.json or {}
    mode = str(data.get("search_mode", "")).strip().lower()
    if mode not in ("rag", "legacy"):
        return api_error("search_mode는 rag 또는 legacy여야 합니다", "INVALID_MODE", 400)
    if not save_search_mode(mode):
        return api_error("설정 저장 실패", "SAVE_FAILED", 500)
    return jsonify(api_success("검색 모드 변경됨", {"search_mode": mode}))


@rag_bp.route("/settings/rag", methods=["GET"])
def get_rag_settings():
    config = RagConfig.from_settings()
    return jsonify(
        api_success(
            "RAG 설정",
            {
                "rag": {
                    "llm": _redact_llm_settings(config.llm),
                    "llm_fallback": _redact_llm_settings(config.llm_fallback),
                    "retrieval": config.retrieval,
                    "guardrails": config.guardrails,
                }
            },
        )
    )


@rag_bp.route("/settings/rag", methods=["POST"])
@admin_required
def set_rag_settings():
    data = request.json or {}
    rag_data = data.get("rag") if isinstance(data.get("rag"), dict) else data
    if not isinstance(rag_data, dict):
        return api_error("rag 설정 객체가 필요합니다", "INVALID_BODY", 400)
    if not save_rag_config(rag_data):
        return api_error("RAG 설정 저장 실패", "SAVE_FAILED", 500)
    global _pipeline
    _pipeline = RAGPipeline(config=RagConfig.from_settings())
    cfg = RagConfig.from_settings()
    return jsonify(
        api_success(
            "RAG 설정 저장됨",
            {
                "rag": {
                    "llm": _redact_llm_settings(cfg.llm),
                    "llm_fallback": _redact_llm_settings(cfg.llm_fallback),
                    "retrieval": cfg.retrieval,
                    "guardrails": cfg.guardrails,
                }
            },
        )
    )


def _parse_chat_request() -> tuple[dict, tuple | None]:
    data = request.json or {}
    message = str(data.get("message", "")).strip()
    if len(message) < 2:
        return data, api_error("메시지는 2자 이상이어야 합니다", "MESSAGE_TOO_SHORT", 400)
    max_len = int(getattr(AppConfig, "MAX_RAG_MESSAGE_LENGTH", 4000) or 4000)
    if len(message) > max_len:
        return data, api_error(ErrorMessages.MESSAGE_TOO_LONG, "MESSAGE_TOO_LONG", 400)
    return data, None


@rag_bp.route("/rag/chat/sync", methods=["POST"])
def rag_chat_sync():
    if not _rag_rate_allowed():
        return api_error(ErrorMessages.SEARCH_RATE_LIMITED, "RATE_LIMITED", HttpStatus.TOO_MANY_REQUESTS)
    if getattr(qa_system, "is_loading", False) and getattr(AppConfig, "BLOCK_SEARCH_DURING_INDEXING", True):
        return api_error(ErrorMessages.SEARCH_INDEXING_IN_PROGRESS, "INDEXING", 503)
    if not search_queue.acquire(timeout=10):
        return api_error(ErrorMessages.SEARCH_QUEUE_FULL, "QUEUE_FULL", HttpStatus.SERVICE_UNAVAILABLE)
    try:
        data, err = _parse_chat_request()
        if err:
            return err
        message = str(data.get("message", "")).strip()
        conversation_id = data.get("conversation_id")
        filter_file_id = data.get("filter_file_id")
        filter_file = data.get("filter_file")
        use_agentic = bool(data.get("agentic", False))
        history = _sanitize_history(data.get("history") if isinstance(data.get("history"), list) else [])

        if conversation_id and not _can_access_conversation(str(conversation_id)):
            return api_error(ErrorMessages.CONVERSATION_FORBIDDEN, "FORBIDDEN", 403)

        if not _pipeline.is_ready():
            return api_error("인덱스가 준비되지 않았습니다", "INDEX_NOT_READY", 503)

        if use_agentic:
            result = run_agentic_rag(
                message,
                filter_file_id=filter_file_id,
                filter_file=filter_file,
                history=history,
            )
        else:
            result = _pipeline.run(
                message,
                filter_file_id=filter_file_id,
                filter_file=filter_file,
                history=history,
            )

        if not conversation_id:
            conversation_id = _store.create_conversation(title=message[:40])
        _claim_conversation(str(conversation_id))
        _store.add_message(conversation_id, "user", message)
        msg_id = _store.add_message(
            conversation_id,
            "assistant",
            result.answer,
            [c.to_dict() for c in result.citations],
        )
        result.conversation_id = conversation_id
        result.message_id = msg_id
        return jsonify(api_success("RAG 응답", result.to_dict()))
    finally:
        search_queue.release()


@rag_bp.route("/rag/chat", methods=["POST"])
def rag_chat_stream():
    if not _rag_rate_allowed():
        return api_error(ErrorMessages.SEARCH_RATE_LIMITED, "RATE_LIMITED", HttpStatus.TOO_MANY_REQUESTS)
    if getattr(qa_system, "is_loading", False) and getattr(AppConfig, "BLOCK_SEARCH_DURING_INDEXING", True):
        return api_error(ErrorMessages.SEARCH_INDEXING_IN_PROGRESS, "INDEXING", 503)
    if not search_queue.acquire(timeout=10):
        return api_error(ErrorMessages.SEARCH_QUEUE_FULL, "QUEUE_FULL", HttpStatus.SERVICE_UNAVAILABLE)

    data, err = _parse_chat_request()
    if err:
        search_queue.release()
        return err

    message = str(data.get("message", "")).strip()
    conversation_id = data.get("conversation_id")
    filter_file_id = data.get("filter_file_id")
    filter_file = data.get("filter_file")
    history = _sanitize_history(data.get("history") if isinstance(data.get("history"), list) else [])
    stream = data.get("stream", True)

    if conversation_id and not _can_access_conversation(str(conversation_id)):
        search_queue.release()
        return api_error(ErrorMessages.CONVERSATION_FORBIDDEN, "FORBIDDEN", 403)

    if not _pipeline.is_ready():
        search_queue.release()
        return api_error("인덱스가 준비되지 않았습니다", "INDEX_NOT_READY", 503)

    if not stream:
        search_queue.release()
        return rag_chat_sync()

    if not conversation_id:
        conversation_id = _store.create_conversation(title=message[:40])
    _claim_conversation(str(conversation_id))
    _store.add_message(conversation_id, "user", message)

    def event_stream():
        answer_parts: list[str] = []
        citations: list[dict] = []
        done_payload: dict = {}
        try:
            yield f"event: meta\ndata: {json.dumps({'conversation_id': conversation_id}, ensure_ascii=False)}\n\n"
            for item in _pipeline.stream(
                message,
                filter_file_id=filter_file_id,
                filter_file=filter_file,
                history=history,
            ):
                event = item.get("event")
                payload = item.get("data")
                if event == "token":
                    answer_parts.append(str(payload))
                    yield f"event: token\ndata: {json.dumps({'text': payload}, ensure_ascii=False)}\n\n"
                elif event == "citation":
                    citations.append(payload)
                    yield f"event: citation\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                elif event == "replace":
                    # 가드레일 거부 시 스트림 본문 교체
                    if isinstance(payload, dict) and payload.get("answer") is not None:
                        answer_parts = [str(payload.get("answer"))]
                    yield f"event: replace\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                elif event == "done":
                    done_payload = payload if isinstance(payload, dict) else {}
            answer = "".join(answer_parts) or str(done_payload.get("answer", ""))
            if done_payload.get("refused") and done_payload.get("answer"):
                answer = str(done_payload.get("answer"))
            msg_id = _store.add_message(conversation_id, "assistant", answer, citations)
            done_payload["conversation_id"] = conversation_id
            done_payload["message_id"] = msg_id
            yield f"event: done\ndata: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"RAG 스트리밍 오류: {e}")
            yield f"event: error\ndata: {json.dumps({'message': '응답 생성 중 오류가 발생했습니다'}, ensure_ascii=False)}\n\n"
        finally:
            search_queue.release()

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@rag_bp.route("/rag/conversations", methods=["GET"])
def list_conversations():
    limit = int(request.args.get("limit", 30))
    all_items = _store.list_conversations(limit=max(limit, 100))
    if session.get("admin_authenticated"):
        items = all_items[:limit]
    else:
        allowed = set(_session_conversation_ids())
        items = [c for c in all_items if c.get("id") in allowed][:limit]
    return jsonify(api_success("대화 목록", {"conversations": items}))


@rag_bp.route("/rag/conversations/<conversation_id>", methods=["GET"])
def get_conversation(conversation_id: str):
    if not _can_access_conversation(conversation_id):
        return api_error(ErrorMessages.CONVERSATION_FORBIDDEN, "FORBIDDEN", 403)
    conv = _store.get_conversation(conversation_id)
    if not conv:
        return api_error("대화를 찾을 수 없습니다", "NOT_FOUND", 404)
    return jsonify(api_success("대화 조회", conv))


@rag_bp.route("/rag/conversations/<conversation_id>", methods=["DELETE"])
def delete_conversation(conversation_id: str):
    if not _can_access_conversation(conversation_id):
        return api_error(ErrorMessages.CONVERSATION_FORBIDDEN, "FORBIDDEN", 403)
    if not _store.delete_conversation(conversation_id):
        return api_error("대화를 찾을 수 없습니다", "NOT_FOUND", 404)
    ids = [i for i in _session_conversation_ids() if i != conversation_id]
    session["rag_conversation_ids"] = ids
    return jsonify(api_success("대화 삭제됨"))


@rag_bp.route("/rag/conversations/<conversation_id>/export", methods=["GET"])
def export_conversation(conversation_id: str):
    if not _can_access_conversation(conversation_id):
        return api_error(ErrorMessages.CONVERSATION_FORBIDDEN, "FORBIDDEN", 403)
    fmt = request.args.get("format", "md").lower()
    conv = _store.get_conversation(conversation_id)
    if not conv:
        return api_error("대화를 찾을 수 없습니다", "NOT_FOUND", 404)
    if fmt == "json":
        return jsonify(api_success("대화보내기", conv))
    lines = [f"# {conv.get('title', '대화')}", ""]
    for msg in conv.get("messages", []):
        role = "사용자" if msg["role"] == "user" else "어시스턴트"
        lines.append(f"## {role}")
        lines.append(msg.get("content", ""))
        if msg.get("citations"):
            lines.append("")
            lines.append("### 인용")
            for c in msg["citations"]:
                lines.append(f"- [{c.get('index')}] {c.get('source')} {c.get('article_no', '')}")
        lines.append("")
    return Response("\n".join(lines), mimetype="text/markdown; charset=utf-8")
