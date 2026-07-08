# Implementation Review — v3.0 (2026-07-08)

## Scope

- RAG v3 (`rag/` 패키지, `/api/rag/chat`)
- SOLID 백엔드 분할 (`search/`, `document/`, `files/`, routes 분리)
- 프론트 ESM + CSS 분할
- `server.py` / `server_backup.py` 삭제

## Verification

```bash
python -m pytest -q
# 78 passed

Get-ChildItem -Name *.spec | ForEach-Object { python -m py_compile $_ }
# all spec syntax OK
```

## Packaging

- `server.spec` → `run.py` 엔트리
- AI spec: `collect_submodules('rag')`, `httpx`, `langgraph`
- Lite spec: `app.services.files`, `api_tags`, `api_revisions` 추가

## Removed (intentional)

- `app/services/search.py`
- `app/services/document.py`
- `server.py`, `server_backup.py`
- `static/rag/*.js` (→ `static/js/rag/`)

## Docs updated

- `README.md`, `CLAUDE.md`, `BUILD.md`, `GEMINI.md`, `OFFLINE_SETUP.md`
- `.gitignore` — `*.bak`, `terminals/`, `.codegraph/`