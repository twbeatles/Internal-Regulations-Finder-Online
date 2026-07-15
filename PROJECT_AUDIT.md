# Project Audit

**대상**: 사내 규정 검색기 v3.0 (`Internal-Regulations-Finder-Online`)  
**감사일**: 2026-07-15  
**방법**: README.md / CLAUDE.md 검토 → CodeGraph 호출 관계 분석 → 필요 시 소스·테스트 보조 확인  
**범위**: 기능 구현·예외·입력 검증·상태/동시성·경로/인코딩·DB/캐시·보안·테스트·문서 정합성  
**비고**: 2026-07-15 개선 구현 착수 — 1~2단계 및 테스트 상당 부분 반영. 상세는 하단 *Remediation Log* 참고.

---

## 1. Executive Summary

이 프로젝트는 **Flask 기반 사내 규정 RAG + 하이브리드 검색 시스템**으로, v2.x 보안·인덱스 정합 보강과 v3.0 SOLID/RAG 리팩토링이 상당 부분 반영되어 있다. 업로드 경로 sanitization, ZIP 제한, `admin_required`, CORS allowlist, 리비전 경로 검증, 검색 rate limit/queue 등 **기본 안전장치는 구현되어 있다**.

다만 기능·운영 관점에서 아래 리스크가 남아 있다.

| 영역 | 요약 | 위험도 |
|------|------|--------|
| 인덱스 동시성 | 검색 경로가 `qa_system._lock`을 잡지 않음. 재인덱싱 중 검색 시 데이터 레이스 가능 | **High** |
| 동기화 중복 실행 | `initialize()`가 진행 중 여부를 막고 백그라운드 작업을 중복 submit 가능 | **High** |
| RAG 대화 격리 | 대화 목록/조회/삭제에 인증·소유권 없음. UUID 추측 시 타 대화 열람/삭제 가능 | **High** (다중 사용자) |
| RAG 가드레일 타이밍 | 스트리밍은 토큰을 먼저 보내고, 거부/검증은 `done` 단계에서 적용 | **High** (기능 신뢰성) |
| 관리자 로그인 보호 | 비밀번호 검증에 rate limit/lockout 없음 | **High** (네트워크 노출 시) |
| MCP 재인덱스 | `mcp.admin_token` 미설정 시 토큰 없이 reindex 가능 | **High** (MCP 활성 시) |
| FAISS 캐시 역직렬화 | `allow_dangerous_deserialization=True` | **Medium~High** |
| 입력 상한 | RAG 메시지/검색어 최소 길이만 있고 상한 부족 → LLM/CPU DoS 여지 | **Medium** |
| 문서 정합성 | CLAUDE.md 일부 경로·테스트 수가 구현과 불일치 | **Low** (문서) |

**전체 위험도**: 사내 단일 사용자·로컬호스트 전용 배포면 **Medium**. 같은 LAN에 다중 사용자로 열어 두거나 MCP/클라우드 LLM을 쓰는 운영이면 **High**.

핵심 권고는 (1) 인덱스 읽기/쓰기 락 정합, (2) 동기화 단일 실행 가드, (3) RAG 대화·스트리밍 가드레일·입력 상한, (4) 관리자 로그인 보호, (5) MCP 토큰 필수화다.

---

## 2. Project Understanding

### 2.1 목적 (README / CLAUDE)

- **목적**: 사내 규정 문서에 대한 **RAG 질의응답**과 **레거시 하이브리드 검색(Vector + BM25)** 제공  
- **스택**: Flask, PyTorch/ONNX 임베딩, LangChain/FAISS, SQLite, Ollama/클라우드 LLM  
- **실행**: `run.py`(콘솔), `server_gui.py`(PyQt6 GUI). `server.py`는 제거됨  
- **모드**: `search_mode`: `rag` | `legacy`

### 2.2 아키텍처 (문서 + CodeGraph)

```
Browser (static/js/bootstrap + legacy/app.js + rag/*)
        │
        ▼
Flask create_app (app/__init__.py)
  ├─ routes: search / files / tags / revisions / system / rag
  ├─ services/search  (RegulationQASystem, BM25, cache, queue, rate_limiter)
  ├─ services/document, files, parsers, db, settings_store
  └─ rag/pipeline (retrieve → rerank → context → generate → guardrails)
```

**주요 실행 흐름**

1. **기동**: `run.py` → `app` import → `create_app()`(모듈 레벨) + 백그라운드 `initialize_server()`  
   GUI는 `server_gui.ModuleLoaderThread` → `_load_heavy_modules()` → `create_app()` / `load_model`
2. **검색**: `POST /api/search` → rate limit → SearchQueue → `qa_system.search` → `HybridSearchService`  
   (Vector ∥ BM25, 캐시 키에 filter/sort 포함)
3. **RAG**: `POST /api/rag/chat` SSE → `RAGPipeline.stream` → 동일 `qa_system` retrieval 재사용  
4. **관리 작업**: `admin_required`(세션) 후 업로드/삭제/동기화/모델 변경  
5. **선택 MCP**: `app/mcp/builder.py` → 검색/RAG/재인덱스 도구 (기본 포트 8081)

### 2.3 잘 되어 있는 점 (감사 중 확인)

- 관리 상태 변경 API에 `admin_required` 적용 (업로드, 삭제, sync, cache clear 등)
- 업로드 파일명 sanitization + 업로드 루트 밖 경로 차단
- ZIP: `max_entries` / 압축 해제 총량 / 단일 파일 크기 제한, `..` 멤버 차단
- 리비전 파일 경로를 `revisions` 루트 내부로 제한
- `FilePathResolver`는 인덱스에 있는 경로만 해석(basename 정규화)
- 삭제 기본 정책 `index_only`, 원본 삭제는 허용 루트 검사
- 검색 rate limit + 동시 검색 큐
- SQLite 파라미터 바인딩, WAL, busy_timeout
- 테스트 수집 기준 **92 tests** (README v3.0 표기와 일치)

### 2.4 문서 vs 구현 불일치 (요약)

| 문서 | 내용 | 실제 |
|------|------|------|
| CLAUDE.md §2 | `app/services/document.py` | `app/services/document/` 패키지 |
| CLAUDE.md 디렉터리 트리 | `search.py`, `document.py` 단파일 | 패키지로 분리 (v3.0 메모에는 올바름) |
| CLAUDE.md 검증 | `78 passed` | 수집 **92** / README 상단도 92 |
| README 하단 일부 | 78 passed (2026-07-08) | 상단 92와 혼재 |
| CLAUDE 구조 | MCP 언급 약함 | `app/mcp/*`, README에 MCP 섹션 존재 |

---

## 3. High-Risk Issues

### H-1. 검색과 재인덱싱 사이 공유 상태 race

* **위치**: `app/services/search/hybrid_search.py` (`HybridSearchService.search`), `app/services/search/qa_system.py` (`process_documents` / `_process_internal`, `process_single_file`, `initialize`)
* **문제**: 인덱스 변경은 `qa_system._lock`을 사용하지만, **검색 경로는 동일 락을 획득하지 않는다**. 재인덱싱 중 `documents`/`doc_meta`/`vector_store`/`bm25`가 교체·삭제되는 동안 검색이 진행되면 잘못된 점수, 빈 결과, 예외(인덱스 범위 오류 등)가 날 수 있다.
* **영향**: 동기화·업로드·삭제와 검색이 겹칠 때 간헐적 검색 실패 또는 오염된 결과. 운영 중 “가끔 검색이 깨짐”으로 나타남.
* **근거**:
  - `process_documents`는 `with self._lock` 후 `_process_internal`에서 `documents` 초기화·재구축
  - `HybridSearchService`는 `qa.documents`/`qa.doc_meta`/`qa.vector_store`를 락 없이 읽음
  - BM25 자체는 RLock이 있으나, 문서 리스트와 메타는 보호되지 않음
* **권장 수정 방향**:
  - 검색 시 짧은 읽기 스냅샷 또는 `RLock` 공유
  - 또는 재인덱싱 중 검색을 503으로 거절하고 UI에 “인덱싱 중” 표시
  - 이상적으로는 immutable 인덱스 스왑(빌드 완료 후 포인터 교체)
* **우선순위**: **High**

---

### H-2. `initialize()` 중복 백그라운드 실행

* **위치**: `RegulationQASystem.initialize` (`qa_system.py` ~683행)
* **문제**: `_is_loading` 여부를 **시작 전에 검사하지 않고** `_executor.submit(bg_process)`만 한다. 관리자/UI가 sync를 연속 호출하거나 모델 변경 시 재인덱스와 수동 sync가 겹치면 **복수 백그라운드 인덱싱**이 동시에 돈다.
* **영향**: CPU/메모리 폭증, 인덱스 최종 상태 비결정, 취소 플래그 경합, “완료/오류” 상태 표시 혼란.
* **근거**:
  - `load_model`은 `if self._is_loading: return ...` 가드 있음
  - `initialize`는 가드 없이 submit → `bg_process`에서야 `_is_loading = True`
  - submit과 플래그 설정 사이에 윈도우 존재
* **권장 수정 방향**:
  - `initialize` 진입 시 lock + “이미 로딩 중이면 거부/큐잉”
  - 단일 flight token / generation counter로 늦은 작업 결과 무시
* **우선순위**: **High**

---

### H-3. RAG 대화 API 인증·소유권 부재

* **위치**: `rag/routes/api_chat.py` — `list_conversations`, `get_conversation`, `delete_conversation`, `export_conversation`, chat 저장 경로  
  `rag/store/conversations.py` — `ConversationStore`
* **문제**: 대화 CRUD가 **세션/사용자 식별 없이** 전역 SQLite에 저장된다. 목록 API로 전체 대화 id를 열거할 수 있고, id만 알면 조회·삭제·export 가능하다. 클라이언트가 넘긴 `conversation_id`에 메시지를 그대로 append한다.
* **영향**: 동일 서버를 여러 사람이 쓰면 **타인 질의/규정 해석 내용 유출·삭제**. 단일 사용자 로컬 도구라면 영향 축소.
* **근거**:
  - `GET /api/rag/conversations` 인증 없음
  - `DELETE /api/rag/conversations/<id>` 인증 없음
  - `add_message(conversation_id, ...)`에 소유자 검증 없음
* **권장 수정 방향**:
  - 최소: 대화 삭제/목록에 admin 또는 opaque 소유 토큰
  - 권장: 세션/사용자별 conversation 스코프, 생성 시 서버 발급 id만 허용
* **우선순위**: **High** (다중 사용자 가정 시) / 단일 로컬 전용이면 **Medium**

---

### H-4. 스트리밍 응답이 가드레일 적용 전에 노출됨

* **위치**: `rag/pipeline/orchestrator.py` `stream()`, `rag/pipeline/guardrails.py` `apply()`
* **문제**: `stream()`은 LLM 토큰을 즉시 `token` 이벤트로 내보낸 뒤, 전체 답변에 대해 `guardrails.apply()`를 수행한다. confidence 미달·무근거 답변 거부 로직은 **이미 스트리밍된 본문을 되돌리지 못한다**.
* **영향**: UI가 토큰 스트림을 그대로 보여주면 **거부되어야 할 답변이 사용자에게 노출**된 뒤 `done`에서만 `refused`/대체 문구가 올 수 있음. 규정 검색 신뢰성·컴플라이언스 측면에서 기능 결함.
* **근거**:
  - orchestrator: `for token in self.generator.stream(...): yield token` 후 `guardrails.apply`
  - `run()`(non-stream)은 생성 후 한 번에 apply → 동작 불일치
* **권장 수정 방향**:
  - 스트리밍 완료 후 거부 시 클라이언트에 `retract`/`replace` 이벤트
  - 또는 버퍼링 후 검증 통과 시에만 스트림 (지연↑)
  - 프론트는 `done.refused`면 스트림 본문 폐기
* **우선순위**: **High**

---

### H-5. 관리자 비밀번호 무차별 대입 방어 부재

* **위치**: `app/routes/api_system.py` `verify_password`, `admin_auth`  
  `app/services/settings_store.py` `verify_admin_password`
* **문제**: 로그인 실패에 대한 **IP rate limit / 계정 lockout / exponential backoff**가 없다. 검색 API rate limit과 분리되어 있다.
* **영향**: 네트워크에서 관리 포트가 열려 있으면 약한 비밀번호에 대한 브루트포스. 성공 시 업로드·삭제·폴더 동기화 권한 탈취.
* **근거**: 인증 엔드포인트에 `rate_limiter` 미사용; 실패 시 401만 반환
* **권장 수정 방향**:
  - 로그인 전용 RateLimiter (예: 5회/분) + 실패 로그/알림
  - 가능하면 상수 시간 응답 유지(이미 hmac.compare_digest 사용 중 — 유지)
* **우선순위**: **High** (외부/LAN 노출 시) / 로컬호스트 전용이면 **Medium**

---

### H-6. MCP `regulations.reindex` 토큰 선택적 검증

* **위치**: `app/mcp/tools.py` `regulations_reindex`
* **문제**:
  ```python
  if expected and admin_token != expected:
      return failure
  ```
  `mcp.admin_token`이 비어 있으면 **토큰 검사 자체를 건너뛴다**. MCP가 켜진 환경에서 로컬 포트(8081)에 접근 가능한 프로세스가 재인덱스를 트리거할 수 있다.
* **영향**: 의도치 않은 전체 재인덱스(부하/서비스 중단). 토큰 미설정 운영이 기본이 되기 쉬움.
* **근거**: CodeGraph/소스 상 empty token bypass 명시
* **권장 수정 방향**:
  - reindex는 토큰 **필수**(미설정 시 도구 비활성 또는 항상 거부)
  - MCP bind를 127.0.0.1 유지 + 문서에 강제 안내
* **우선순위**: **High** (MCP 사용 시) / MCP 미사용이면 **Low**

---

### H-7. FAISS 로컬 캐시 위험 역직렬화

* **위치**: `app/services/search/qa_system.py` `_process_internal` 캐시 로드  
  `FAISS.load_local(..., allow_dangerous_deserialization=True)`
* **문제**: LangChain/FAISS 로컬 로드가 pickle 기반일 수 있으며, 캐시 디렉터리에 악의적 파일이 있으면 **역직렬화 시 코드 실행** 위험이 있다.
* **영향**: 캐시 경로 쓰기 권한이 있는 다른 사용자/악성 업로드 경로와 결합 시 RCE. 전용 PC·단일 사용자면 완화.
* **근거**: 코드에 `allow_dangerous_deserialization=True` 명시
* **권장 수정 방향**:
  - 캐시 디렉터리 권한 제한, 무결성 해시(모델 id + 파일 mtime 서명)
  - 가능하면 안전한 직렬화 포맷/공식 권고 경로 사용
* **우선순위**: **Medium** (단일 사용자) / **High** (공유 머신)

---

### M-1. RAG/검색 입력 상한 부족 (DoS·비용)

* **위치**: `rag/routes/api_chat.py` `_parse_chat_request` (최소 2자만), `HybridSearchService` (최소 2자, `k`만 상한)
* **문제**: 메시지/쿼리 **최대 길이** 제한이 없다. 거대 payload + history 주입 시 임베딩·LLM 비용/지연·메모리 부담.
* **영향**: rate limit 안에서도 요청당 비용이 커져 서비스 마비·API 과금 가능.
* **근거**: `len(message) < 2`만 검사; history는 list면 그대로 최대 6개 role 내용 무제한 길이
* **권장 수정 방향**: 예: message 2k~4k자, history 항목 길이 제한, search query 상한
* **우선순위**: **Medium**

---

### M-2. ZIP 메타데이터 신뢰 (실제 바이트 미검증)

* **위치**: `app/routes/api_files.py` `upload_folder`
* **문제**: 제한은 `ZipInfo.file_size` 합에 의존. 해제 시 `shutil.copyfileobj`로 실제 기록 바이트를 세지 않는다. 메타를 속인 zip bomb 변형에 취약할 수 있다.
* **영향**: 디스크 고갈, 인덱싱 장시간 점유. admin 전용이라 외부 익스플로잇 면적은 축소.
* **근거**: 사전 검사 후 open/copy만 수행, write 중 `max_single_file_bytes` 재검사 없음
* **권장 수정 방향**: 스트림 복사 시 바이트 카운트, 초과 시 파일 삭제·중단
* **우선순위**: **Medium**

---

### M-3. `/api/init` 경로 검증이 `/api/sync/start`보다 약함

* **위치**: `api_system.py` `init_server` vs `sync_start`
* **문제**: `sync_start`는 realpath/위험 패턴 검사 후 디렉터리 여부 확인. `init`은 `exists` 정도만 보고 `initialize`에 넘긴다.
* **영향**: 동일 관리자 권한이라도 일관되지 않은 경로 정책. 파일 경로를 폴더처럼 walk 할 여지(동작 오류).
* **근거**: `init_server` 126–136행 vs `sync_start` 225–263행
* **권장 수정 방향**: 공통 `validate_folder_path()` 헬퍼로 통일
* **우선순위**: **Medium**

---

### M-4. 예외 메시지 클라이언트 노출

* **위치**: `app/__init__.py` 500 핸들러 (`'error': str(e)`), 여러 API `str(e)` 반환, RAG stream `event: error`에 `str(e)`
* **문제**: 내부 예외 문자열이 응답에 포함될 수 있다.
* **영향**: 경로·라이브러리·설정 정보 유출 (정보 공개).
* **권장 수정 방향**: 클라이언트는 일반 메시지, 상세는 서버 로그만
* **우선순위**: **Medium**

---

### M-5. 클라이언트 `history` 신뢰 (프롬프트 주입)

* **위치**: `rag/pipeline/generation.py` `build_messages`, `api_chat` request body `history`
* **문제**: 클라이언트가 보낸 history를 system 다음 대화로 삽입한다. 조작된 assistant 턴으로 모델 행동을 유도할 수 있다.
* **영향**: 가드레일/컨텍스트 규칙 우회 시도. 서버 저장 히스토리만 쓰는 편이 안전.
* **권장 수정 방향**: DB 대화 이력만 사용하거나 history 역할을 user로 고정/길이 제한
* **우선순위**: **Medium**

---

### M-6. 매 요청 `llm.health()` 호출

* **위치**: `rag/pipeline/orchestrator.py` `run`/`stream`
* **문제**: 매 RAG 요청마다 provider health 확인. Ollama는 `/api/tags` HTTP 호출.
* **영향**: 지연 증가, 로컬 Ollama 부하. health 실패 시 retrieval_only 플래그는 세팅하지만 이후 `generator.stream`은 다시 LLM을 시도할 수 있어 의미도 모호.
* **권장 수정 방향**: TTL 캐시(예: 30s), health와 실제 생성 경로 정책 통일
* **우선순위**: **Medium** (성능/기능 일관성)

---

### M-7. 모듈 레벨 `create_app()` 이중 생성 여지

* **위치**: `app/__init__.py` 끝 `app = create_app()`, `server_gui._load_heavy_modules` 내 재호출, 테스트마다 `create_app()`
* **문제**: Flask 앱 인스턴스는 여러 개 생길 수 있으나 **전역 `qa_system`/`db`는 공유**된다. 설정/시크릿 키 파일 부작용은 대체로 괜찮지만, 테스트·GUI 혼용 시 상태 공유로 flaky 테스트 가능.
* **영향**: 테스트 격리 약화, 드물게 설정 혼선.
* **권장 수정 방향**: 앱 팩토리만 사용하고 모듈 전역 app은 lazy; 테스트는 app context fixture 단일화
* **우선순위**: **Low~Medium**

---

### L-1. 문서 경로·테스트 수 불일치

* **위치**: `CLAUDE.md`, `README.md` 일부 섹션
* **문제**: 삭제된 `document.py`/`search.py` 단파일 표기 잔존, 테스트 78 vs 92 혼재
* **영향**: 신규 기여자/AI 어시스턴트 오해, 잘못된 수정 경로
* **권장 수정 방향**: 패키지 경로·`pytest` 기준 수치 단일화
* **우선순위**: **Low**

---

### L-2. 미리보기/다운로드/검색 결과 비인증 (설계 트레이드오프)

* **위치**: `api_files` preview/download, `POST /api/search`, 파일 목록
* **문제**: 관리 작업만 인증하고 읽기 경로는 공개. 사내 도구로 합리적일 수 있으나, 서버가 노출되면 규정 전문 유출.
* **영향**: 네트워크 경계에 전적으로 의존
* **권장 수정 방향**: 배포 가이드에 “내부망 전용” 명시, 필요 시 읽기 토큰/SSO
* **우선순위**: **Low** (제품 요구에 따라 **Medium**으로 상승 가능) — **설계 이슈, 버그 단정 아님**

---

## 4. Potential Functional Gaps

확실하지 않은 항목은 **(추정)** 표시.

1. **인덱싱 중 검색 UX**  
   상태 API에 loading은 있으나, 검색 API가 인덱싱 중 명시적 거부를 하지 않음. 프론트가 ready만 보면 부분 인덱스 검색 가능. **(추정: 의도일 수 있음)**

2. **FolderWatcher 연동 완성도**  
   `FolderWatcher` 클래스는 있으나, 실제 `run.py`/라우트에서 상시 감시 콜백으로 자동 재인덱싱이 항상 켜져 있는지는 배포 설정 의존. 수동 sync 중심일 가능성. **(추정)**

3. **RAG citation UI ↔ 스트림 retract**  
   가드레일 거부 후 프론트가 스트림 본문을 어떻게 처리하는지에 따라 사용자 경험 품질이 갈림. **(추정: 프론트 보완 필요 가능)**

4. **대화 제목/정리/보관 정책**  
   대화 무한 적재 시 SQLite 비대화. retention/purge 없음. **(추정)**

5. **멀티 프로세스/멀티 워커**  
   전역 인메모리 인덱스 → waitress threads에는 맞지만 프로세스 복제 시 인덱스 불일치. 문서상 threads 중심. 프로세스 스케일아웃 미지원으로 보임.

6. **Lite 모드 + RAG**  
   임베딩 없이 BM25만 있을 때 RAG retrieval은 동작 가능하나 LLM 품질·가드레일 민감도 차이. 안내/테스트 부족. **(추정)**

7. **설정 `rag`에 api_key 저장 시 노출**  
   `GET /api/settings/rag`는 admin 없이 설정 일부를 반환. 기본 스키마는 env 키를 쓰지만, 병합 저장으로 키가 들어가면 유출 가능. **(추정: 운영 실수 시)**

8. **MCP list_files path 노출**  
   절대 경로가 MCP 응답에 포함 → 로컬 도구로는 유용, 원격 MCP면 정보 공개.

9. **취소 후 벡터 인덱스 일관성**  
   `_cancelled_result`는 BM25 재구축만 시도하고 벡터 쪽은 부분 상태일 수 있음. **(추정: 재현 테스트 필요)**

10. **기능 추가 후보 (제품)**  
    - 사용자/부서별 문서 ACL  
    - 감사 로그(누가 어떤 규정 질의를 했는지)  
    - 답변 피드백(좋아요/부정확) → 검색 품질 루프  
    - 조문 단위 deep-link 안정화  
    - 오프라인 패키지에 RAG(Ollama) 원클릭 가이드 강화

---

## 5. Recommended Fix Plan

### 1단계 — 즉시 (안정성·보안 핫픽스)

1. **`initialize` 단일 비행 가드** — 중복 인덱싱 차단, 진행 중 재요청 명확한 응답  
2. **검색 vs 인덱스 쓰기 정책** — 최소: 로딩 중 검색 503 또는 읽기 락/스냅샷  
3. **관리자 로그인 rate limit**  
4. **MCP reindex 토큰 필수화** (MCP 사용 시)  
5. **RAG 메시지/history 길이 상한**  
6. **스트리밍 가드레일 정책 결정** — 프론트 `refused` 시 본문 폐기 또는 서버 retract 이벤트

### 2단계 — 안정성 개선

1. ZIP 실제 바이트 카운트  
2. `/api/init`와 `/api/sync/start` 경로 검증 통일  
3. API 500 응답에서 내부 예외 문자열 제거  
4. `llm.health()` TTL 캐시 및 retrieval_only 일관 동작  
5. 대화 API 소유권/최소 인증  
6. FAISS 캐시 무결성·권한 점검

### 3단계 — 구조 개선

1. 인덱스 **불변 스냅샷 스왑** 아키텍처 (검색 무중단 재인덱싱)  
2. RAG 대화 멀티유저 모델 (세션/사용자 FK)  
3. 앱 팩토리/전역 싱글톤 테스트 격리  
4. CLAUDE.md/README 구조·테스트 수 SSOT 정리  
5. (선택) 읽기 API 인증 계층, 감사 로그, ACL

---

## 6. Test Recommendations

현재 **92 tests collected**. 보안은 보안·ZIP·검색 필터·RAG 스모크·정규화에 강하나, 동시성·대화 격리·스트리밍 가드레일·MCP 토큰은 약하다.

### 6.1 반드시 추가

| 테스트 | 목적 |
|--------|------|
| `test_initialize_rejects_when_loading` | 중복 sync submit 거부 |
| `test_search_during_reindex_*` | 로딩 중 검색 정책(503 또는 일관 스냅샷) 고정 |
| `test_rag_stream_refused_not_showing_raw` 또는 서버 이벤트 계약 | 가드레일 거부 시 클라이언트 계약 |
| `test_rag_message_max_length` | 과대 메시지 400 |
| `test_admin_auth_rate_limited` | 연속 실패 시 429 |
| `test_mcp_reindex_requires_token` | 빈 토큰 시 거부 |
| `test_conversation_delete_without_auth_policy` | 의도한 접근 정책 문서화·고정 |
| `test_zip_copy_enforces_actual_bytes` | 메타 조작 zip 차단 |

### 6.2 보강 권장

| 테스트 | 목적 |
|--------|------|
| `process_single_file` ∥ `search` 동시 스레드 | 레이스 회귀 |
| `remove_file_from_index` 후 BM25+vector 정합 (일부 있음) + 검색 결과 0건 확인 | 삭제 일관성 |
| `init` vs `sync/start` 동일 path traversal 케이스 | 경로 정책 통일 |
| `build_messages` history 길이/role 제한 | 주입·DoS |
| `guardrails.apply` confidence 경계값 테이블 | 회귀 |
| 통합: 업로드 → 검색 히트 → 삭제 → 검색 미스 | E2E |
| `settings.json` 손상/부분 JSON | 설정 로드 실패 안전 |

### 6.3 문서/회귀

- CI에서 `pytest -q` 기대 개수를 **92+**로 문서와 동기화  
- `scripts/evaluate_search_quality.py` golden을 PR 게이트에 선택 포함  
- CodeGraph blast radius 상 `admin_required`, `clear_index`, `ConversationStore` 등 **테스트 미표기 심볼**부터 커버

---

## Appendix A. CodeGraph 중심 호출 맵 (요약)

```
run.py / server_gui
  └─ create_app
       ├─ db.init_db
       ├─ search_bp → rate_limiter, search_queue, qa_system.search
       ├─ files_bp → file_path_resolver, preview_service, process_single_file
       ├─ system_bp → verify_admin_password, initialize, load_model
       └─ rag_bp → RAGPipeline → HybridRetriever(qa_system) → AnswerGenerator → Guardrails

qa_system.initialize → executor → process_documents(_lock) → index lifecycle / FAISS cache
mcp.tools → qa_system / RAGPipeline (Flask 우회)
```

## Appendix B. 감사 방법 메모

- 1차: README.md, CLAUDE.md  
- 2차: CodeGraph `codegraph_explore` (엔트리, 인증, 검색, 업로드, RAG, DB, MCP)  
- 3차: 핵심 라우트 파일 정독, `pytest --collect-only` (92)  
- 코드를 수정하지 않음. 본 파일만 산출물.

---

---

## Remediation Log (2026-07-15)

감사 권고 구현 반영 요약. 전 스위트 `python -m pytest -q` → **104 passed**.

| 권고 | 상태 | 주요 변경 |
|------|------|-----------|
| H-1 검색/인덱스 race | 반영 | `HybridSearchService`가 `qa._lock` 하에서 검색; 인덱싱 중 차단 옵션 |
| H-2 initialize 중복 | 반영 | 단일 비행 가드 (`SYNC_ALREADY_RUNNING`) |
| H-3 대화 소유권 | 반영 | 세션 스코프 `rag_conversation_ids` + admin 전체 조회 |
| H-4 스트림 가드레일 | 반영 | `replace` SSE 이벤트 + 프론트 교체 |
| H-5 로그인 rate limit | 반영 | `_admin_auth_limiter` (분당 기본 20) |
| H-6 MCP reindex 토큰 | 반영 | 토큰 미설정 시 거부 |
| H-7 FAISS 무결성 | 부분 | `docs`/`faiss` integrity 메타 검증 후 로드 |
| M-1 입력 상한 | 반영 | 검색/RAG 메시지·history 길이 제한 |
| M-2 ZIP 실제 바이트 | 반영 | 해제 시 바이트 카운트 |
| M-3 경로 검증 통일 | 반영 | `validate_folder_path` → init/sync 공용 |
| M-4 예외 노출 | 반영 | 500/`SEARCH_FAILED`/RAG error 일반 메시지 |
| M-5 history 주입 | 반영 | role·길이 sanitize |
| M-6 health 캐시 | 반영 | `LLM_HEALTH_CACHE_TTL` |
| L-1 문서 정합 | 부분 | CLAUDE.md 경로·테스트 수 갱신 |
| 테스트 | 반영 | `tests/test_audit_hardening.py`, `tests/conftest.py` |

미완/후속(3단계 성격):

- 인덱스 **불변 스냅샷 스왑**(무중단 재인덱싱) — 현 구조는 락+인덱싱 중 검색 차단
- 대화 멀티유저 DB 스키마(사용자 FK) — 현재는 세션 스코프
- 읽기 API 인증 계층(배포 정책 의존)
- FAISS pickle 자체를 대체하는 완전 안전 직렬화

*End of Project Audit*
