# CLAUDE.md - AI Assistant Context for Internal Regulations Search System

> 이 문서는 Claude AI가 프로젝트를 빠르게 이해하고 효과적으로 지원할 수 있도록 핵심 컨텍스트를 제공합니다.

## 📋 프로젝트 개요

**프로젝트명**: 사내 규정 검색기 v2.6  
**목적**: AI 기반 하이브리드 검색 시스템 (Vector + BM25)으로 사내 규정 문서를 검색  
**기술 스택**: Flask + PyTorch + LangChain + FAISS + SQLite  
**실행 방식**: 웹 서버 (콘솔 또는 GUI/시스템 트레이)

---

## 🏗️ 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                         Client (Browser)                     │
│  static/app.js - SPA 스타일 프론트엔드 (v1.7)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Flask Server                            │
│  run.py (콘솔) / server_gui.py (PyQt6 GUI)                   │
├─────────────────────────────────────────────────────────────┤
│  app/__init__.py        - Flask 앱 팩토리                    │
│  app/config.py          - 설정 (모델, 청킹, 동시성)          │
│  app/constants.py       - 에러 메시지, 상수, 정규식           │
│  app/exceptions.py      - 커스텀 예외 클래스                 │
│  app/utils.py           - 유틸리티 (TaskResult, FileInfo 등) │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│    Routes       │ │    Services     │ │    Database     │
│ api_search.py   │ │ search.py       │ │ db.py (SQLite)  │
│ api_files.py    │ │ document.py     │ │                 │
│ api_system.py   │ │ file_manager.py │ │                 │
│ main_routes.py  │ │ metadata.py     │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

---

## 🔑 핵심 컴포넌트 상세

### 1. 검색 엔진 (`app/services/search.py`)

| 클래스 | 역할 | 주요 메서드 |
|--------|------|-------------|
| `BM25Light` | 경량 BM25 구현 (스레드 안전, RLock) | `fit()`, `search()`, `_score_optimized()` |
| `SearchCache` | LRU 기반 검색 캐시 (OrderedDict) | `get()`, `set()`, `invalidate_by_file()` |
| `RateLimiter` | IP 기반 요청 제한 (deque 기반 O(1)) | `is_allowed()`, `get_stats()` |
| `SearchQueue` | 동시 검색 제한 (Semaphore) | `acquire()`, `release()` |
| `SearchHistory` | 검색 히스토리 (DB 연동) | `add()`, `get_recent()`, `suggest()` |
| `RegulationQASystem` | 하이브리드 검색 통합 시스템 | `load_model()`, `initialize()`, `search()` |

**중요 인스턴스**:
```python
qa_system = RegulationQASystem()  # 전역 검색 시스템
search_queue = SearchQueue(max_concurrent=10)
rate_limiter = RateLimiter(requests_per_minute=300)
```

**검색 가중치**:
```python
VECTOR_WEIGHT = 0.7  # 의미 기반 검색 (FAISS)
BM25_WEIGHT = 0.3    # 키워드 기반 검색 (BM25Light)
```

### 2. 문서 처리 (`app/services/document.py`)

| 클래스 | 역할 | 주요 메서드 |
|--------|------|-------------|
| `DocumentExtractor` | 다양한 포맷 텍스트 추출 | `extract()`, `_extract_docx()`, `_extract_hwp()` |
| `ArticleParser` | 규정 문서의 조문 구조 파싱 | `parse_articles()`, `search_article()` |
| `DocumentSplitter` | 청킹 처리 | `split()`, `split_by_articles()` |
| `DocumentComparator` | 문서 버전 비교 (diff) | `compare()` |
| `TextHighlighter` | 검색어 하이라이트 | `highlight()`, `extract_keywords()` |

**지원 파일 형식**:
```python
SUPPORTED_EXTENSIONS = {'.txt', '.docx', '.pdf', '.xlsx', '.xls', '.hwp'}
```

### 3. 파일 관리 (`app/services/file_manager.py`)

| 클래스 | 역할 |
|--------|------|
| `RevisionTracker` | 규정 개정 이력 관리 (SQLite + 파일 시스템) |
| `FolderWatcher` | watchdog 기반 폴더 변경 감지 |

### 4. 데이터베이스 (`app/services/db.py`)

- **싱글톤 패턴** + **스레드 로컬** 연결 관리
- **WAL 모드** + PRAGMA 최적화 (cache_size=4MB, mmap_size=256MB)
- **테이블**: `tags`, `revisions`, `search_history`

```python
from app.services.db import db  # 전역 싱글톤

db.execute(query, args)       # INSERT/UPDATE/DELETE
db.fetchall(query, args)      # SELECT 다중 결과
db.fetchone(query, args)      # SELECT 단일 결과
```

### 5. 유틸리티 (`app/utils.py`)

| 클래스/함수 | 역할 |
|-------------|------|
| `TaskResult` | 작업 결과 데이터 클래스 (success, message, data) |
| `FileInfo` | 파일 정보 데이터 클래스 (path, status, chunks) |
| `FileStatus` | 파일 상태 Enum (PENDING, PROCESSING, SUCCESS, FAILED, CACHED) |
| `MemoryMonitor` | 메모리 사용량 모니터링 (psutil 연동) |
| `CustomJSONEncoder` | NumPy/Set/Enum JSON 직렬화 |
| `api_success()` | 표준 성공 응답 생성 |
| `api_error()` | 표준 에러 응답 생성 |
| `api_paginated()` | 페이지네이션 응답 생성 |

### 6. 예외 클래스 (`app/exceptions.py`)

```python
RegSearchError                   # 기본 예외
├── ModelError
│   ├── ModelNotLoadedError     # 모델 미로드
│   ├── ModelLoadError          # 모델 로드 실패
│   └── ModelOfflineError       # 오프라인 모델 없음
├── DocumentError
│   ├── DocumentNotFoundError   # 문서 파일 없음
│   ├── DocumentExtractionError # 텍스트 추출 실패
│   ├── DocumentTypeError       # 지원하지 않는 포맷
│   └── DocumentEmptyError      # 빈 문서
├── SearchError
│   ├── SearchTimeoutError      # 검색 타임아웃
│   ├── SearchRateLimitError    # 요청 제한 초과
│   └── SearchQueueFullError    # 검색 큐 포화
├── FolderError
│   ├── FolderNotFoundError     # 폴더 없음
│   └── FolderNotInitializedError # 폴더 미초기화
└── AuthError
    ├── AuthenticationError     # 인증 실패
    └── AuthorizationError      # 권한 부족
```

---

## 📂 디렉토리 구조

```
├── run.py                  # 콘솔 엔트리포인트
├── server_gui.py           # GUI 서버 (PyQt6, 시스템 트레이)
├── download_models.py      # 오프라인용 모델 다운로드
├── download_static.py      # 오프라인용 정적 리소스(JS/Font) 다운로드 (v2.6.1)
├── app/
│   ├── __init__.py         # Flask 앱 팩토리 (create_app)
│   ├── config.py           # AppConfig 설정 클래스
│   ├── constants.py        # ErrorMessages, SuccessMessages, Limits, Patterns
│   ├── exceptions.py       # 커스텀 예외 클래스
│   ├── utils.py            # TaskResult, FileInfo, MemoryMonitor, api_*
│   ├── routes/
│   │   ├── api_search.py   # POST /api/search, /api/search/history
│   │   ├── api_files.py    # /api/files, /api/upload, /api/tags, /api/revisions
│   │   ├── api_system.py   # /api/status, /api/sync, /api/models
│   │   └── main_routes.py  # GET /, /admin
│   └── services/
│       ├── search.py       # RegulationQASystem, BM25Light, SearchCache
│       ├── document.py     # DocumentExtractor, ArticleParser, DocumentSplitter
│       ├── db.py           # DBManager 싱글톤
│       ├── file_manager.py # RevisionTracker, FolderWatcher
│       ├── metadata.py     # TagManager
│       └── embeddings_backends/  # 임베딩 백엔드 (v2.6 신규)
│           ├── __init__.py       # create_embeddings export
│           ├── factory.py        # 백엔드 선택 팩토리 (torch/onnx)
│           ├── torch_backend.py  # PyTorch/HuggingFace 백엔드
│           └── onnx_backend.py   # ONNX Runtime 백엔드
├── static/
│   ├── app.js              # 프론트엔드 SPA (3313 lines 기준, v1.7)
│   ├── style.css           # CSS 스타일 (다크/라이트 테마)
│   └── sw.js               # PWA 서비스 워커
├── templates/
│   ├── index.html          # 메인 페이지
│   └── admin.html          # 관리자 페이지
├── config/
│   ├── settings.json       # 런타임 설정 (folder, offline_mode 등)
│   └── regulations.db      # SQLite 데이터베이스
├── uploads/                # 업로드된 문서 저장
├── revisions/              # 개정 이력 파일 저장
└── models/                 # 다운로드된 AI 모델 캐시
```

---

## 🌐 API 엔드포인트 상세

### Search API (`api_search.py`)
| 메서드 | 경로 | 설명 | 요청 |
|--------|------|------|------|
| POST | `/api/search` | 하이브리드 검색 | `{query, k, hybrid, sort_by, filter_file}` |
| GET | `/api/search/history` | 검색 히스토리 | `?limit=10` |
| GET | `/api/search/suggest` | 자동완성 제안 | `?q=검색어&limit=8` |
| POST | `/api/cache/clear` | 검색 캐시 초기화 | - |

### Files API (`api_files.py`)
| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/files` | 파일 목록 조회 |
| GET | `/api/files/names` | 파일명 목록 (필터용) |
| POST | `/api/upload` | 파일 업로드 + 자동 인덱싱 |
| DELETE | `/api/files/<filename>` | 파일 삭제 |
| DELETE | `/api/files/all` | 전체 파일 삭제 |
| GET/POST/DELETE | `/api/tags` | 태그 관리 |
| GET | `/api/revisions` | 개정 이력 조회 |
| POST | `/api/revisions` | 개정 이력 저장 |
| POST | `/api/compare` | 문서 비교 (diff) |

### System API (`api_system.py`)
| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/status` | 서버 상태 (ready, loading, model) |
| GET | `/api/health` | 헬스 체크 |
| GET | `/api/stats` | 통계 (문서, 캐시, 메모리) |
| GET | `/api/models` | 사용 가능 모델 목록 |
| POST | `/api/sync/start` | 폴더 동기화 시작 |
| POST | `/api/sync/stop` | 동기화 중지 |
| POST | `/api/process` | 문서 재처리 |
| POST | `/api/verify_password` | 관리자 비밀번호 확인 |

---

## 🎨 프론트엔드 (`static/app.js`)

### 주요 모듈
| 모듈 | 역할 |
|------|------|
| `Logger` | 프로덕션 안전 로깅 (dev에서만 debug) |
| `PerformanceUtils` | debounce, throttle, AbortController 관리 |
| `RippleEffect` | 버튼 리플 효과 (이벤트 위임) |
| `SkeletonLoading` | 로딩 스켈레톤 UI 생성 |
| `NetworkStatus` | 네트워크 상태 감지 (online/offline) |
| `KeyboardShortcuts` | 단축키 관리 (Ctrl+K, J/K, T, Esc) |
| `ReaderMode` | 읽기 모드 모달 |
| `HighlightNavigator` | 하이라이트 탐색 (N/P) |
| `SearchResultNavigator` | 검색 결과 키보드 탐색 |
| `ExportResults` | 결과 내보내기 (TXT, MD, JSON) |
| `API` | API 클라이언트 (중복 요청 방지, 재시도) |
| `BookmarkManager` | 북마크 관리 (LocalStorage) |
| `Toast` | 토스트 알림 |
| `ThemeManager` | 다크/라이트 테마 관리 |
| `AppState` | 앱 상태 관리 |

### 단축키
| 키 | 기능 |
|----|------|
| `Ctrl+K` | 검색창 포커스 |
| `J/K` 또는 `↓/↑` | 결과 탐색 |
| `N/P` | 하이라이트 탐색 |
| `R` | 읽기 모드 |
| `T` | 테마 전환 |
| `Esc` | 모달 닫기 |

---

## ⚠️ 개발 시 주의사항

### 1. 스레드 안전성
```python
# DBManager: 스레드 로컬 연결 (각 스레드별 별도 연결)
# BM25Light: threading.RLock 사용
# SearchCache: threading.Lock + OrderedDict
# SearchQueue: threading.Semaphore
# file_lock: 파일 작업 시 threading.Lock
```

### 2. 지연 로딩 (Lazy Import)
```python
# PyTorch/LangChain은 필요 시 로드 (서버 시작 시간 단축)
def _lazy_import_langchain():
    global CharacterTextSplitter, Document, HuggingFaceEmbeddings, FAISS
    # 지연 임포트...
```

### 3. 오프라인 모드
```python
# settings.json
{
    "offline_mode": true,
    "local_model_path": "./models/snunlp--KR-SBERT-V40K-klueNLI-augSTS"
}

# 환경 변수
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
```

### 4. 에러 처리 패턴
```python
from app.constants import ErrorMessages
from app.exceptions import DocumentNotFoundError

# 적절한 예외 클래스 사용
raise DocumentNotFoundError(path)

# API 응답
return jsonify({
    'success': False,
    'message': ErrorMessages.FILE_NOT_FOUND,
    'error_code': 'FILE_NOT_FOUND'
}), HttpStatus.NOT_FOUND
```

### 5. XSS 보안 (프론트엔드)
```javascript
function escapeHtml(str) {
    if (!str) return '';
    // v2.6.1 최적화: Regex 치환으로 DOM 생성 비용 제거
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
```

### 6. 오프라인 모드 (폐쇄망)
```python
# settings.json
{
    "offline_mode": true,
    "local_model_path": "./models/..."
}

# 리소스 폴백 (index.html)
# <script src="CDN" onerror="this.src='/static/vendor/...'"></script>
```

### 7. Path Traversal 방지 (api_system.py)
```python
# 경로 정규화
normalized_path = os.path.normpath(os.path.realpath(folder_path))

# 위험 패턴 차단
if any(p in folder_path for p in ['..', '//']):
    return api_error('유효하지 않은 경로 형식입니다')
```

### 7. 파일 잠금 패턴
```python
# 타임아웃 있는 잠금
if not acquire_file_lock(timeout=LOCK_TIMEOUT):
    return api_error('서버가 바쁩니다')

try:
    # 파일 작업...
finally:
    file_lock.release()
```

---

## 🚀 실행 방법

```bash
# 개발 서버 (콘솔)
python run.py

# GUI 서버 (시스템 트레이)
python server_gui.py

# 브라우저 접속
http://localhost:8080
http://localhost:8080/admin  # 관리자
```

---

## 📦 빌드

```bash
# AI 포함 GUI 버전 (500-800MB)
pyinstaller regulation_search_gui.spec --clean

# 경량 버전 - BM25만 (60-100MB)
pyinstaller regulation_search_ultra_lite_gui.spec --clean

# 오프라인용 모델 다운로드
python download_models.py
```

---

## 📊 성능 특성

| 항목 | 값 |
|------|-----|
| 검색 응답 시간 | ~80ms (환경 의존) |
| 캐시 히트율 | ~90% |
| 메모리 사용량 | ~0.6GB (AI 모델 로드 시) |
| 동시 검색 제한 | 10개 (SearchQueue) |
| 분당 요청 제한 | 300회/IP (RateLimiter) |
| 캐시 TTL | 600초 (적응형 연장, 최대 2배) |
| 캐시 키 | `query + k + hybrid + sort_by` |

### 프론트엔드 성능 참고
- `static/app.js`는 관리자 초기화 경로를 단일화했으며 `window.__adminBootstrapped`로 중복 부팅을 방지함
- 관리자 상태 폴링은 단일 타이머 + Visibility API 기반으로 동작함
- PDF 내보내기 라이브러리(jsPDF/AutoTable)는 클릭 시점 지연 로딩됨

---

## 🧪 성능 측정 명령

```bash
pytest -q
python scripts/perf_smoke.py
python scripts/perf_smoke.py --base-url http://127.0.0.1:8080 --query "휴가 규정"
```

`scripts/perf_smoke.py` 기본 시나리오: 워밍업 30회, 측정 200회, 동시성 1/5/10.

---

## 🔧 설정 (`app/config.py`)

```python
class AppConfig:
    # 서버
    SERVER_PORT = 8080
    SERVER_THREADS = 32
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    
    # AI 모델
    DEFAULT_MODEL = "SNU SBERT (고성능)"
    AVAILABLE_MODELS = {
        "SNU SBERT (고성능)": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        "BM-K Simal (균형)": "BM-K/ko-simal-roberta-base",
        "JHGan SBERT (빠름)": "jhgan/ko-sbert-nli"
    }
    
    # 오프라인
    OFFLINE_MODE = False
    LOCAL_MODEL_PATH = ""
    
    # 검색
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 80
    VECTOR_WEIGHT = 0.7
    BM25_WEIGHT = 0.3
    
    # 성능
    MAX_WORKERS = 8
    SEARCH_CACHE_SIZE = 1000
    MAX_CONCURRENT_SEARCHES = 10
    RATE_LIMIT_PER_MINUTE = 300
    
    # 임베딩 백엔드 (v2.6 신규)
    EMBED_BACKEND = "onnx_fp32"  # "torch" | "onnx_fp32" | "onnx_int8"
    EMBED_NORMALIZE = True   # L2 정규화 여부
```

---

## 🔍 코드 수정 시 체크리스트

- [ ] 에러 메시지는 `constants.py`의 `ErrorMessages` 클래스 사용
- [ ] 새 상수는 `constants.py`에 정의
- [ ] 새 예외는 `exceptions.py`에 적절한 계층으로 추가
- [ ] DB 쿼리는 파라미터 바인딩 사용 (`?` 플레이스홀더)
- [ ] 스레드 안전성 검토 (Lock/RLock/Semaphore)
- [ ] 프론트엔드 XSS 방지 (`escapeHtml()`)
- [ ] 파일 변경 시 캐시 무효화 (`_search_cache.invalidate_by_file()`)
- [ ] API 응답은 `api_success()` / `api_error()` 헬퍼 사용
- [ ] 로깅은 `logger` 객체 사용 (print 금지)
- [ ] 경로 검증 (Path Traversal 방지)

---

## 📌 v2.7 구현 반영 메모 (2026-02-20)

### API/스키마
- `POST /api/search`는 `filter_file_id`를 지원하며 `filter_file`은 하위호환으로 유지.
- `/api/status`는 `progress`와 `load_progress`를 동시에 제공.
- `/api/sync/status`는 기존 필드와 함께 `status` 객체를 병행 제공.
- `/api/revisions` GET은 `history`와 `revisions`를 동시에 제공.
- `/api/tags/auto`는 `suggested_tags`와 `tags`를 동시에 제공.

### 파일 식별자 정책
- 파일 식별은 `filename`보다 `file_id`를 우선 사용.
- `file_id`는 정규화된 절대경로 기반 해시(`FileUtils.make_file_id`)로 생성.
- by-id 라우트:
  - `GET /api/files/by-id/<file_id>/preview`
  - `GET /api/files/by-id/<file_id>/download`
  - `DELETE /api/files/by-id/<file_id>`
  - `GET /api/files/by-id/<file_id>/versions`
  - `GET /api/files/by-id/<file_id>/versions/<version>`
  - `GET /api/files/by-id/<file_id>/versions/compare`

### 보안 정책
- 기본 관리자 비밀번호 fallback 제거(미설정 시 인증 실패).
- 상태 변경 API에 `admin_required` 적용:
  - `/api/upload`, `/api/upload/folder`, `/api/revisions` POST,
  - `/api/tags/set`, `/api/tags/auto`, `/api/cache/clear`
- 검색 결과 렌더링은 서버 HTML 신뢰 대신 클라이언트 escape 기반 하이라이트 사용.
- 리비전 파일 저장은 안전 파일명 + `revisions` 루트 내부 경로 검증 필수.

### Lite(BM25-only) 경로
- 임베딩 모델 미로드 상태에서도 캐시/검색 경로가 동작하도록 분기 강화.
- 벡터 캐시 로드는 `embedding_model && FAISS`일 때만 활성화.

### v2.8 추가 반영 (2026-02-25)
- CORS는 `CORS_ALLOWED_ORIGINS` allowlist 기반으로만 허용되며, API 응답에 CORS 허용/차단 로그가 남는다.
- 세션 쿠키 정책(`SESSION_COOKIE_HTTPONLY`, `SESSION_COOKIE_SAMESITE`, `SESSION_COOKIE_SECURE`)은 명시 설정으로 강제된다.
- `/api/search/history`는 `success`와 `popular_legacy`를 포함하고, `popular` 표준 포맷은 `[{query, count}]`다.
- `/api/search/suggest`, `/api/status`, `/api/health`는 `success` envelope를 사용한다.
- 삭제 API 기본 정책은 `index_only`, 원본 삭제는 `delete_source=true`일 때만 수행된다.
- `/api/upload/folder`는 ZIP 제한 파라미터(`max_entries`, `max_uncompressed_bytes`, `max_single_file_bytes`)를 지원한다.
- `/api/models`는 `reindex` 기본값이 `true`이며, 실제 재인덱스 트리거 결과를 응답한다.
- `/api/sync/stop`은 TODO가 아니라 실제 취소 이벤트를 통해 동기화 중단 요청을 처리한다.
- Service Worker는 `GET allowlist` API만 캐시하고 인증/관리/비GET 요청은 캐시하지 않는다.

### 정합성 보충 메모 (2026-03-01)
- `static/app.js` 라인수 표기는 점검 시점 기준 `3313`으로 갱신.
- 주요 spec 5종(`regulation_search_gui.spec`, `regulation_search_ultra_lite_gui.spec`, `regulation_search_onefile.spec`, `regulation_search_ultra_lite.spec`, `server_gui.spec`)은 `config` 폴더 전체 대신 `config/settings.example.json`만 포함.
- `python -m py_compile *.spec` 기준 spec 문법은 모두 정상.
