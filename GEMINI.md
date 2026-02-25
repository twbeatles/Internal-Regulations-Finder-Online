# GEMINI.md - AI Assistant Context for Internal Regulations Search System

> 이 문서는 Gemini AI가 프로젝트를 빠르게 이해하고 효과적으로 지원할 수 있도록 핵심 컨텍스트를 제공합니다.

## 📋 Quick Summary

| 항목 | 내용 |
|------|------|
| **프로젝트** | 사내 규정 검색기 v2.6.1 |
| **목적** | AI 기반 하이브리드 검색 (Vector + BM25) |
| **언어** | Python 3.10+ (3.14 호환) |
| **프레임워크** | Flask + PyTorch + LangChain + FAISS |
| **DB** | SQLite (WAL 모드) |
| **실행** | 웹 서버 (localhost:8080) |

---

## 🏗️ Project Structure

```
Internal-Regulations-Finder-Online-main/
├── run.py                  # 콘솔 서버 엔트리포인트
├── server_gui.py           # GUI 서버 (PyQt6, 시스템 트레이)
├── download_models.py      # 오프라인용 모델 다운로드
├── app/                    # Flask 앱 패키지
│   ├── __init__.py         # 앱 팩토리 (create_app)
│   ├── config.py           # AppConfig 설정 클래스
│   ├── constants.py        # 상수, 에러 메시지, 정규식 패턴
│   ├── exceptions.py       # 커스텀 예외 클래스 계층
│   ├── utils.py            # 유틸리티 (TaskResult, FileInfo, MemoryMonitor)
│   ├── routes/             # API 라우트
│   │   ├── api_search.py   # 검색 API (/api/search)
│   │   ├── api_files.py    # 파일/태그/개정 관리 API
│   │   ├── api_system.py   # 시스템/동기화 API
│   │   └── main_routes.py  # 메인 페이지 라우트
│   └── services/           # 비즈니스 로직 서비스
│       ├── search.py       # 하이브리드 검색 엔진 (RegulationQASystem)
│       ├── document.py     # 문서 추출/파싱/분할/비교
│       ├── db.py           # SQLite 싱글톤 DB 관리
│       ├── file_manager.py # RevisionTracker, FolderWatcher
│       ├── metadata.py     # TagManager
│       └── embeddings_backends/  # 임베딩 백엔드 (v2.6 신규)
│           ├── __init__.py       # create_embeddings export
│           ├── factory.py        # 백엔드 선택 팩토리
│           ├── torch_backend.py  # PyTorch/HuggingFace 백엔드
│           └── onnx_backend.py   # ONNX Runtime 백엔드
├── static/                 # 프론트엔드 정적 파일
│   ├── app.js              # SPA 스타일 클라이언트 (v1.7, 3800+ lines)
│   ├── style.css           # CSS 스타일 (다크/라이트 테마)
│   └── sw.js               # PWA 서비스 워커
├── templates/              # HTML 템플릿
│   ├── index.html          # 메인 페이지
│   └── admin.html          # 관리자 페이지
├── config/                 # 런타임 설정
│   ├── settings.json       # 사용자 설정
│   └── regulations.db      # SQLite 데이터베이스
├── uploads/                # 업로드된 문서
├── revisions/              # 개정 이력 파일
└── models/                 # AI 모델 캐시
```

---

## 🔑 Core Components

### 1. Search Engine (`app/services/search.py`)

**핵심 클래스**:
| 클래스 | 역할 |
|--------|------|
| `BM25Light` | 경량 BM25 검색 (사전 컴파일 토큰화, RLock) |
| `SearchCache` | LRU 캐시 (적응형 TTL, OrderedDict) - v2.6.1 |
| `RateLimiter` | IP 기반 요청 제한 (deque 구조 O(1)) |
| `SearchQueue` | 동시 검색 제한 (Semaphore) |
| `SearchHistory` | 검색 히스토리 (DB 연동) |
| `RegulationQASystem` | 하이브리드 검색 통합 시스템 |

**전역 인스턴스**:
```python
qa_system = RegulationQASystem()
search_queue = SearchQueue(max_concurrent=10)
rate_limiter = RateLimiter(requests_per_minute=300)
```

**주요 메서드**:
```python
qa_system.load_model(model_name, offline_mode, local_model_path)
qa_system.initialize(folder_path)  # 문서 로드 및 인덱싱
qa_system.search(query, k=5, hybrid=True)
qa_system.process_single_file(file_path)  # 단일 파일 즉시 인덱싱
```

### 2. Document Processing (`app/services/document.py`)

| 클래스 | 역할 |
|--------|------|
| `DocumentExtractor` | 텍스트 추출 (TXT, DOCX, PDF, XLSX, HWP) |
| `ArticleParser` | 조문 구조 파싱 (제X조, 제X장, 제X절) |
| `DocumentSplitter` | 청킹 (chunk_size=800, overlap=80) |
| `DocumentComparator` | 문서 버전 비교 (diff) |
| `TextHighlighter` | 검색어 하이라이트 |

**지원 파일 형식**:
```python
SUPPORTED_EXTENSIONS = {'.txt', '.docx', '.pdf', '.xlsx', '.xls', '.hwp'}
```

### 3. Database (`app/services/db.py`)

**싱글톤 DBManager**:
- 스레드 로컬 연결 관리 (`threading.local`)
- WAL 모드 + PRAGMA 최적화 (cache_size=4MB, mmap_size=256MB)
- 테이블: `tags`, `revisions`, `search_history`

```python
from app.services.db import db

db.execute(query, args)      # INSERT/UPDATE/DELETE
db.fetchall(query, args)     # SELECT 다중 결과
db.fetchone(query, args)     # SELECT 단일 결과
db.close()                   # 현재 스레드 연결 닫기
```

### 4. Utilities (`app/utils.py`)

| 클래스/함수 | 역할 |
|-------------|------|
| `TaskResult` | 작업 결과 데이터 클래스 |
| `FileInfo` | 파일 정보 데이터 클래스 |
| `FileStatus` | 파일 상태 Enum |
| `MemoryMonitor` | 메모리 모니터링 |
| `api_success()` | 표준 성공 응답 |
| `api_error()` | 표준 에러 응답 |

### 5. Exceptions (`app/exceptions.py`)

```python
RegSearchError              # 기본 예외
├── ModelError              # 모델 관련
│   ├── ModelNotLoadedError
│   ├── ModelLoadError
│   └── ModelOfflineError
├── DocumentError           # 문서 관련
│   ├── DocumentNotFoundError
│   ├── DocumentExtractionError
│   └── DocumentTypeError
├── SearchError             # 검색 관련
│   ├── SearchTimeoutError
│   ├── SearchRateLimitError
│   └── SearchQueueFullError
└── AuthError               # 인증 관련
```

### 6. Configuration (`app/config.py`)

```python
class AppConfig:
    SERVER_PORT = 8080
    SERVER_THREADS = 32
    
    # AI 모델
    DEFAULT_MODEL = "SNU SBERT (고성능)"
    AVAILABLE_MODELS = {
        "SNU SBERT (고성능)": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        "BM-K Simal (균형)": "BM-K/ko-simal-roberta-base",
        "JHGan SBERT (빠름)": "jhgan/ko-sbert-nli"
    }
    
    # 검색 설정
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 80
    VECTOR_WEIGHT = 0.7
    BM25_WEIGHT = 0.3
    
    # 성능
    SEARCH_CACHE_SIZE = 1000
    MAX_CONCURRENT_SEARCHES = 10
    RATE_LIMIT_PER_MINUTE = 300
    
    # 오프라인 모드
    OFFLINE_MODE = False
    LOCAL_MODEL_PATH = ""
    
    # 임베딩 백엔드 (v2.6 신규)
    EMBED_BACKEND = "onnx_fp32"  # "torch" | "onnx_fp32" | "onnx_int8"
    EMBED_NORMALIZE = True   # L2 정규화 여부
    
    # 성능 최적화 (v2.6.1 신규)
    SEARCH_CACHE_TTL = 600         # 캐시 TTL (10분)
    ADAPTIVE_CACHE_TTL = True      # 적응형 TTL 활성화
    PARALLEL_SEARCH = True         # 병렬 검색 활성화
    COMPRESS_MIN_SIZE = 500        # Gzip 압축 임계값
    MAX_CONTENT_PREVIEW = 1500     # 콘텐츠 미리보기 최대 길이
```

---

## 🌐 API Reference

### Search API
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/search` | 하이브리드 검색 `{query, k, hybrid}` |
| GET | `/api/search/history` | 검색 히스토리 |
| GET | `/api/search/suggest` | 자동완성 `?q=검색어` |
| POST | `/api/cache/clear` | 캐시 초기화 |

### Files API
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/files` | 파일 목록 |
| GET | `/api/files/names` | 파일명 목록 |
| POST | `/api/upload` | 파일 업로드 + 인덱싱 |
| DELETE | `/api/files/<filename>` | 파일 삭제 |
| GET/POST/DELETE | `/api/tags` | 태그 관리 |
| GET/POST | `/api/revisions` | 개정 이력 |

### System API
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | 서버 상태 |
| GET | `/api/health` | 헬스 체크 |
| GET | `/api/stats` | 통계 (캐시, 메모리) |
| GET | `/api/models` | 모델 목록 |
| POST | `/api/sync/start` | 동기화 시작 |
| POST | `/api/sync/stop` | 동기화 중지 |

---

## 🎨 Frontend (`static/app.js`)

**주요 모듈**:
| 모듈 | 역할 |
|------|------|
| `Logger` | 프로덕션 안전 로깅 |
| `PerformanceUtils` | debounce, throttle, cleanup |
| `API` | API 클라이언트 (중복 요청 방지, 재시도) |
| `BookmarkManager` | 북마크 (LocalStorage) |
| `Toast` | 토스트 알림 |
| `ThemeManager` | 다크/라이트 테마 |
| `KeyboardShortcuts` | 단축키 관리 |
| `ExportResults` | 결과 내보내기 (TXT, MD, JSON) |

**단축키**:
| 키 | 기능 |
|----|------|
| `Ctrl+K` | 검색창 포커스 |
| `J/K` | 결과 탐색 |
| `T` | 테마 전환 |
| `R` | 읽기 모드 |
| `Esc` | 모달 닫기 |

---

## ⚠️ Development Guidelines

### 1. Thread Safety
```python
# DBManager: 스레드 로컬 사용
# BM25Light: RLock 보호
# SearchCache: Lock + OrderedDict
# SearchQueue: Semaphore
# api_files.py: file_lock
```

### 2. Error Handling
```python
from app.constants import ErrorMessages
from app.exceptions import DocumentNotFoundError

raise DocumentNotFoundError(path)

# API 응답
return api_error(ErrorMessages.FILE_NOT_FOUND, status_code=404)
```

### 3. Lazy Imports
```python
# PyTorch/LangChain은 필요 시 로드
_lazy_import_langchain()
```

### 4. XSS Prevention (Frontend)
```javascript
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
```

### 5. Path Traversal Prevention
```python
normalized_path = os.path.normpath(os.path.realpath(folder_path))
if any(p in folder_path for p in ['..', '//']):
    return api_error('유효하지 않은 경로')
```

### 6. Database Queries
```python
# 항상 파라미터 바인딩 사용
db.execute("SELECT * FROM tags WHERE filename = ?", (filename,))
```

---

## 🚀 Quick Commands

```bash
# 개발 서버 실행
python run.py

# GUI 서버 실행
python server_gui.py

# 패키지 설치
pip install -r requirements.txt        # 전체
pip install -r requirements_lite.txt   # 경량 (BM25만)

# 빌드
pyinstaller regulation_search_gui.spec --clean           # AI 포함
pyinstaller regulation_search_ultra_lite_gui.spec --clean  # 경량

# 모델 다운로드 (오프라인 준비)
python download_models.py
```

---

## 📊 Performance Metrics

| Metric | Value (v2.6.2) |
|--------|----------------|
| Search Response | ~80ms (↓ 47%) |
| Cache Hit Rate | ~90% (↑ 12%) |
| Memory Usage | ~600MB (with AI) |
| Max Concurrent | 10 searches |
| Rate Limit | 300 req/min/IP |
| Response Compression | ~75% reduction (Gzip) |
| Cache TTL | 600s (adaptive, max 2x) |
| Cache Key | `query + k + hybrid + sort_by` |

### Frontend Runtime Notes
- Admin bootstrap is guarded by `window.__adminBootstrapped` (single init path).
- Admin polling uses one timer + Visibility API.
- jsPDF/AutoTable are lazy-loaded at export-time (local vendor first, CDN fallback).

---

## 🧪 Perf Commands

```bash
pytest -q
python scripts/perf_smoke.py
python scripts/perf_smoke.py --base-url http://127.0.0.1:8080 --query "휴가 규정"
```

Default benchmark scenario: warmup 30, measure 200, concurrency 1/5/10.

---

## 🔧 Troubleshooting

| 문제 | 해결 |
|------|------|
| 모델 로드 실패 | `OFFLINE_MODE=True` + `LOCAL_MODEL_PATH` 설정 |
| 검색 느림 | 캐시 크기 증가 (`SEARCH_CACHE_SIZE`) |
| 메모리 부족 | 경량 버전 사용 (`requirements_lite.txt`) |
| HWP 추출 실패 | `olefile` 패키지 설치 확인 |
| OCR 필요 | Tesseract 설치 + `pytesseract` 패키지 |
| Rate Limit | `RATE_LIMIT_PER_MINUTE` 증가 |
| 동시 검색 제한 | `MAX_CONCURRENT_SEARCHES` 조정 |

---

## 📝 Code Review Checklist

- [ ] 에러 메시지: `ErrorMessages` 클래스 사용
- [ ] 상수: `constants.py` 정의 값 활용
- [ ] 예외: `exceptions.py` 적절한 클래스 사용
- [ ] DB: 파라미터 바인딩 사용
- [ ] 스레드: Lock/RLock/Semaphore 적절히 사용
- [ ] 프론트엔드: `escapeHtml()` XSS 방지
- [ ] 캐시: 파일 변경 시 무효화 로직 확인
- [ ] API 응답: `api_success()` / `api_error()` 사용
- [ ] 로깅: `logger` 객체 사용 (print 금지)
- [ ] 경로: Path Traversal 방지 검증

---

## 📌 v2.7 반영 노트 (2026-02-20)

### Security
- Admin password fallback (`"admin"`) removed; auth is fail-closed when unset.
- Mutating endpoints now require admin auth:
  - `POST /api/upload`
  - `POST /api/upload/folder`
  - `POST /api/revisions`
  - `POST /api/tags/set`
  - `POST /api/tags/auto`
  - `POST /api/cache/clear`
- Search result rendering migrated to client-side safe highlight (escape-first).
- Revision save path hardened with sanitized names + revisions-root boundary checks.

### API Compatibility
- `/api/status` returns both `progress` and `load_progress`.
- `/api/sync/status` returns both legacy keys and `status` object.
- `/api/revisions` GET returns both `history` and `revisions`.
- `/api/tags/auto` returns both `suggested_tags` and `tags`.

### File Identity
- Introduced `file_id` (normalized absolute path hash).
- Search responses and file list payloads include `file_id`.
- Search request supports `filter_file_id` with `filter_file` compatibility.
- New by-id routes for preview/download/delete/versions/compare.

### Functional Fixes
- ZIP folder upload implemented: `POST /api/upload/folder` (zip-slip guarded).
- Download route restored (by-id + legacy filename route).
- Frontend upload extension filter aligned with backend (`txt/docx/pdf/xlsx/xls/hwp`).
- PWA icon paths aligned via `static/icons/icon-192.png`, `static/icons/icon-512.png`.

### v2.8 Additions (2026-02-25)
- CORS is allowlist-based via `CORS_ALLOWED_ORIGINS`; credentials are no longer reflected to arbitrary origins.
- Session cookie policy is explicitly enforced (`HttpOnly`, `SameSite`, `Secure`).
- `/api/search/history` now returns `success`, normalized `popular[{query,count}]`, and `popular_legacy`.
- `/api/search/suggest`, `/api/status`, `/api/health` now include `success` envelopes.
- File delete policy defaults to `index_only`; physical delete requires `delete_source=true`.
- ZIP upload adds hard limits (`max_entries`, `max_uncompressed_bytes`, `max_single_file_bytes`).
- `/api/models` defaults `reindex=true` and reports whether reindex was actually triggered.
- `/api/sync/stop` is implemented using cancellation events (no longer TODO).
- Service Worker caches only GET allowlisted APIs and never caches auth/admin or mutating API calls.
