# 📚 사내 규정 검색기 v2.6

> AI 기반 하이브리드 검색 시스템 (Vector + BM25)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ v2.6.1 업데이트 (2026-01-18)

### 🚀 성능 최적화 (Advanced)
- **API 응답 최적화** - Gzip 압축, JSON 정밀도 제한, 콘텐츠 미리보기 제한
- **렌더링 가속** - HTML Resource Hints (preload/preconnect), Script defer, DOM 처리 최적화
- **정규식 캐싱** - 백엔드 TextHighlighter LRU 캐시 도입

### 🔌 완전 오프라인 지원
- **정적 자원 다운로더** - `download_static.py`로 jsPDF, AutoTable 등 필수 라이브러리 로컬 저장
- **자동 폴백 시스템** - CDN 연결 실패 시 로컬 리소스 자동 전환, 시스템 폰트 폴백
- **검증된 모델 다운로드** - `download_models.py`를 통한 안정적인 모델 준비

### 🧠 임베딩 백엔드 시스템
- **설정 가능한 임베딩 엔진** - torch / onnx_fp32 / onnx_int8 선택 가능
- **ONNX Runtime 지원** - 인텀 없이 경량 추론 가능
- **자동 Fallback** - ONNX 실패 시 torch로 자동 전환
- **GUI 설정** - 서버 GUI에서 백엔드 선택 UI 추가

## ✨ v2.6.2 업데이트 (2026-02)

### 🚀 성능 리팩토링
- **클라이언트 중복 제거** - 관리자 초기화/업로드/상태 갱신 레거시 블록 정리
- **관리자 부트스트랩 단일화** - `initAdmin()` idempotent 가드 + 단일 폴링 루프
- **API 소프트 캐시** - `/api/status`, `/api/stats` 1.5초 메모리 캐시로 중복 호출 억제
- **PDF 지연 로딩** - jsPDF/AutoTable을 내보내기 클릭 시점에 로드 (로컬 vendor 우선)

### 🧠 서버 검색 경로 최적화
- **SearchCache 구조 개선** - 튜플 대신 명시 엔트리 구조 사용
- **캐시 키 정합성 강화** - `sort_by`를 캐시 키에 포함
- **TTL 정책 정리** - 기본 TTL + 적응형 연장(상한 2배)
- **하이라이트 최적화** - 쿼리 단일 정규식 패턴 캐시 적용

### 📄 파일 미리보기 최적화
- **Preview LRU 캐시** - `path+mtime+length` 기반 캐시
- **TXT fast-path** - 미리보기 요청 시 선두 텍스트 우선 읽기
- **추출기 재사용** - 요청마다 `DocumentExtractor` 재생성 제거

## ✨ v2.8 구현 반영 (2026-02-25)

### 🔐 보안/운영 안전장치
- **CORS allowlist 강제** - `CORS_ALLOWED_ORIGINS` 기반 허용 origin만 응답
- **세션 쿠키 정책 명시 적용** - `HttpOnly`, `SameSite`, `Secure`를 설정값으로 고정
- **운영 fail-fast** - `APP_ENV=production`에서 waitress 미설치 시 서버 즉시 종료

### 🔌 API 계약 표준화
- `/api/search/history`: `success` + `popular[{query,count}]` + `popular_legacy` 동시 제공
- `/api/search/suggest`: `success` envelope 추가
- `/api/status`, `/api/health`: `success` envelope 추가(기존 필드 유지)
- `/api/models`: `reindex` 기본값 `true`, 실제 재인덱스 트리거 여부 응답 포함

### 📁 파일/업로드 정책 강화
- 파일 삭제 기본 정책을 `index_only`로 전환 (`delete_source=true` 옵션 시에만 물리 삭제)
- 삭제 응답에 `deletion_policy`, `deleted_source`, `deleted_from_index` 포함
- ZIP 업로드 제한 도입:
  - `max_entries`
  - `max_uncompressed_bytes`
  - `max_single_file_bytes`

### 🌐 프론트/PWA 정합성
- 자동완성 히스토리가 표준/레거시 포맷 모두 처리
- 검색 결과 하이라이트 상태(`lastRenderedQuery`) 동기화
- 관리자 테마 아이콘 로직을 `ThemeManager.getTheme()` 기반으로 수정
- Toast 렌더링을 `textContent` 기반으로 변경(XSS 방어)
- Service Worker는 GET allowlist API만 캐시하고 POST/인증/관리 API 캐시 제외

### 📦 이전 버전
| 버전 | 주요 변경 |
|------|----------|
| v2.5 | 벡터 검색 최적화, DB 인덱스, API 경량화 |
| v2.4 | 검색 캐시 히트율 향상, DB PRAGMA 최적화 |
| v2.3 | BM25 최적화, Lazy Import, 예외 처리 강화 |
| v2.2 | API 호환성, LangChain 버전 대응 |
| v2.1 | HWP/Excel 지원, OCR 기능 |

---

## 📋 주요 기능

| 기능 | 설명 |
|------|------|
| 📂 **다양한 포맷** | TXT, DOCX, PDF, XLSX, HWP |
| 🔍 **하이브리드 검색** | Vector + BM25 결합 검색 |
| 🏷️ **문서 태깅** | 자동 분류 및 파일별 태그 필터링 |
| 🔄 **버전 관리** | 개정 이력 추적, Diff 비교 |
| 📌 **북마크** | 검색 결과 즐겨찾기 |
| 🌙 **다크/라이트** | 테마 전환 지원 |
| 🔌 **오프라인 모드** | 폐쇄망 지원 (로컬 모델) |
| 📊 **통계 대시보드** | 실시간 시스템 상태 모니터링 |

---

## 🚀 빠른 시작

### 시스템 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| **Python** | 3.10+ | 3.11+ |
| **RAM** | 4GB | 8GB (AI 모델 사용 시) |
| **디스크** | 1GB | 3GB (오프라인 모델 포함) |
| **OS** | Windows 10, Linux, macOS | Windows 11 |

### 설치

```bash
# 1. 저장소 클론
git clone <repository-url>
cd Internal-Regulations-Finder-Online-main

# 2. 가상환경 생성 (권장)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# 3. 의존성 설치
# 전체 기능 (AI 포함, ~2GB)
pip install -r requirements.txt

# 경량 버전 (BM25만, ~200MB)
pip install -r requirements_lite.txt
```

### 실행

```bash
# 콘솔 서버
python run.py

# GUI 서버 (시스템 트레이, 권장)
python server_gui.py
```

**브라우저 접속**: `http://localhost:8080`  
**관리자 페이지**: `http://localhost:8080/admin`

---

## 📦 빌드 (PyInstaller)

### 빌드 옵션 비교

| Spec 파일 | 모드 | AI 기능 | 예상 크기 | 용도 |
|-----------|------|---------|-----------|------|
| `regulation_search_gui.spec` | GUI | ✅ 포함 | 500-800MB | 고성능 검색 필요 시 |
| `regulation_search_ultra_lite_gui.spec` | GUI | ❌ BM25만 | 60-100MB | 경량 배포, 빠른 시작 |
| `regulation_search_onefile.spec` | GUI (단일 파일) | ❌ BM25만 | 40-60MB | USB 휴대용 |

### 빌드 명령

```bash
# AI 포함 GUI 버전 (고성능)
pyinstaller regulation_search_gui.spec --clean

# 초경량 GUI 버전 (BM25만)
pyinstaller regulation_search_ultra_lite_gui.spec --clean

# 단일 실행 파일 (휴대용)
pyinstaller regulation_search_onefile.spec --clean
```

### 빌드 최적화 팁

1. **UPX 설치**: 추가 30-50% 크기 절감
   ```bash
   # Windows: https://github.com/upx/upx/releases
   # spec 파일에서 upx=True 설정
   ```

2. **CUDA 제외**: GPU 미사용 시 CUDA 라이브러리 제외 (이미 설정됨)

3. **불필요한 언어 제외**: PyQt6 locale 파일 제거

---

## 🏗️ 프로젝트 구조

```
📁 Internal-Regulations-Finder-Online-main/
├── 📄 run.py                  # 콘솔 엔트리포인트
├── 📄 server_gui.py           # GUI 서버 (PyQt6, 시스템 트레이)
├── 📄 download_models.py      # 오프라인용 모델 다운로드
├── 📁 app/                    # Flask 앱 패키지
│   ├── __init__.py            # 앱 팩토리 (create_app)
│   ├── config.py              # AppConfig 설정 클래스
│   ├── constants.py           # 상수, 에러 메시지, 정규식 패턴
│   ├── exceptions.py          # 커스텀 예외 클래스 계층
│   ├── utils.py               # 유틸리티 (TaskResult, MemoryMonitor)
│   ├── 📁 routes/             # API 라우트
│   │   ├── api_search.py      # 검색 API (/api/search)
│   │   ├── api_files.py       # 파일/태그/개정 관리 API
│   │   ├── api_system.py      # 시스템/동기화 API
│   │   └── main_routes.py     # 메인 페이지 라우트
│   └── 📁 services/           # 비즈니스 로직 서비스
│       ├── search.py          # 하이브리드 검색 엔진 (RegulationQASystem)
│       ├── document.py        # 문서 추출/파싱/분할/비교
│       ├── db.py              # SQLite 싱글톤 DB 관리
│       ├── file_manager.py    # RevisionTracker, FolderWatcher
│       ├── metadata.py        # TagManager
│       └── embeddings_backends/  # 임베딩 백엔드 (v2.6)
│           ├── factory.py        # torch/onnx 선택
│           ├── torch_backend.py  # PyTorch 백엔드
│           └── onnx_backend.py   # ONNX Runtime 백엔드
├── 📁 static/                 # 프론트엔드 정적 파일
│   ├── app.js                 # SPA 스타일 클라이언트
│   ├── style.css              # CSS 스타일 (다크/라이트 테마)
│   └── sw.js                  # PWA 서비스 워커
├── 📁 templates/              # HTML 템플릿
│   ├── index.html             # 메인 검색 페이지
│   └── admin.html             # 관리자 페이지
├── 📁 config/                 # 런타임 설정
│   ├── settings.json          # 사용자 설정
│   └── regulations.db         # SQLite 데이터베이스
├── 📁 uploads/                # 업로드된 문서
├── 📁 revisions/              # 개정 이력 파일
└── 📁 models/                 # AI 모델 캐시
```

---

## 🔌 API 엔드포인트

### 검색 API
| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/search` | 하이브리드 검색 `{query, k, hybrid}` |
| GET | `/api/search/history` | 검색 히스토리 (`success`, `recent`, `popular`, `popular_legacy`) |
| GET | `/api/search/suggest` | 자동완성 `?q=검색어` (`success`, `suggestions`) |
| POST | `/api/cache/clear` | 캐시 초기화 |

### 파일 API
| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/files` | 파일 목록 |
| POST | `/api/upload` | 파일 업로드 + 인덱싱 |
| POST | `/api/upload/folder` | ZIP 업로드 + 자동 인덱싱 (제한 파라미터 지원) |
| DELETE | `/api/files/<filename>` | 파일 삭제 (기본 `index_only`, `delete_source=true` 옵션) |
| DELETE | `/api/files/all` | 전체 삭제 (기본 `index_only`) |
| GET/POST/DELETE | `/api/tags` | 태그 관리 |
| GET/POST | `/api/revisions` | 개정 이력 |

### 시스템 API
| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/status` | 서버 상태 (`success` + `progress/load_progress`) |
| GET | `/api/health` | 헬스 체크 (`success` envelope) |
| GET | `/api/models` | 모델 목록 |
| POST | `/api/models` | 모델 변경 (`reindex` 기본 `true`) |
| POST | `/api/sync/start` | 동기화 시작 |
| POST | `/api/sync/stop` | 동기화 중지 요청 |

---

## ⌨️ 단축키

| 키 | 기능 |
|----|------|
| `Ctrl+K` | 검색창 포커스 |
| `J` / `K` | 결과 위/아래 탐색 |
| `N` / `P` | 하이라이트 탐색 |
| `T` | 테마 전환 |
| `R` | 읽기 모드 |
| `Esc` | 모달 닫기 |

---

## ⚙️ 설정

### app/config.py

```python
class AppConfig:
    # 서버
    SERVER_PORT = 8080
    SERVER_THREADS = 32
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    APP_ENV = "development"  # production 시 waitress 필수
    CORS_ALLOWED_ORIGINS = ["http://localhost:8080", "http://127.0.0.1:8080"]
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = False  # production+HTTPS에서는 True 권장
    
    # AI 모델
    DEFAULT_MODEL = "SNU SBERT (고성능)"
    AVAILABLE_MODELS = {
        "SNU SBERT (고성능)": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        "BM-K Simal (균형)": "BM-K/ko-simal-roberta-base",
        "JHGan SBERT (빠름)": "jhgan/ko-sbert-nli"
    }
    
    # 오프라인 모드
    OFFLINE_MODE = False
    LOCAL_MODEL_PATH = "./models/snunlp--KR-SBERT-V40K-klueNLI-augSTS"
    
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

    # ZIP 업로드 제한
    ZIP_MAX_ENTRIES = 1000
    ZIP_MAX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024
    ZIP_MAX_SINGLE_FILE_BYTES = 50 * 1024 * 1024
```

### 오프라인 모드 설정

```bash
# 모델 다운로드
python download_models.py

# config/settings.json
{
    "offline_mode": true,
    "local_model_path": "./models/snunlp--KR-SBERT-V40K-klueNLI-augSTS"
}
```

---

## 📊 성능 벤치마크

| 항목 | v2.5 | v2.6.2 |
|------|------|--------|
| 검색 응답 시간 | ~120ms | ~80ms |
| 캐시 히트율 | ~85% | ~90% |
| 메모리 사용량 | ~0.65GB | ~0.60GB |
| 응답 압축 | 기본 | Gzip 활성화 |

### 성능 특성

| 지표 | 값 |
|------|-----|
| 동시 검색 제한 | 10개 (SearchQueue) |
| 분당 요청 제한 | 300회/IP (RateLimiter) |
| 캐시 TTL | 600초 (적응형 최대 2배) |
| 캐시 키 | `query + k + hybrid + sort_by` |
| 최대 업로드 | 50MB |

---

## 🧪 성능 측정

```bash
# 테스트 실행 (표준)
pytest -q

# 검색 API 스모크 벤치마크
python scripts/perf_smoke.py
python scripts/perf_smoke.py --base-url http://127.0.0.1:8080 --query "휴가 규정"
```

`scripts/perf_smoke.py`는 워밍업 30회, 측정 200회, 동시성 1/5/10 기준으로 p50/p95/p99, 평균 응답 크기, 에러율을 출력합니다.

---

## 🔒 보안

- **XSS 방지**: 모든 사용자 입력에 `escapeHtml()` 적용
- **Path Traversal 방지**: 경로 정규화 및 `..` 패턴 차단
- **SQL Injection 방지**: 파라미터 바인딩 사용
- **Rate Limiting**: IP 기반 요청 제한
- **CORS Allowlist**: 허용된 Origin만 credentials 응답 허용
- **세션 쿠키 강화**: `HttpOnly`/`SameSite`/`Secure` 정책 적용

---

## 🐛 트러블슈팅

| 문제 | 해결 |
|------|------|
| 모델 로드 실패 | `OFFLINE_MODE=True` + `LOCAL_MODEL_PATH` 설정 |
| 검색 느림 | 캐시 크기 증가 (`SEARCH_CACHE_SIZE`) |
| 메모리 부족 | 경량 버전 사용 (`requirements_lite.txt`) |
| HWP 추출 실패 | `pip install olefile` |
| OCR 필요 | Tesseract 설치 + `pip install pytesseract pdf2image` |
| 빌드 오류 | PyInstaller 최신 버전 사용, `--clean` 옵션 |
| 운영 실행 시 즉시 종료 | `APP_ENV=production`이면 `waitress` 설치 필요 (`pip install waitress`) |

---

## 📝 라이선스

© 2026 사내 규정 검색기

---

## 🔗 관련 링크

- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- [HuggingFace 한국어 SBERT](https://huggingface.co/snunlp/KR-SBERT-V40K-klueNLI-augSTS)
- [PyInstaller 문서](https://pyinstaller.org/)
- [Flask 문서](https://flask.palletsprojects.com/)

---

## ✨ v2.7 구현 반영 (2026-02-20)

### 🔐 보안 강화
- 기본 관리자 비밀번호 fallback(`"admin"`) 제거: 미설정 시 fail-closed
- 상태 변경 API 관리자 권한 통일:
  - `POST /api/upload`
  - `POST /api/upload/folder`
  - `POST /api/revisions`
  - `POST /api/tags/set`
  - `POST /api/tags/auto`
  - `POST /api/cache/clear`
- 검색 결과 렌더링 XSS 방어: 클라이언트 안전 하이라이트(escape 기반)로 전환
- 리비전 저장 경로 검증 강화: revisions 루트 이탈 차단

### 🧭 파일 식별 체계 개선
- `file_id`(정규화 절대경로 기반 해시) 도입
- 파일 관련 API/검색 결과에 `file_id` 병행 제공
- 동명이인 파일 충돌 완화(미리보기/태그/리비전/필터)

### 🔌 API 정합화(하위호환 alias 포함)
- `/api/status`: `load_progress` 유지 + `progress` alias 추가
- `/api/sync/status`: 기존 필드 유지 + `status.{running,current_folder,progress,error}` 추가
- `/api/revisions` GET: `history` 유지 + `revisions` alias 추가
- `/api/tags/auto`: `suggested_tags` 유지 + `tags` alias 추가

### 📁 파일/업로드 기능
- ZIP 폴더 업로드 API 추가: `POST /api/upload/folder`
- 다운로드 복구:
  - `GET /api/files/by-id/<file_id>/download`
  - `GET /api/files/<path:filename>/download` (호환)
- by-id 파일 API 추가:
  - `GET /api/files/by-id/<file_id>/preview`
  - `DELETE /api/files/by-id/<file_id>`
  - `GET /api/files/by-id/<file_id>/versions`
  - `GET /api/files/by-id/<file_id>/versions/<version>`
  - `GET /api/files/by-id/<file_id>/versions/compare`

### 🧠 Lite(BM25-only) 안정화
- 모델 미로드 환경 캐시 경로 안정화
- splitter/캐시 로직의 BM25-only fallback 보강

### 📱 PWA/UX 정합화
- 프론트 업로드 확장자 필터를 백엔드와 일치(`txt/docx/pdf/xlsx/xls/hwp`)
- PWA 아이콘 경로 정합화:
  - `static/icons/icon-192.png`
  - `static/icons/icon-512.png`

### 🧪 테스트
- 신규/보강 테스트 추가 후 기준 통과:
  - `pytest -q`
