# 📚 사내 규정 검색기 v2.5

> AI 기반 하이브리드 검색 시스템 (Vector + BM25)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ v2.5 업데이트 (2026-01-15)

### ⚡ 성능 최적화
- **벡터 검색 최적화** - 비하이브리드 검색 시 50% 연산 감소
- **DB 인덱스 추가** - 검색 히스토리 중복 체크 속도 향상
- **API 응답 경량화** - 불필요한 중복 필드 제거
- **태그 설정 트랜잭션** - 원자적 처리로 데이터 무결성 보장

### 🛠️ 안정성 개선
- **FolderWatcher 타임아웃** - 5초 제한으로 무한 블로킹 방지
- **모델 변경 안내** - 재인덱싱 필요성 표시
- **검색 히스토리 중복 방지** - 5분 내 동일 쿼리 필터링

### 📦 이전 버전
| 버전 | 주요 변경 |
|------|----------|
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
│       └── metadata.py        # TagManager
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
| GET | `/api/search/history` | 검색 히스토리 |
| GET | `/api/search/suggest` | 자동완성 `?q=검색어` |
| POST | `/api/cache/clear` | 캐시 초기화 |

### 파일 API
| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/files` | 파일 목록 |
| POST | `/api/upload` | 파일 업로드 + 인덱싱 |
| DELETE | `/api/files/<filename>` | 파일 삭제 |
| GET/POST/DELETE | `/api/tags` | 태그 관리 |
| GET/POST | `/api/revisions` | 개정 이력 |

### 시스템 API
| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/status` | 서버 상태 |
| GET | `/api/health` | 헬스 체크 |
| GET | `/api/models` | 모델 목록 |
| POST | `/api/models` | 모델 변경 |
| POST | `/api/sync/start` | 동기화 시작 |

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

| 항목 | v2.4 | v2.5 |
|------|------|------|
| 검색 응답 시간 | ~150ms | ~120ms |
| 캐시 히트율 | 80% | 85% |
| 메모리 사용량 | 0.7GB | 0.65GB |
| API 응답 크기 | 기준 | -20% |

### 성능 특성

| 지표 | 값 |
|------|-----|
| 동시 검색 제한 | 10개 (SearchQueue) |
| 분당 요청 제한 | 300회/IP (RateLimiter) |
| 캐시 TTL | 300초 |
| 최대 업로드 | 50MB |

---

## 🔒 보안

- **XSS 방지**: 모든 사용자 입력에 `escapeHtml()` 적용
- **Path Traversal 방지**: 경로 정규화 및 `..` 패턴 차단
- **SQL Injection 방지**: 파라미터 바인딩 사용
- **Rate Limiting**: IP 기반 요청 제한

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

---

## 📝 라이선스

© 2026 사내 규정 검색기

---

## 🔗 관련 링크

- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- [HuggingFace 한국어 SBERT](https://huggingface.co/snunlp/KR-SBERT-V40K-klueNLI-augSTS)
- [PyInstaller 문서](https://pyinstaller.org/)
- [Flask 문서](https://flask.palletsprojects.com/)
