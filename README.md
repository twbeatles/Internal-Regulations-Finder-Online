# 📚 사내 규정 검색기 v2.3

> AI 기반 하이브리드 검색 시스템 (웹 서버 버전)

## ✨ v2.3 업데이트 (2026-01-07)

### ⚡ 성능 최적화
- **BM25 검색 최적화** - term frequency 사전 계산으로 검색 속도 40-50% 향상
- **정규식 사전 컴파일** - 토큰화 성능 20-30% 개선
- **Lazy Import 패턴** - GUI 시작 시간 단축 (PyTorch, LangChain 지연 로딩)
- **메모리 효율화** - `__slots__` 적용, GC 최적화

### 🛠️ 코드 품질 개선
- **예외 처리 강화** - 커스텀 예외 클래스 체계화 (`app/exceptions.py`)
- **상수 추출** - 매직 넘버/문자열을 상수화 (`app/constants.py`)
- **타입 힌트 추가** - 주요 함수에 타입 어노테이션
- **로깅 개선** - 구조화된 로깅, 로그 레벨 분리

### 🐛 버그 수정 (v2.2)
- 검색 API 파라미터 호환성 (`q`/`query` 모두 지원)
- LangChain 버전 호환성 문제 해결
- CSP 정책 및 API 라우트 누락 수정

---

## 📋 주요 기능

| 기능 | 설명 |
|------|------|
| 📂 **다양한 포맷** | TXT, DOCX, PDF, XLSX, HWP |
| 🔍 **하이브리드 검색** | Vector + BM25 결합 검색 |
| 🏷️ **문서 태깅** | 파일별 태그 필터링 |
| 🔄 **버전 관리** | 개정 이력 추적, Diff 비교 |
| 📌 **북마크** | 검색 결과 즐겨찾기 |
| 🌙 **다크/라이트** | 테마 전환 지원 |
| 📱 **PWA/반응형** | 모바일 최적화 |
| 🔌 **오프라인 모드** | 폐쇄망 지원 |

---

## 🚀 빠른 시작

### 설치
```bash
# 전체 기능 (AI 포함)
pip install -r requirements.txt

# 경량 버전 (BM25만)
pip install -r requirements_lite.txt
```

### 실행
```bash
# 콘솔 서버 (권장)
python run.py

# GUI 서버 (시스템 트레이)
python server_gui.py
```

브라우저: `http://localhost:8080`

---

## 📦 빌드

### 경량 버전 (AI 포함, ~200MB)
```bash
pyinstaller regulation_search.spec
```

### 초경량 버전 (BM25만, ~80MB)
```bash
pyinstaller regulation_search_ultra_lite.spec
```

> 💡 UPX 압축 시 추가 30-50% 크기 절감

---

## 🏗️ 프로젝트 구조

```
├── run.py                  # 콘솔 엔트리포인트
├── server_gui.py           # GUI 서버 (PyQt6)
├── app/
│   ├── __init__.py         # Flask 앱 팩토리
│   ├── config.py           # 설정
│   ├── constants.py        # 상수 정의
│   ├── exceptions.py       # 커스텀 예외
│   ├── routes/
│   │   ├── api_search.py   # 검색 API
│   │   ├── api_files.py    # 파일 API
│   │   └── api_system.py   # 시스템 API
│   └── services/
│       ├── search.py       # 검색 엔진 (BM25, Vector)
│       ├── document.py     # 문서 처리
│       └── db.py           # SQLite DB
├── static/                 # 프론트엔드 (JS, CSS)
└── templates/              # HTML 템플릿
```

---

## ⌨️ 단축키

| 키 | 기능 |
|----|------|
| `Ctrl+K` | 검색창 포커스 |
| `?` | 단축키 도움말 |
| `J`/`K` | 결과 탐색 |
| `T` | 테마 전환 |
| `Esc` | 모달 닫기 |

---

## 🔌 API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/status` | 시스템 상태 |
| POST | `/api/search` | 문서 검색 |
| GET | `/api/search/history` | 검색 기록 |
| GET | `/api/files` | 파일 목록 |
| POST | `/api/sync/start` | 동기화 시작 |
| POST | `/api/process` | 재처리 |

---

## ⚙️ 설정

### 환경 설정 (app/config.py)
```python
# 오프라인 모드 (폐쇄망)
OFFLINE_MODE = True
LOCAL_MODEL_PATH = "./models/ko-sbert"

# 검색 설정
CHUNK_SIZE = 800
VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3
```

---

## 📊 성능 벤치마크

| 항목 | v2.2 | v2.3 |
|------|------|------|
| 검색 응답 시간 | ~280ms | ~180ms |
| GUI 시작 시간 | ~5s | ~2s |
| 메모리 사용량 | 1.0GB | 0.8GB |
| 빌드 크기 (초경량) | 120MB | 80MB |

---

## ⚙️ 요구사항

- **Python** 3.10+
- **OS** Windows 10/11, Linux, macOS
- **RAM** 4GB+ (AI 모델 사용 시 8GB 권장)

### 선택적 의존성

| 기능 | 패키지 |
|------|--------|
| HWP 지원 | `olefile` |
| OCR 지원 | `pytesseract`, `pdf2image` |
| Excel 지원 | `openpyxl` |

---

## 📝 라이선스

© 2026 사내 규정 검색기

---

## 🔗 관련 링크

- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- [HuggingFace 모델](https://huggingface.co/snunlp/KR-SBERT-V40K-klueNLI-augSTS)
