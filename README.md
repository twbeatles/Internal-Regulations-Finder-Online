# 📚 사내 규정 검색기 v2.4

> AI 기반 하이브리드 검색 시스템 (웹 서버 버전)

## ✨ v2.4 업데이트 (2026-01-08)

### ⚡ 성능 최적화
- **검색 캐시 히트율 향상** - 쿼리 정규화로 20-30% 캐시 히트 증가
- **DB 성능 개선** - PRAGMA cache_size/mmap_size, 복합 인덱스 추가
- **메모리 모니터링** - 대용량 문서 처리 시 자동 경고

### 🛠️ 빌드 개선
- **GUI 모드 지원** - 콘솔 창 없이 백그라운드 실행
- **torch DLL 자동 수집** - shm.dll 의존성 문제 해결
- **경량화 옵션** - Ultra Lite (60MB) / Full (400MB) 선택

### 📦 변경 이력
- v2.3: BM25 최적화, Lazy Import, 예외 처리 강화
- v2.2: API 호환성, LangChain 버전 대응
- v2.1: HWP/Excel 지원, OCR 기능

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
# 콘솔 서버
python run.py

# GUI 서버 (시스템 트레이)
python server_gui.py
```

브라우저: `http://localhost:8080`

---

## 📦 빌드

### 빌드 옵션 비교

| Spec 파일 | 모드 | AI 기능 | 예상 크기 |
|-----------|------|---------|-----------|
| `regulation_search_gui.spec` | GUI | ✅ 포함 | 400-600MB |
| `regulation_search_ultra_lite_gui.spec` | GUI | ❌ BM25만 | 60-100MB |
| `regulation_search.spec` | 콘솔 | ✅ 포함 | 400-600MB |

### GUI 빌드 (권장)
```bash
# AI 포함 버전
pyinstaller regulation_search_gui.spec --clean

# 초경량 버전 (BM25만)
pyinstaller regulation_search_ultra_lite_gui.spec --clean
```

> 💡 UPX 설치 시 추가 30-50% 크기 절감

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
│   ├── routes/             # API 라우트
│   └── services/           # 비즈니스 로직
├── static/                 # 프론트엔드
└── templates/              # HTML 템플릿
```

---

## ⌨️ 단축키

| 키 | 기능 |
|----|------|
| `Ctrl+K` | 검색창 포커스 |
| `J`/`K` | 결과 탐색 |
| `T` | 테마 전환 |
| `Esc` | 모달 닫기 |

---

## 🔌 API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/status` | 시스템 상태 |
| POST | `/api/search` | 문서 검색 |
| GET | `/api/files` | 파일 목록 |
| POST | `/api/sync/start` | 동기화 시작 |

---

## ⚙️ 설정 (app/config.py)

```python
OFFLINE_MODE = True         # 폐쇄망 모드
LOCAL_MODEL_PATH = "./models/ko-sbert"
CHUNK_SIZE = 800
VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3
```

---

## 📊 성능 벤치마크

| 항목 | v2.3 | v2.4 |
|------|------|------|
| 검색 응답 시간 | ~180ms | ~150ms |
| 캐시 히트율 | 60% | 80% |
| 메모리 사용량 | 0.8GB | 0.7GB |

---

## ⚙️ 요구사항

- **Python** 3.10+ (3.14 호환)
- **OS** Windows 10/11, Linux, macOS
- **RAM** 4GB+ (AI 모델 사용 시 8GB 권장)

---

## 📝 라이선스

© 2026 사내 규정 검색기

---

## 🔗 관련 링크

- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- [HuggingFace 모델](https://huggingface.co/snunlp/KR-SBERT-V40K-klueNLI-augSTS)
