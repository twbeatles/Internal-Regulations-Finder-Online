# 사내 규정 검색기 - 웹 서버 버전 v2.0

AI 기반 하이브리드 검색 시스템을 활용한 사내 규정 검색 웹 서버입니다.

## ✨ 주요 기능

### 🔍 검색 기능
- **하이브리드 검색**: 벡터(FAISS) + 키워드(BM25) 검색 조합
- **검색어 하이라이팅**: 검색 결과에서 키워드 강조 표시
- **검색 히스토리**: 최근 검색어 및 인기 검색어
- **자동완성**: 입력시 실시간 검색어 추천

### 📤 결과 내보내기 (v2.0 신규)
- **텍스트** (.txt): 간단한 텍스트 형식
- **마크다운** (.md): 문서화에 적합한 형식
- **JSON** (.json): 데이터 처리용 형식

### 🎨 프리미엄 UI/UX (v2.0 신규)
- **글래스모피즘**: 반투명 카드 효과
- **리플 효과**: 버튼 클릭 시 Material Design 스타일
- **스켈레톤 로딩**: 콘텐츠 로드 중 애니메이션
- **다크/라이트 테마**: 사용자 선호도에 맞는 테마 전환

### ♿ 접근성 (v2.0 신규)
- **스킵 네비게이션**: 키보드 사용자를 위한 건너뛰기 링크
- **ARIA 레이블**: 스크린리더 호환성
- **키보드 단축키**: Ctrl+K (검색 포커스), / (검색)

### 🖥️ 서버 기능
- **다중 사용자 지원**: Waitress 프로덕션 서버 (8 스레드)
- **헬스체크 API**: 서버 상태 모니터링
- **응답 시간 측정**: 검색 성능 추적
- **시스템 트레이**: GUI로 서버 관리

### 📁 지원 문서
- `.txt` (텍스트)
- `.docx` (Word 문서)
- `.pdf` (PDF 문서)

---

## 🚀 시작하기

### 필수 요구사항

- Python 3.9+
- Windows 10/11
- 4GB 이상 RAM 권장

### 설치

```bash
pip install -r requirements.txt
```

### 실행 방법

#### 1. GUI 모드 (권장)

```bash
python server_gui.py
```

- 시스템 트레이에서 서버 관리
- Windows 시작 시 자동 실행 설정 가능

#### 2. 콘솔 모드

```bash
python server.py
```

#### 3. 최소화 시작

```bash
python server_gui.py --minimized
```

### 접속

| 페이지 | URL |
|--------|-----|
| 검색 (사용자) | http://localhost:8080 |
| 관리자 | http://localhost:8080/admin |
| 헬스체크 | http://localhost:8080/api/health |

---

## 📁 프로젝트 구조

```
├── server.py           # Flask 서버 (핵심)
├── server_gui.py       # PyQt6 GUI (시스템 트레이)
├── templates/
│   ├── index.html      # 검색 페이지 (ARIA 지원)
│   └── admin.html      # 관리자 페이지
├── static/
│   ├── style.css       # 글래스모피즘 + 다크/라이트 테마
│   └── app.js          # UX 유틸리티 + 내보내기 기능
├── uploads/            # 업로드된 규정 파일
├── models/             # AI 모델 캐시
└── logs/               # 서버 로그
```

---

## 📡 API

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/api/status` | 서버 상태 |
| GET | `/api/health` | 헬스체크 (v2.0) |
| POST | `/api/search` | 검색 (응답시간 포함) |
| POST | `/api/upload` | 파일 업로드 |
| GET | `/api/files` | 파일 목록 |
| DELETE | `/api/files/{name}` | 파일 삭제 |
| GET | `/api/files/{name}/preview` | 파일 미리보기 |
| DELETE | `/api/cache` | 캐시 삭제 |

### 검색 API 예시

```bash
curl -X POST http://localhost:8080/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "휴가 규정", "k": 3, "hybrid": true}'
```

**응답:**
```json
{
  "success": true,
  "results": [...],
  "query": "휴가 규정",
  "response_time_ms": 125.5,
  "result_count": 3
}
```

### 헬스체크 API

```bash
curl http://localhost:8080/api/health
```

**응답:**
```json
{
  "status": "healthy",
  "ready": true,
  "model_loaded": true,
  "documents_loaded": true,
  "document_count": 150,
  "file_count": 5,
  "cpu_percent": 25.5,
  "memory_percent": 45.2,
  "version": "1.0 (웹 서버)"
}
```

---

## 🤖 AI 모델

| 모델 | 특징 |
|------|------|
| JHGan SBERT (기본) | 빠른 속도 |
| SNU SBERT | 높은 정확도 |
| BM-K Simal | 균형 잡힌 성능 |

---

## ⚙️ 설정

`server.py`의 `AppConfig` 클래스에서 설정 변경:

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `SERVER_PORT` | 8080 | 서버 포트 |
| `MAX_CONTENT_LENGTH` | 50MB | 최대 업로드 크기 |
| `CHUNK_SIZE` | 800 | 문서 청킹 크기 |
| `SEARCH_CACHE_SIZE` | 100 | 검색 캐시 크기 |

---

## 🔨 빌드 (PyInstaller)

```bash
# GUI 버전 빌드 (권장)
pyinstaller server_gui.spec

# 콘솔 버전 빌드
pyinstaller server.spec
```

빌드된 실행 파일: `dist/사내규정검색기서버/`

---

## 📝 변경 이력

### v2.0 (2025-12)
- 🎨 글래스모피즘 UI 디자인
- 📤 검색 결과 내보내기 (txt, md, json)
- ♿ ARIA 접근성 개선
- 🔔 네트워크 상태 감지
- ⏱️ 스켈레톤 로딩 UX
- 🏥 헬스체크 API (/api/health)
- ⚡ 검색 응답 시간 측정

### v1.0 (2025-12)
- 초기 웹 서버 버전
- 하이브리드 검색 (FAISS + BM25)
- PyQt6 시스템 트레이 GUI

---

## 📄 라이선스

