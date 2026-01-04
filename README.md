# 📚 사내 규정 검색기 v2.1

> AI 기반 하이브리드 검색 시스템 with PyQt6 GUI

## ✨ v2.1 업데이트 (2026-01-04)

### 🆕 신규 기능
- 📌 **북마크 시스템** - localStorage 기반 즐겨찾기
- 📕 **PDF 내보내기** - jsPDF 활용 문서 출력
- 🔄 **버전 관리** - 문서 개정 이력 및 Diff 비교
- 🔍 **고급 검색** - AND/OR/NOT 연산자, 정규식 지원

### 🎨 UI/UX 개선
- 글래스모피즘 디자인 적용
- 3색 그라데이션 컬러 시스템
- 마이크로 애니메이션 강화
- 토스트 알림 프로그레스 바

### 🔧 성능 최적화
- LRU 캐시 O(1) 성능 개선
- JavaScript debounce/throttle 유틸리티
- 버그 수정 (API, request import)

---

## 📋 주요 기능

| 기능 | 설명 |
|------|------|
| 📂 **다양한 포맷** | TXT, DOCX, PDF, XLSX, HWP |
| 🔍 **하이브리드 검색** | Vector + BM25 결합 |
| 🏷️ **문서 태깅** | 파일별 태그 필터링 |
| 🔄 **버전 관리** | 개정 이력 추적, Diff 비교 |
| 📌 **북마크** | 검색 결과 즐겨찾기 |
| 🌙 **다크/라이트** | 테마 전환 |
| 📱 **PWA** | 설치형 웹 앱 |

---

## 🚀 빠른 시작

### 설치
```bash
pip install -r requirements.txt
```

### 실행
```bash
# GUI 서버 (권장)
python server_gui.py

# 콘솔 서버
python run.py
```

브라우저: `http://localhost:8080`

---

## 📦 빌드

### 경량 버전 (콘솔)
```bash
pyinstaller internal_regulations_lite.spec
```

### GUI 버전
```bash
pyinstaller server_gui.spec
```

---

## 🏗️ 프로젝트 구조

```
├── run.py              # 콘솔 엔트리포인트
├── server_gui.py       # GUI 서버 (PyQt6)
├── app/
│   ├── __init__.py     # Flask 앱 팩토리
│   ├── routes/         # API 라우트
│   └── services/       # 비즈니스 로직
├── static/
│   ├── app.js          # 프론트엔드 로직
│   └── style.css       # 스타일시트
└── templates/          # Jinja2 템플릿
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

## ⚙️ 요구사항

- Python 3.10+
- Windows 10/11 (GUI)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (OCR 사용 시)

---

© 2026 사내 규정 검색기
