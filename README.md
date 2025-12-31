# 📚 사내 규정 검색기 v2.0

> AI 기반 하이브리드 검색 시스템 with PyQt6 GUI 서버 관리

## ✨ v1.4 업데이트 (2025-12-31)

- 🖥️ **GUI 개선**: 모든 버튼에 한글 툴팁 추가
- 🎨 **웹 UI/UX 개선**: 검색 팁 섹션, 선택된 결과 카드 애니메이션 추가
- 🐛 **버그 수정**: JavaScript 중복 메서드 오류 해결

## 📋 주요 기능

| 기능 | 설명 |
|------|------|
| 📂 **다양한 포맷** | TXT, DOCX, PDF, XLSX, HWP 파일 지원 |
| 🔍 **하이브리드 검색** | Vector Search (AI) + BM25 (키워드) 결합 |
| 🏷️ **문서 태깅** | 파일별 태그 설정 및 필터링 |
| 🔄 **폴더 동기화** | 자동 인덱싱 (Watchdog) |
| 🌙 **다크 모드** | 프리미엄 다크 테마 |
| 📱 **PWA 지원** | 설치형 웹 앱 |

---

## 🚀 빠른 시작

### 설치
```bash
pip install -r requirements.txt
```

### 실행 방법

**1. GUI 서버 (권장)**
```bash
python server_gui.py
```
- 시스템 트레이 지원
- Windows 시작 프로그램 등록 가능
- 포트 설정, 비밀번호 보호 등 관리 기능

**2. 콘솔 서버**
```bash
python run.py
```

서버 시작 후 브라우저에서 `http://localhost:8080` 접속

---

## 📦 빌드 (배포용)

### GUI 버전 빌드
```bash
pyinstaller server_gui.spec
```

### 콘솔 버전 빌드
```bash
pyinstaller internal_regulations.spec
```

생성된 실행 파일 위치:
- `dist/사내규정검색기/` (GUI)
- `dist/InternalRegulationsFinder/` (콘솔)

---

## 🏗️ 프로젝트 구조

```
Root/
├── run.py                 # 콘솔 실행 엔트리포인트
├── server_gui.py          # GUI 서버 (PyQt6)
├── server.py              # Flask 서버 코어
├── app/                   # 메인 애플리케이션 패키지
│   ├── routes/            # API 라우트
│   └── services/          # 비즈니스 로직
├── static/                # 프론트엔드 리소스
├── templates/             # HTML 템플릿
└── config/                # 사용자 설정 (자동 생성)
```

---

## ⌨️ 단축키 (웹 UI)

| 단축키 | 기능 |
|--------|------|
| `Ctrl + K` | 검색창 포커스 |
| `?` | 단축키 도움말 |
| `J` / `K` | 결과 탐색 (다음/이전) |
| `N` / `P` | 하이라이트 탐색 |
| `T` | 테마 전환 |
| `Esc` | 모달 닫기 |

---

## ⚠️ 요구사항

- Python 3.10+
- Windows 10/11 (GUI 버전)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (OCR 사용 시)

---

© 2025 사내 규정 검색기
