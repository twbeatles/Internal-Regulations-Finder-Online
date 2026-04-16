# 📦 빌드 가이드

## 빌드 옵션

| Spec 파일 | 모드 | AI 기능 | 예상 크기 | 특징 |
|-----------|------|---------|-----------|------|
| `regulation_search_gui.spec` | GUI | ✅ | 500-800MB | AI 벡터 검색 + BM25 |
| `regulation_search_ultra_lite_gui.spec` | GUI | ❌ | 60-100MB | BM25만 (torch 제외) |

> 💡 **v2.8.2 반영**: `.hwpx` ZIP+XML 추출 경로와 `.hwp`/`.hwpx` 진단 정보가 앱 기반 빌드 흐름에도 연결됩니다.

---

## 사전 요구사항

### 필수
```bash
pip install -r requirements.txt
# 또는 직접 설치
pip install pyinstaller flask-compress
```

### 오프라인/폐쇄망 배포 준비
빌드 전, 정적 자원 및 모델을 미리 다운로드하여 포함할 수 있습니다.

```bash
# 1. 정적 자원(JS, Fonts) 다운로드 -> static/vendor 폴더 생성
python download_static.py

# 2. AI 모델 다운로드 -> models 폴더 생성 (AI 버전 빌드 시)
python download_models.py
```

문서 추출 관련 참고:
- `.hwpx`는 추가 패키지 없이 앱 내 ZIP/XML 파서로 처리됩니다.
- `.hwp`는 `olefile`이 필요하며 `requirements.txt`, `requirements_lite.txt`에 포함되어 있습니다.
- 인덱싱은 추출된 `text`만 사용하고, `metadata`/`tables`/`diagnostics`는 부가정보로 유지됩니다.

---

## 빌드 명령어

```powershell
cd "d:\twbeatles-repos\Internal-Regulations-Finder-Online"

# AI 포함 버전 (벡터 검색 + BM25)
# [권장] 대부분의 환경에서 사용
pyinstaller regulation_search_gui.spec --clean

# Lite 버전 (BM25만, AI 제외)
# [저사양] AI 기능이 필요 없는 경우
pyinstaller regulation_search_ultra_lite_gui.spec --clean

# 레거시 콘솔 빌드가 필요하면
pyinstaller internal_regulations.spec --clean
pyinstaller internal_regulations_lite.spec --clean
```

---

## 빌드 출력

```
dist/
├── 사내규정검색기/           # AI 버전
│   ├── 사내규정검색기.exe
│   └── _internal/
│
└── 사내규정검색기_Lite/      # Lite 버전
    ├── 사내규정검색기_Lite.exe
    └── _internal/
```

---

## 문제 해결

### torch 모듈 오류
```
No module named 'torch'
```
**정상 동작**: Lite 버전에서는 torch 없이 BM25 검색만 사용

### 빌드 크기가 너무 큼
1. Lite 버전 사용 권장
2. `.spec` 파일의 `excluded_binaries` 확인

### 실행 시 콘솔 창 표시
**해결**: `console=False` 확인 (두 spec 모두 GUI 모드)

### HWP/HWPX 패키징 확인
1. `.hwpx` 업로드 후 미리보기에 본문과 `diagnostics.engine_used=hwpx-xml`이 보이는지 확인
2. `.hwp` 업로드 후 `olefile` 누락 환경에서는 파일 단위 오류만 발생하고 서버는 계속 동작하는지 확인

---

## CPU 전용 빌드 (CUDA 제외)

모든 `.spec` 파일은 기본적으로 CUDA를 제외합니다:
- `cuda`, `cudnn`, `cublas` 등 필터링됨
- GPU 필요 시 `excluded_binaries`에서 CUDA 관련 항목 제거

---

## 배포 체크리스트

- [ ] 빌드 완료
- [ ] 실행 테스트
- [ ] 브라우저 접속 확인 (`localhost:8080`)
- [ ] 검색 기능 테스트
- [ ] 압축 및 배포

---

## 🔄 v2.7 빌드 반영 메모 (2026-02-20)

### 코드 변경 관련 빌드 영향
- `file_id` 기반 API 추가, ZIP 업로드 구현, 보안/인증 강화는 Python 소스 레벨 변경이며 빌드 시스템(PyInstaller) 자체 설정 변경은 필수 아님.
- PWA 아이콘 경로 정합화(`static/icons/icon-192.png`, `static/icons/icon-512.png`)는 `static` 폴더 전체가 spec의 `datas`에 포함되어 자동 반영됨.

### `.spec` 점검 결과
- 주요 spec(`regulation_search_gui.spec`, `regulation_search_ultra_lite_gui.spec`, `regulation_search_onefile.spec`, `regulation_search_ultra_lite.spec`) 모두 `datas`에 `(project_dir/static, static)`를 포함.
- `excluded_datas` 규칙에 `icons` 또는 `icon-*.png`를 직접 제거하는 패턴이 없어 추가 수정 없이 신규 아이콘 포함 가능.
- 결론:
  - 기능 변경 관점에서는 별도 spec 구조 변경이 불필요.
  - `regulation_search_ultra_lite_gui.spec` 시작 docstring 문법은 현재 정상(`"""`)이며 추가 수정이 필요하지 않음.

## ✅ 정합성 점검 메모 (2026-03-01)

- 대상 spec(`regulation_search_gui.spec`, `regulation_search_ultra_lite_gui.spec`, `regulation_search_onefile.spec`, `regulation_search_ultra_lite.spec`, `server_gui.spec`)은 `config` 폴더 전체 대신 `config/settings.example.json`만 포함하도록 통일함.
- `python -m py_compile *.spec` 기준 모든 spec 문법 정상 확인.
- 런타임 API/기능 변경 없이 빌드 산출물의 민감 설정 포함 위험만 축소함.

## 2026-03-15 추가 점검 메모

- 콘솔/레거시 spec(`internal_regulations.spec`, `internal_regulations_lite.spec`, `regulation_search.spec`, `regulation_search_lite.spec`, `server.spec`)도 `config/settings.example.json`을 포함하도록 정렬함.
- 최근 코드의 동적 import를 반영해 full-AI 계열 spec에는 다음 경로를 다시 확인함:
  - `flask.json.provider`
  - `flask_compress`
  - `langchain_text_splitters`
  - `langchain_core.documents`
  - `app.services.embeddings_backends`
- OCR은 여전히 선택 사항이며, 기본 requirements/spec 조합만으로는 이미지 PDF OCR이 자동 포함되지 않을 수 있음. OCR이 필요하면 `pytesseract`, `pdf2image`, `Pillow`, 시스템 Tesseract를 준비한 뒤 별도 검증 권장.

## 2026-03-16 추가 점검 메모

- 수동 `hiddenimports`를 쓰는 앱 기반 spec에 HWP/HWPX 파서 서브모듈 경로를 명시적으로 보강함:
  - `regulation_search_onefile.spec`
  - `regulation_search_ultra_lite.spec`
  - `regulation_search_ultra_lite_gui.spec`
  - `regulation_search_lite.spec`
  - `server_gui.spec`
- 위 조정으로 `app.services.parsers.*`가 정적 분석 누락 없이 포함되도록 보수적으로 정렬함.
- 권장 스모크 테스트:
  - `.hwpx` 업로드/미리보기
  - `.hwp` 업로드
  - ZIP 업로드 내 `.hwpx` 포함 케이스

### PowerShell spec 검증 명령

```powershell
Get-ChildItem -Name *.spec | ForEach-Object { python -m py_compile $_ }
```

### 권장 검증
1. 빌드 후 `dist/.../static/icons/`에 `icon-192.png`, `icon-512.png` 존재 확인
2. 실행 후 PWA install prompt 및 서비스워커 install 로그 확인
3. 관리자 화면에서 ZIP 업로드/파일 다운로드/리비전 기능 스모크 테스트

## 2026-04-16 spec 점검 메모

- 전체 `.spec` 문법 재점검:
  - `Get-ChildItem -Name *.spec | ForEach-Object { python -m py_compile $_ }`
  - 결과: 전체 정상
- `internal_regulations_lite.spec` 확인 결과:
  - `templates`, `static`, `config/settings.example.json` 포함 정책 유지
  - `collect_submodules("app")`로 현재 검색/파일/리비전 경로 변경분을 추가 수정 없이 수집 가능
  - 이번 검색 캐시/인덱스/리비전/ReaderMode 수정으로 인한 추가 `hiddenimports` 보강은 불필요
- 권장 회귀 검증:
  - `python -m pytest -q`
  - 파일 삭제 후 검색 스모크 테스트
  - 동명 파일 2개 포함 폴더 동기화 후 검색/미리보기 테스트

## ✅ 정합성 점검 메모 (2026-03-09)

- optional import 사용 경로와 빌드 스펙 동기화:
  - `server.spec`
  - `internal_regulations.spec`
  - `regulation_search.spec`
  - `regulation_search_gui.spec`
- 보강 항목:
  - `watchdog.observers`, `watchdog.events`
  - `langchain_text_splitters`, `langchain_core.documents`, `langchain_community.embeddings`, `langchain_community.vectorstores`
  - run.py 기반 full spec에서 `onnxruntime` 및 `app.services.embeddings_backends` 수집 경로 명시
