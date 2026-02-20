# 📦 빌드 가이드

## 빌드 옵션

| Spec 파일 | 모드 | AI 기능 | 예상 크기 | 특징 |
|-----------|------|---------|-----------|------|
| `regulation_search_gui.spec` | GUI | ✅ | 500-800MB | AI 벡터 검색 + BM25 |
| `regulation_search_ultra_lite_gui.spec` | GUI | ❌ | 600MB | BM25만 (torch 제외) |

> 💡 **v2.6.1**: 성능 최적화(압축, 캐싱) 및 오프라인 모드 지원

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

---

## 빌드 명령어

```powershell
cd "d:\google antigravity\Internal-Regulations-Finder-Online-main"

# AI 포함 버전 (벡터 검색 + BM25)
# [권장] 대부분의 환경에서 사용
pyinstaller regulation_search_gui.spec --clean

# Lite 버전 (BM25만, AI 제외)
# [저사양] AI 기능이 필요 없는 경우
pyinstaller regulation_search_ultra_lite_gui.spec --clean
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
  - 단, 실제 빌드 과정에서 `regulation_search_ultra_lite_gui.spec` 시작 docstring이 `\"\"\"`로 오기입되어 있어 `"""`로 1줄 수정 필요(문법 오류 해결).

### 권장 검증
1. 빌드 후 `dist/.../static/icons/`에 `icon-192.png`, `icon-512.png` 존재 확인
2. 실행 후 PWA install prompt 및 서비스워커 install 로그 확인
3. 관리자 화면에서 ZIP 업로드/파일 다운로드/리비전 기능 스모크 테스트
