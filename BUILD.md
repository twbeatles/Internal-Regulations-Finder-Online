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
