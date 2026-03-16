# 🔒 폐쇄망 환경 설정 가이드

> 사내 규정 검색기 v2.2 - 인터넷 연결 없이 운영하기

## 📋 개요

본 프로그램은 **폐쇄망(Air-gapped Network)** 환경에서도 완벽하게 동작하도록 설계되었습니다. 
AI 모델을 사전 다운로드하여 로컬에서 실행할 수 있습니다.

---

## 🚀 빠른 설정

### 1단계: 모델 및 정적 자원 다운로드 (인터넷 환경)

인터넷이 가능한 환경에서 AI 모델과 필수 정적 자원(JS 라이브러리)을 미리 다운로드합니다.

```bash
# 1. 모델 다운로드 (AI 기능용)
# 기본 경로(./models)에 다운로드
python download_models.py

# 2. 정적 자원 다운로드 (필수 라이브러리, 폰트 폴백 등)
# static/vendor 폴더 자동 생성 및 다운로드
python download_static.py
```

### 2단계: 설정 파일 수정

`config/settings.json` 파일을 열어 다음과 같이 설정합니다:

```json
{
  "folder": "",
  "theme": "dark",
  "offline_mode": true,
  "local_model_path": "./models/ko-sbert"
}
```

### 3단계: 프로그램 실행

```bash
python run.py
```

### 문서 포맷 참고

- `.hwpx`는 추가 패키지 없이 ZIP+XML 경로로 추출됩니다.
- `.hwp`는 `olefile` 의존성이 필요합니다. 기본 requirements에는 포함되어 있으며, 커스텀 설치 환경에서는 누락 여부를 확인해야 합니다.
- `.hwp` 또는 `.hwpx` 파싱 실패는 파일 단위로 처리되며, 서버 전체는 계속 동작합니다.

---

## ⚙️ 상세 설정

### 환경 변수 (자동 설정됨)

프로그램이 오프라인 모드로 실행될 때 다음 환경 변수가 자동으로 설정됩니다:

| 환경 변수 | 값 | 설명 |
|-----------|-----|------|
| `HF_HUB_OFFLINE` | `1` | HuggingFace Hub 오프라인 모드 |
| `TRANSFORMERS_OFFLINE` | `1` | Transformers 라이브러리 오프라인 모드 |

### 설정 옵션

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `offline_mode` | boolean | `false` | 오프라인 모드 활성화 |
| `local_model_path` | string | `""` | 로컬 모델 경로 |

### Python 코드에서 직접 설정

```python
from app.config import AppConfig

# 오프라인 모드 활성화
AppConfig.OFFLINE_MODE = True
AppConfig.LOCAL_MODEL_PATH = "D:/models/ko-sbert"
```

---

## 📦 사용 가능한 모델

| 모델명 | 크기 | 특징 |
|--------|------|------|
| SNU SBERT (고성능) | ~500MB | 한국어 특화, 높은 정확도 |
| BM-K Simal (균형) | ~400MB | 속도와 성능 균형 |
| JHGan SBERT (빠름) | ~350MB | 빠른 추론 속도 |

---

## 🔧 문제 해결

### 오류: "오프라인 모드에서는 로컬 모델 경로를 설정해야 합니다"

**원인:** `offline_mode`가 `true`이지만 `local_model_path`가 설정되지 않음

**해결:**
1. `config/settings.json`에서 `local_model_path` 경로 확인
2. 해당 경로에 모델 파일이 존재하는지 확인
3. 모델이 없으면 `download_models.py` 실행

### 오류: "모델 파일을 찾을 수 없습니다"

**원인:** 지정된 경로에 모델 파일이 없음

**해결:**
```bash
# 모델 파일 구조 확인
ls ./models/ko-sbert/

# 필요한 파일:
# - config.json
# - model.safetensors (또는 pytorch_model.bin)
# - tokenizer.json
# - tokenizer_config.json
```

### 오류: "네트워크 연결 시도"

**원인:** 오프라인 모드가 제대로 활성화되지 않음

**해결:**
1. `settings.json`의 `offline_mode`가 `true`인지 확인
2. 프로그램 재시작
3. 환경 변수 수동 설정:
   ```bash
   set HF_HUB_OFFLINE=1
   set TRANSFORMERS_OFFLINE=1
   python run.py
   ```

### 오류: "HWP 추출 실패" / "HWPX ZIP 오류"

**원인:**
- `.hwp`: `olefile` 누락 또는 손상된 문서
- `.hwpx`: ZIP 컨테이너 손상 또는 내부 `Contents/section*.xml` 누락

**해결:**
1. `pip show olefile`로 `.hwp` 의존성 설치 여부 확인
2. `.hwpx`는 압축 해제 가능한 정상 ZIP인지 확인
3. 문제 파일만 제외해도 나머지 업로드/검색 서비스는 계속 동작

---

## 📁 모델 폴더 구조

다운로드된 모델은 다음과 같은 구조를 가집니다:

```
models/
└── ko-sbert/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── vocab.txt
    └── special_tokens_map.json
```

---

## 🏗️ 빌드 및 배포

### PyInstaller 빌드 (폐쇄망 배포용)

```bash
# AI 기능 포함
pyinstaller regulation_search.spec

# 초경량 (BM25만, AI 제외)
pyinstaller regulation_search_lite.spec
```

### 배포 패키지 구성

폐쇄망 환경에 배포할 때 다음 파일들을 함께 전달합니다:

```
배포_패키지/
├── 사내규정검색기/           # 빌드된 실행 파일
│   ├── 사내규정검색기.exe
│   └── ...
├── models/                  # 사전 다운로드된 AI 모델
│   └── ko-sbert/
├── config/
│   └── settings.json        # offline_mode: true 설정됨
└── 설치_가이드.txt
```

---

## ✅ 체크리스트

폐쇄망 배포 전 확인사항:

- [ ] `download_models.py`로 모델 다운로드 완료
- [ ] `config/settings.json`에 `offline_mode: true` 설정
- [ ] `local_model_path` 경로가 올바르게 설정됨
- [ ] 모델 폴더에 필요한 파일들이 모두 존재
- [ ] 테스트 환경에서 오프라인 실행 확인
- [ ] PyInstaller 빌드 테스트 완료

---

## 📞 지원

문제가 지속되면 다음 정보와 함께 문의해 주세요:

1. `config/settings.json` 내용
2. 콘솔 에러 메시지
3. 모델 폴더 구조 (`dir /s models\`)
4. Python 버전 (`python --version`)

---

© 2026 사내 규정 검색기 - 폐쇄망 환경 가이드

---

## 🆕 v2.7 폐쇄망 운영 참고 (2026-02-20)

### Lite(BM25-only) 안정화
- `requirements_lite.txt` 환경에서 모델 미로드 상태라도 초기화/업로드/검색 흐름이 동작하도록 fallback 경로가 보강되었습니다.
- 운영 시 AI 임베딩 없이도 BM25 검색 서비스 구성이 가능하며, 필요 시 추후 모델만 추가 로드할 수 있습니다.

### 관리자 보안 기본값 변경
- 관리자 비밀번호 미설정 상태에서 더 이상 기본값(`admin`)으로 로그인되지 않습니다.
- 폐쇄망 배포 전 반드시 관리자 비밀번호를 설정해야 합니다.

### ZIP 업로드 (관리자)
- 관리자 API `POST /api/upload/folder`로 ZIP 묶음 업로드를 지원합니다.
- ZIP 엔트리 경로 정규화 및 `..` 차단(zip-slip 방어)이 적용되어 있습니다.

### 파일 식별자
- 동명이인 파일 충돌 방지를 위해 내부적으로 `file_id`가 도입되었습니다.
- 검색/태그/리비전/다운로드는 점진적으로 `file_id` 중심으로 동작합니다.

---

## 2026-03-15 운영 메모

- Lite/BM25-only 배포는 AI 모델이 없어도 초기 구동과 파일 업로드/텍스트 검색이 가능하도록 유지됩니다.
- OCR 지원은 계속 선택적입니다. 오프라인 배포본에서 이미지 PDF OCR까지 필요하면 `pytesseract`, `pdf2image`, `Pillow`, Tesseract 실행 파일을 별도로 준비해야 합니다.
- 콘솔/레거시 spec(`internal_regulations*.spec`, `regulation_search*.spec`, `server.spec`)도 `config/settings.example.json`을 동봉하는 방향으로 정합화했습니다.

## 2026-03-16 문서 추출 메모

- 폐쇄망 배포에서도 `.hwpx`는 추가 네트워크 없이 바로 처리됩니다.
- `.hwp`는 `olefile`만 준비되면 동작하며, 의존성 누락 시 안전한 fallback 오류 응답을 반환합니다.

### PowerShell에서 spec 점검

```powershell
Get-ChildItem -Name *.spec | ForEach-Object { python -m py_compile $_ }
```
