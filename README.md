# 📚 사내 규정 검색기 (Internal Regulations Finder) v3.0

> **RAG 대화형 질의응답 + AI 하이브리드 검색(Ko-SBERT + BM25) 사내 규정 및 사규 관리 솔루션**  
> 사내 규정, 지침, 업무 매뉴얼을 LLM 기반 대화형 RAG로 질문하고, 하이브리드 검색·조문 단위 열람 및 개정 이력 비교까지 지원합니다. (인터넷망 및 폐쇄망/오프라인 완벽 지원)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-green.svg)](https://flask.palletsprojects.com/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-brightgreen.svg)](https://riverbankcomputing.com/software/pyqt/)
[![RAG](https://img.shields.io/badge/RAG-Ollama%20%7C%20Cloud%20LLM-blueviolet.svg)](https://ollama.ai/)
[![ONNX Runtime](https://img.shields.io/badge/Engine-ONNX%20%7C%20PyTorch-orange.svg)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 💡 핵심 기능

* 🤖 **RAG 대화형 AI 질의응답**: 로컬 LLM(Ollama) 및 클라우드 LLM(OpenAI/Anthropic/Gemini)과 연동하여 사내 규정을 기반으로 정확한 답변과 조문 출처(Citation)를 실시간 SSE 스트리밍으로 제공
* 🔍 **AI 하이브리드 검색 & 질의 정규화**: 시맨틱 벡터 검색(Ko-SBERT)과 키워드 검색(BM25)을 결합하고, 조문 번호·금액 단위·동의어를 자동 확장하여 높은 정확도 보장
* 📑 **다양한 문서 포맷 지원 (Kordoc 통합)**: `HWP`, `HWPX`, `DOCX`, `PDF`, `XLSX`, `XLS`, `TXT` 문서의 본문, 표, 조문 구조 정보 자동 추출
* 📖 **조문 단위 파싱 & 리더 모드**: 제X조/제X장 단위의 스마트 분할, 검색어 하이라이트 순차 이동, 글자 크기 조절 가능한 원문 리더
* 🔄 **개정 이력 관리 & Diff 비교**: 규정 변경 시 버전별 이력 추적 및 추가/수정/삭제된 조항의 시각적 비교
* 🔌 **KCSC-MCP 연동 지원**: Model Context Protocol 서버를 내장하여 외부 AI 에이전트(Claude Desktop, Cursor, Grok 등)와 손쉽게 사내 규정 연동
* 🖥️ **PyQt6 시스템 트레이 GUI & 모던 웹 UI**: 백그라운드 트레이 구동, 윈도우 시작프로그램 등록, RAG 채팅/레거시 검색 탭 전환이 가능한 반응형 UI
* ⚡ **완전 오프라인(폐쇄망) 지원**: 인터넷이 차단된 사내망에서도 사전 다운로드된 모델, Ollama 로컬 LLM, 정적 리소스로 100% 로컬 구동 가능

---

## 🚀 빠른 시작 (Quick Start)

### 1. 시스템 요구사항

| 구분 | RAG + AI 하이브리드 모드 (기본) | 초경량 모드 (BM25 키워드 전용) |
|------|-------------------------------|-----------------------------|
| **Python** | 3.10 이상 | 3.10 이상 |
| **RAM** | 최소 8GB / 권장 16GB (로컬 LLM 구동 시) | 최소 2GB / 권장 4GB |
| **저장공간** | 약 3~5GB (임베딩 및 로컬 LLM 포함) | 약 200MB |
| **운영체제** | Windows 10/11, Linux, macOS | Windows 10/11, Linux, macOS |

### 2. 설치

```bash
# 1. 저장소 클론 및 이동
git clone <repository-url>
cd Internal-Regulations-Finder-Online

# 2. 가상환경 생성 및 활성화 (권장)
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux / macOS

# 3. 의존성 패키지 설치 (환경에 따라 선택)
pip install -r requirements.txt       # 전체 기능 (RAG + AI 하이브리드)
# pip install -r requirements_lite.txt # 초경량 모드 (BM25 전용)

# (선택) Kordoc 고도화 파서 브릿지 사용 시
# npm install
```

### 3. 서버 실행

```bash
# [추천] 시스템 트레이 지원 GUI 서버 실행
python server_gui.py

# 또는 백그라운드/콘솔 서버 실행
python run.py
```

* **사용자 웹 페이지 (RAG 채팅 & 검색)**: `http://localhost:8080`
* **관리자 설정 페이지**: `http://localhost:8080/admin`
* **MCP 서버 엔드포인트**: `http://localhost:8081` (설정 활성화 시)

---

## 📖 상세 사용 가이드 (User Guide)

### 1️⃣ 서버 실행 및 GUI 환경 설정 (`server_gui.py`)

GUI 서버를 실행하면 작업 표시줄 오른쪽 **시스템 트레이**에 상주하며 간편하게 서버를 제어할 수 있습니다.

```
[GUI 주요 기능]
┌─────────────────────────────────────────────────────────┐
│ 📚 사내 규정 검색기 서버                                │
├─────────────────────────────────────────────────────────┤
│ 상태: 🟢 실행 중 (http://localhost:8080)               │
│                                                         │
│ [ 웹 브라우저 열기 ]   [ 관리자 페이지 ]   [ 서버 재시작 ]│
├─────────────────────────────────────────────────────────┤
│ ⚙️ 환경 설정                                           │
│  - 서버 포트: [ 8080 ]                                  │
│  - AI 모델: [ SNU SBERT (고성능) ▼ ]                    │
│  - 임베딩 백엔드: [ onnx_fp32 (ONNX 고속 추론) ▼ ]       │
│  - 검색 모드: [ rag (RAG 대화형) ▼ ]                    │
│  - [x] Windows 시작 시 자동 실행                        │
│  - [x] 오프라인(폐쇄망) 모드 활성화                     │
│  - [ 관리자 비밀번호 변경 ]                             │
├─────────────────────────────────────────────────────────┤
│ 📋 실시간 서버 로그 모니터링...                         │
└─────────────────────────────────────────────────────────┘
```

* **트레이 최소화**: 창을 닫아도 트레이 영역에서 백그라운드로 계속 실행됩니다. (트레이 아이콘 우클릭으로 종료)
* **포트 및 모델/백엔드 변경**: 포트 번호(기본 `8080`), AI 모델, 임베딩 백엔드(`torch`, `onnx_fp32`, `onnx_int8`)를 UI에서 즉시 변경할 수 있습니다.
* **관리자 비밀번호 설정**: 웹 관리자 페이지 접속 시 사용할 비밀번호를 GUI에서 안전하게 등록/변경할 수 있습니다.

---

### 2️⃣ 규정 문서 등록 및 인덱싱 (관리자)

웹 브라우저에서 `http://localhost:8080/admin`으로 접속하여 사내 규정 문서를 시스템에 등록합니다.

```
[관리자 페이지 워크플로]
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ 1. 관리자 인증  │ ──> │ 2. 규정 문서    │ ──> │ 3. 자동 인덱싱  │
│ (비밀번호 입력) │     │    업로드 / 동기화 │   │ (청킹 + 벡터화) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

1. **관리자 인증**: 설정된 관리자 비밀번호를 입력하고 로그인합니다.
2. **문서 업로드**:
   * **개별 파일 업로드**: `HWPX`, `HWP`, `PDF`, `DOCX`, `XLSX`, `TXT` 파일을 드래그 앤 드롭하여 즉시 업로드합니다.
   * **ZIP 폴더 업로드**: 폴더째 압축된 ZIP 파일을 업로드하면 내부 폴더 구조를 유지하면서 일괄 추출 및 인덱싱됩니다.
3. **자동 폴더 동기화 (Watchdog)**:
   * 사내 공유 드라이브나 로컬 폴더 경로를 지정하고 `▶ 시작`을 누르면, 파일 추가/수정/삭제 시 실시간으로 검색 인덱스에 자동 반영됩니다.
4. **문서 관리 및 캐시 초기화**:
   * 등록된 문서 목록에서 파일별 상태, 청크 수, 버전을 확인하고 필요 시 `⚡ 재처리` 또는 `🗑️ 캐시 삭제`를 수행할 수 있습니다.

---

### 3️⃣ RAG 대화형 질문 & 하이브리드 검색 (사용자)

메인 페이지(`http://localhost:8080`)에서 자연어 질문이나 검색어를 입력하여 원하는 규정을 빠르게 찾습니다.

```
┌─────────────────────────────────────────────────────────┐
│ 📚 사내 규정 검색기 (RAG 모드)                          │
│                                                         │
│ [ 💬 질문: 부모님 회갑 때 경조사 휴가는 며칠인가요?    ][ 전송 ]│
│                                                         │
│ 🤖 답변:                                                │
│ "취업규칙 제25조에 따르면, 부모의 환갑(회갑) 시         │
│  유급휴가 1일이 부여됩니다. [출처: 취업규칙.docx (제25조)]" │
└─────────────────────────────────────────────────────────┘
```

* **🤖 RAG 대화형 질의응답 (기본)**:
  * 복잡한 질문도 사내 규정을 실시간 검색/참조하여 명확한 문장으로 정리하고, 근거 조항을 링크(출처)와 함께 안내합니다.
  * SSE(Server-Sent Events) 실시간 스트리밍으로 지연 없이 빠르게 답변을 확인합니다.
* **🔍 하이브리드 키워드 검색 모드**:
  * 상단 토글을 통해 기존 카드형 하이브리드 검색(`Vector + BM25`)으로 전환할 수 있습니다.
  * 관련도순, 파일명순, 길이순 정렬 및 파일 필터링 지원.
* **고급 검색 연산자 지원**:
  * `AND`, `OR`, `NOT`, `"정확한 문구"`, 정규식 검색 지원.

---

### 4️⃣ 상세 읽기, 북마크 & 결과 내보내기

* **📖 읽기 모드 (Reader Mode)**:
  * 검색 결과 카드 또는 RAG 출처 링크를 클릭하면 조문 단위로 깔끔하게 정리된 뷰어가 열립니다.
  * `원문` 버튼으로 전체 문서 원본 열람 및 글자 크기(`A+`/`A-`) 조절이 가능합니다.
* **🔦 하이라이트 순차 탐색**:
  * 문서 내에서 키워드가 강조 표시되며, <kbd>N</kbd>(다음) / <kbd>P</kbd>(이전) 키로 하이라이트 위치를 빠르게 넘나듭니다.
* **📌 북마크 & 💾 결과 내보내기**:
  * 자주 보는 규정은 즐겨찾기에 등록하고, 검색 결과를 `TXT`, `Markdown`, `JSON`, `PDF` 형식으로 다운로드할 수 있습니다.

---

### 5️⃣ 규정 개정 이력 관리 및 Diff 비교

규정이 개정되었을 때 이전 버전과 현재 버전의 차이점을 한눈에 파악할 수 있습니다.

```
[버전 비교 예시]
───────────────────────────────────────────────────────────
- 제12조 (연차유급휴가) 연간 15일의 유급휴가를 부여한다.
+ 제12조 (연차유급휴가) 연간 16일의 유급휴가를 부여하며, 근속연수에 따라 가산한다.
───────────────────────────────────────────────────────────
```

* 관리자 페이지 또는 문서 메뉴에서 `버전 이력`을 선택합니다.
* 비교할 두 버전을 선택하면 추가된 내용(초록색 `+`), 삭제된 내용(빨간색 `-`), 수정된 문단이 시각적인 Diff로 표시됩니다.

---

### 6️⃣ KCSC-MCP (Model Context Protocol) 연동

사내 규정 검색기 v3.0은 **MCP 서버**를 내장하여 외부 AI 도구에서 사내 규정을 조회할 수 있습니다.

* `config/settings.json`에서 `"mcp": {"enabled": true, "port": 8081}` 설정
* Claude Desktop 또는 Cursor의 MCP 설정에 추가:
```json
{
  "mcpServers": {
    "regulations": {
      "url": "http://127.0.0.1:8081/sse"
    }
  }
}
```

---

### 7️⃣ 오프라인(폐쇄망) 환경 완벽 구동 가이드

외부 인터넷 연결이 불가능한 폐쇄망 환경에서도 아래 3단계로 완벽하게 작동합니다.

#### 1) 인터넷이 되는 PC에서 리소스 준비
```bash
# 1. HuggingFace AI 모델 다운로드 (models/ 디렉토리에 저장)
python download_models.py

# 2. 웹 UI 필수 라이브러리 로컬 다운로드 (static/vendor/ 에 저장)
python download_static.py

# 3. (RAG 사용 시) Ollama 모델 준비 (예: qwen2.5, llama3 등)
```

#### 2) 폐쇄망 서버로 파일 복사
* 소스 코드 전체와 함께 생성된 `models/` 및 `static/vendor/` 폴더를 폐쇄망 서버로 이동합니다.

#### 3) 오프라인 모드 실행
* `config/settings.json`에서 `"offline_mode": true`로 설정하거나 GUI에서 오프라인 모드를 체크합니다.
```json
{
  "offline_mode": true,
  "local_model_path": "./models/snunlp--KR-SBERT-V40K-klueNLI-augSTS",
  "embed_backend": "onnx_fp32",
  "search_mode": "rag"
}
```

---

## ⌨️ 키보드 단축키 일람

| 단축키 | 기능 | 설명 |
|--------|------|------|
| <kbd>Ctrl</kbd> + <kbd>K</kbd> | **검색/질문창 포커스** | 어느 위치에서든 즉시 입력창으로 이동 |
| <kbd>Enter</kbd> | **검색 / 질문 전송** | 검색 실행 또는 RAG 질문 전송 |
| <kbd>J</kbd> / <kbd>↓</kbd> | **다음 결과 선택** | 검색 결과 목록에서 아래 항목으로 이동 |
| <kbd>K</kbd> / <kbd>↑</kbd> | **이전 결과 선택** | 검색 결과 목록에서 위 항목으로 이동 |
| <kbd>N</kbd> | **다음 하이라이트** | 본문 내 검색어 일치 위치로 다음 이동 |
| <kbd>P</kbd> | **이전 하이라이트** | 본문 내 검색어 일치 위치로 이전 이동 |
| <kbd>R</kbd> | **읽기 모드** | 선택된 문서의 상세 리더 창 토글 |
| <kbd>T</kbd> | **테마 전환** | 다크 모드 ⇄ 라이트 모드 전환 |
| <kbd>?</kbd> | **단축키 안내** | 키보드 단축키 도움말 모달 표시 |
| <kbd>Esc</kbd> | **창 닫기** | 열려 있는 모달/뷰어 닫기 |

---

## ⚙️ 설정 레퍼런스 (Configuration)

### `app/config.py` & `config/settings.json`

| 설정 변수 | 기본값 | 설명 |
|-----------|--------|------|
| `SERVER_PORT` | `8080` | 웹/API 서버 포트 |
| `DEFAULT_MODEL` | `SNU SBERT (고성능)` | 기본 한국어 임베딩 모델 (`snunlp/KR-SBERT-V40K-klueNLI-augSTS`) |
| `EMBED_BACKEND` | `onnx_fp32` | 임베딩 엔진 (`torch`, `onnx_fp32`, `onnx_int8`) |
| `search_mode` | `rag` | 기본 검색 모드 (`rag` 또는 `legacy`) |
| `CHUNK_SIZE` | `800` | 문서 분할 시 조문 단위 청크 크기 (글자 수) |
| `CHUNK_OVERLAP` | `80` | 청크 간 중첩 글자 수 (문맥 보존) |
| `VECTOR_WEIGHT` | `0.7` | 하이브리드 검색 시 시맨틱 벡터 점수 가중치 (0.0 ~ 1.0) |
| `BM25_WEIGHT` | `0.3` | 하이브리드 검색 시 키워드 BM25 점수 가중치 (0.0 ~ 1.0) |
| `SEARCH_CACHE_SIZE` | `1000` | LRU 검색 캐시 최대 항목 수 |
| `SEARCH_CACHE_TTL` | `600` | 검색 캐시 유지 시간 (초, 적응형 2배 연장) |
| `RATE_LIMIT_PER_MINUTE` | `300` | IP당 분당 최대 요청 수 |
| `MAX_CONTENT_LENGTH` | `50MB` | 단일 파일 업로드 최대 크기 |

---

## 📦 배포 및 바이너리 빌드 (PyInstaller)

Python이 설치되지 않은 사용자 PC 배포를 위해 PyInstaller 빌드 스크립트를 제공합니다.

| Spec 파일 | 대상 환경 | AI 기능 | 예상 크기 | 용도 |
|-----------|-----------|---------|-----------|------|
| `regulation_search_gui.spec` | PyQt6 GUI | ✅ RAG + 임베딩 | ~600MB | **일반 권장 배포본** (고성능 AI 검색) |
| `regulation_search_ultra_lite_gui.spec` | PyQt6 GUI | ❌ BM25만 | 60~100MB | 저사양 PC 및 초경량 배포 |
| `regulation_search_onefile.spec` | 단일 exe | ❌ BM25만 | 40~60MB | USB 휴대용 단일 파일 |

```bash
# 1. 정식 GUI 버전 빌드 (권장)
pyinstaller regulation_search_gui.spec --clean

# 2. 초경량 GUI 버전 빌드
pyinstaller regulation_search_ultra_lite_gui.spec --clean

# 3. 단일 실행파일(Onefile) 빌드
pyinstaller regulation_search_onefile.spec --clean
```

> **빌드 결과물 위치**: `dist/` 폴더 내 실행 파일 생성

---

## 🔌 주요 REST API 요약

### 1. RAG & 검색 API
| 메서드 | 엔드포인트 | 설명 |
|--------|------------|------|
| `POST` | `/api/rag/chat` | RAG 대화형 스트리밍 질의응답 (SSE 지원) |
| `POST` | `/api/search` | 하이브리드/키워드 검색 (`query`, `k`, `hybrid`, `sort_by`, `filter_file_id`) |
| `GET` | `/api/search/suggest` | 검색어 자동완성 추천 (`?q=검색어`) |
| `GET` | `/api/search/history` | 최근 및 인기 검색어 히스토리 조회 |
| `POST` | `/api/cache/clear` | 검색 캐시 수동 초기화 |

### 2. 파일 & 개정 관리 API
| 메서드 | 엔드포인트 | 설명 |
|--------|------------|------|
| `GET` | `/api/files` | 등록된 파일 목록 및 상태 조회 |
| `POST` | `/api/upload` | 단일/다중 파일 업로드 및 자동 인덱싱 |
| `POST` | `/api/upload/folder` | ZIP 압축 폴더 업로드 |
| `GET` | `/api/files/by-id/<file_id>/preview` | 파일 원본 텍스트 미리보기 |
| `GET` | `/api/files/by-id/<file_id>/download` | 파일 원본 다운로드 |
| `DELETE` | `/api/files/by-id/<file_id>` | 파일 삭제 (`delete_source=true` 시 물리 삭제) |
| `GET` | `/api/files/by-id/<file_id>/versions` | 문서의 버전별 개정 이력 조회 |
| `GET` | `/api/files/by-id/<file_id>/versions/compare` | 두 버전 간 차이점(Diff) 데이터 조회 |

### 3. 시스템 및 동기화 API
| 메서드 | 엔드포인트 | 설명 |
|--------|------------|------|
| `GET` | `/api/status` | 서버 준비 상태 및 인덱싱 진행률 (`load_progress`) |
| `GET` | `/api/health` | 서버 헬스체크 |
| `GET` / `POST` | `/api/models` | 사용 가능한 AI 모델 목록 조회 및 모델 동적 변경 |
| `POST` | `/api/sync/start` | 폴더 실시간 감시 동기화(Watchdog) 시작 |
| `POST` | `/api/sync/stop` | 동기화 중지 요청 |

---

## 🧪 성능 벤치마크 및 테스트

```bash
# 1. 단위 및 통합 테스트 실행 (104개 테스트 통과)
python -m pytest -q

# 2. 검색 성능 스모크 벤치마크 (동시성 1/5/10, 워밍업 30회, 측정 200회)
python scripts/perf_smoke.py --base-url http://127.0.0.1:8080 --query "휴가 규정"
```

* **평균 검색 응답 속도**: ~80ms (캐시 적중 시 < 5ms)
* **캐시 히트율**: ~90% (적응형 TTL 적용)
* **테스트 스위트**: 104 passed (2026-07-15 hardening 포함)

---

## 🔒 보안 아키텍처

* **XSS 방어**: 프론트엔드 전 영역에서 텍스트 기반 안전 렌더링(`escapeHtml` / `textContent`) 적용
* **Path Traversal 방지**: 업로드 및 미리보기 경로의 정규화(`normpath`)와 상위 디렉터리(`..`) 참조 차단
* **인증 및 상태 변경 제어**: 파일 업로드, 삭제, 태그 수정, 모델 변경 등 상태 변경 API는 관리자 인증 강제
* **CORS Allowlist & 세션 쿠키**: `HttpOnly`, `SameSite=Lax` 정책 적용 및 허용된 오리진만 통신 허용
* **감사 Hardening**: 인덱싱 단일 비행 락, 관리자 로그인 속도 제한, ZIP 압축 해제 용량 실시간 감시

---

## 🐛 트러블슈팅 & FAQ

**Q1. AI 모델 로드 시 메모리 부족(OOM) 오류가 발생합니다.**  
A. `requirements_lite.txt`를 사용하는 초경량 모드(`regulation_search_ultra_lite_gui.spec`)를 사용하거나, `settings.json`에서 `"embed_backend": "onnx_int8"`로 변경하여 메모리 사용량을 대폭 절감할 수 있습니다.

**Q2. 한글(.hwp/.hwpx) 문서 내용이 추출되지 않습니다.**  
A. `.hwpx`는 기본 파서로 동작하며, `.hwp`는 `pip install olefile`이 필요합니다. 더 높은 품질의 파싱을 원하시면 `npm install` 후 Kordoc 브릿지를 활성화하세요.

**Q3. 사내 폐쇄망에서 폰트나 아이콘이 깨집니다.**  
A. 인터넷이 되는 환경에서 `python download_static.py`를 실행하여 `static/vendor/` 폴더를 생성한 후 폐쇄망 서버로 함께 복사하면 시스템 기본 폰트와 로컬 리소스로 자동 전환됩니다.

**Q4. 서버를 다른 PC에서 접속하게 하려면 어떻게 하나요?**  
A. 방화벽에서 포트(기본 `8080`)를 허용하고, 접속하려는 PC의 웹 브라우저에서 `http://[서버IP]:8080`으로 접속하면 됩니다.

---

## 📜 릴리즈 노트 (v3.0 기준)

* **v3.0.0**: RAG 대화형 질의응답 탑재, KCSC-MCP 서버 내장, Kordoc 파서 통합, SOLID 아키텍처 리팩토링 및 감사 hardening
* **v2.8.3**: 파일 필터 캐시 키 분리, 삭제 시 인메모리 벡터/BM25 동시 재구축, 리비전 버전 연속성 보강
* **v2.8.2**: `.hwpx` 전용 ZIP/XML 파서 추가 및 `.hwp` olefile 엔진 분리 안정화
* **v2.8.0**: CORS 화이트리스트 도입, 세션 쿠키 보안 강화, ZIP 폴더 업로드 기능 탑재
* **v2.7.0**: 파일 고유 식별자(`file_id`) 도입, 관리자 인증 강화, 안전한 클라이언트 하이라이트 전환
* **v2.6.1**: ONNX Runtime 임베딩 백엔드 지원, 적응형 LRU 캐시, 완전 오프라인 정적 다운로더 추가

---

## 📝 라이선스

본 프로젝트는 **MIT License**에 따라 자유롭게 수정 및 배포가 가능합니다.  
© 2026 사내 규정 검색기 (Internal Regulations Finder).
