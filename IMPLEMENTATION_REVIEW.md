# IMPLEMENTATION_REVIEW.md

> 업데이트(2026-02-25): 본 문서는 **사전 점검 리포트**이며, 이후 A-01~A-15 개선 구현이 완료되었습니다.  
> 최신 구현 상태와 검증 결과는 `IMPLEMENTATION_AUDIT_2026-02-25.md`의 "구현 완료 현황" 섹션을 우선 참조하세요.

### 반영 완료된 핵심 항목(요약)
- CORS allowlist/세션 쿠키 정책/운영 fail-fast(waitress) 적용
- `/api/search/history`/`/api/search/suggest`/`/api/status`/`/api/health` 스키마 정합화
- 파일 삭제 기본 `index_only`, ZIP 제한 파라미터 도입
- `/api/sync/stop` 취소 이벤트 구현, 모델 변경 시 기본 재인덱스 정책 반영
- 프론트(XSS/하이라이트/테마), Service Worker 캐시 정책 정합화
- 테스트 진입점 `pytest -q` 단일화

## 1. 목적/범위

### 목적
- `CLAUDE.md`, `README.md`에 명시된 기능/아키텍처와 실제 구현(`run.py` + `app/*` + `static/*`) 간 불일치 및 잠재 리스크를 식별한다.
- 구현 리스크를 **문제 진단 + 원인 + 구체 수정안 + 검증 테스트** 단위로 정리해 즉시 실행 가능한 개선 백로그를 만든다.

### 범위
- 포함:
  - API/프론트 인터페이스 정합성
  - 인증/권한/입력검증/XSS/경로 처리
  - 검색/업로드/동기화/태그/리비전/PWA 동작 리스크
  - BM25-lite 경로 안정성
- 제외:
  - 실제 코드 수정/리팩터링
  - legacy `server.py` 기준 기능 복원

### 기준
- 실행 기준 런타임: `run.py` + `app/*` (legacy `server.py`는 비교 참고만 사용)
- 심각도: `Critical / High / Medium / Low`

---

## 2. 점검 방법

### 문서 기준선 확인
- `CLAUDE.md`, `README.md`의 엔드포인트/기능/구조/보안 설명을 기준선으로 사용.

### 정적 점검
- 라우트/서비스/프론트 코드를 교차 검토:
  - `app/routes/*.py`
  - `app/services/*.py`
  - `app/utils.py`
  - `static/app.js`, `static/sw.js`, `templates/*.html`

### 런타임 확인(비변경)
- URL 맵 점검으로 실제 등록 라우트 확인.
- 대표 API 호출로 응답 스키마 확인.
- 테스트 실행:
  - `pytest -q` 기준 통과 확인

---

## 3. 핵심 결론 (Executive Summary)

- 현재 시스템은 핵심 검색 기능 자체는 동작하지만, **프론트-백 인터페이스 불일치**와 **기능 단절(다운로드/ZIP 업로드)**, **보안 경계 취약점(XSS/경로 주입/기본 관리자 비밀번호 fallback)** 때문에 운영 안정성과 신뢰도가 떨어진다.
- 특히 다음 3가지는 즉시 대응이 필요하다.
  - `Critical`: 검색 결과 렌더링 XSS 위험
  - `Critical`: 개정 이력 저장 경로 주입 가능성
  - `Critical`: 해시 미설정 시 기본 관리자 비밀번호 `"admin"` 허용
- 기능적 관점에서는 문서/코드에 명시된 기능 중 일부가 실제로는 깨져 있으며(다운로드, ZIP 업로드, 일부 관리자 모달 데이터), API 응답 스키마 표준화가 최우선이다.

---

## 4. 상세 이슈 목록 (심각도 순)

## Critical

### C-01. 검색 결과 렌더링 XSS 위험
- 증상:
  - 서버가 하이라이트된 HTML을 생성해 내려주고(`app/routes/api_search.py:81`, `app/routes/api_search.py:82`),
  - 클라이언트가 이를 신뢰하고 DOM에 삽입(`static/app.js:2035`, `static/app.js:2045`).
- 영향:
  - 문서 내용에 악성 스크립트/이벤트 속성이 포함될 경우 실행될 수 있음.
  - 내부망 환경에서도 저장형 XSS로 관리자/사용자 세션 악용 가능.
- 원인:
  - `content_highlighted`를 “표시용 안전 마크업”으로 강제하지 않고 일반 HTML 문자열로 취급.
- 권장 수정안:
  - 서버:
    - 원문을 먼저 HTML escape한 뒤, escape된 텍스트에서만 `<mark>` 삽입.
    - 허용 태그 화이트리스트(`<mark>` only) sanitize 적용.
  - 클라이언트:
    - `content_highlighted`를 그대로 렌더링하지 말고 sanitize 후 렌더링.
    - 또는 서버 하이라이트 제거 후 클라이언트에서 안전 하이라이트 수행.
- 검증 테스트:
  - `<img src=x onerror=alert(1)>` 포함 문서 검색 시 alert 미실행 확인.
  - `<script>...</script>` 문자열이 텍스트로만 표시되는지 확인.

### C-02. 개정 이력 저장 경로 주입 가능성
- 증상:
  - `/api/revisions` POST 입력 `filename`을 별도 정규화 없이 사용(`app/routes/api_files.py:375`, `app/routes/api_files.py:378`).
  - `revision_filename = f"{filename}_{version}_{timestamp}.txt"`로 파일명 조합(`app/services/file_manager.py:31`) 후 경로 결합(`app/services/file_manager.py:32`).
- 영향:
  - `filename`에 경로 구분자/상대경로가 들어오면 의도치 않은 경로 쓰기 시도 가능.
- 원인:
  - 리비전 저장 경로에서 basename 강제 및 안전 파일명 정규화 미적용.
- 권장 수정안:
  - 서버에서 `filename`을 반드시 basename으로 강제 후 허용 문자만 남기기.
  - 저장 대상 경로가 `revisions` 디렉토리 내부인지 `resolve()`로 최종 검증.
  - `/api/revisions` POST에 관리자 권한 적용(권한 경계와 함께 처리).
- 검증 테스트:
  - `filename="../x"` 요청 시 400 거부 및 파일 미생성 확인.
  - 정상 파일명 요청은 기존과 동일 동작 확인.

### C-03. 기본 관리자 비밀번호 fallback (`"admin"`)
- 증상:
  - 저장된 해시가 없으면 `"admin"` 허용(`app/services/settings_store.py:134`~`app/services/settings_store.py:135`).
- 영향:
  - 초기 배포/설정 누락 환경에서 인증 우회 가능성 증가.
- 원인:
  - Back-compat 정책이 보안 기본값보다 우선.
- 권장 수정안:
  - 기본 fallback 제거.
  - 최초 실행 시 필수 비밀번호 설정 플로우 강제.
  - 운영 모드에서는 `ADMIN_PASSWORD(_HASH)` 또는 settings 해시가 없으면 관리 기능 비활성.
- 검증 테스트:
  - 비밀번호 미설정 상태에서 `/api/admin/auth` 실패(401) 확인.
  - 설정 완료 후 정상 로그인 확인.

---

## High

### H-01. 다운로드 기능 단절
- 증상:
  - 클라이언트는 `/api/files/{filename}/download` 링크 제공(`static/app.js:2068`).
  - 실제 라우트 미등록(앱 URL 맵 기준 없음).
- 영향:
  - UI상 “원본 파일 다운로드” 버튼이 실제로는 동작하지 않음.
- 원인:
  - legacy `server.py` 라우트가 모듈형 `app/routes`로 이관되지 않음.
- 권장 수정안:
  - `app/routes/api_files.py`에 `GET /files/<path:filename>/download` 추가.
  - 또는 프론트에서 해당 버튼 제거/대체.
- 검증 테스트:
  - 링크 클릭 시 200 + 첨부 파일 응답 확인.

### H-02. 프론트-백 응답 스키마 불일치 (동기화 상태)
- 증상:
  - 프론트는 `result.status.running` 기대(`static/app.js:2379`).
  - 백엔드는 `is_syncing` 반환(`app/routes/api_system.py:171`, `app/routes/api_system.py:175`).
- 영향:
  - 동기화 UI 상태(시작/중지 버튼, 인디케이터) 오동작.
- 원인:
  - API 스키마 변경 후 프론트 동기화 미완료.
- 권장 수정안:
  - 응답을 `{ success, status: { running, current_folder, progress, error } }`로 통일하거나,
  - 프론트를 `is_syncing` 기준으로 수정.
- 검증 테스트:
  - 동기화 시작/중지 시 UI 상태가 즉시 일치하는지 확인.

### H-03. 프론트-백 응답 스키마 불일치 (상태 진행률)
- 증상:
  - 프론트는 `result.progress` 사용(`static/app.js:1411`).
  - 백엔드는 `load_progress` 제공(`app/routes/api_system.py:82`~`app/routes/api_system.py:87`).
- 영향:
  - 로딩 메시지 표시 누락/오표시.
- 원인:
  - 네이밍 불일치.
- 권장 수정안:
  - 백엔드 `progress` alias 제공(기존 `load_progress` 유지) 또는 프론트 키 수정.
- 검증 테스트:
  - 초기화 진행 중 상태 배지가 퍼센트/메시지를 정확히 보여주는지 확인.

### H-04. 프론트-백 응답 스키마 불일치 (리비전)
- 증상:
  - 프론트는 `result.revisions` 기대(`static/app.js:2938`, `static/app.js:3614`).
  - `/api/revisions` GET은 `history` 반환(`app/routes/api_files.py:373`).
- 영향:
  - 변경 이력/버전 비교 모달에서 빈 목록으로 인식.
- 원인:
  - 응답 필드명 이중 체계 미정리.
- 권장 수정안:
  - `/api/revisions` GET에 `revisions` alias 추가(하위호환 유지) 또는 프론트 수정.
- 검증 테스트:
  - 리비전 존재 시 두 모달 모두 동일 데이터 표시 확인.

### H-05. 프론트-백 응답 스키마 불일치 (자동태그)
- 증상:
  - 프론트는 `res.tags` 기대(`static/app.js:2910`).
  - 백엔드는 `suggested_tags` 반환(`app/routes/api_files.py:567`).
- 영향:
  - 자동태그 버튼 성공해도 UI 반영 실패.
- 원인:
  - API 응답 명세와 프론트 사용 키 불일치.
- 권장 수정안:
  - `/api/tags/auto` 응답에 `tags` alias 추가 또는 프론트에서 `suggested_tags` 사용.
- 검증 테스트:
  - 자동태그 후 태그 칩 즉시 렌더링 및 저장 확인.

### H-06. ZIP 폴더 업로드 기능 런타임 오류 (해결됨)
- 증상:
  - 사전 점검 시점에는 `uploadFolderZip()` 경로에서 API 연결 불일치가 존재했음.
  - 현재는 `POST /api/upload/folder` 및 클라이언트 호출 경로가 구현되어 정상 동작.
- 영향:
  - (과거) 관리자 페이지에서 ZIP 업로드 즉시 JS 오류 가능.
- 원인:
  - (과거) UI 기능 노출 대비 API 구현 누락.
- 권장 수정안:
  - 백엔드 ZIP 업로드/해제/인덱싱 구현 + API 메서드 연결(완료).
- 검증 테스트:
  - ZIP 업로드 경로에서 JS 에러가 발생하지 않는지 확인.

### H-07. Lite(BM25-only) 경로 불안정
- 증상:
  - 코드상 BM25-only 지원 의도는 있으나(`app/services/search.py:798`~`app/services/search.py:799`),
  - `_get_cache_dir`가 `model_id` 미설정 시 예외(`app/services/search.py:681`, `app/services/search.py:687`),
  - 처리 중 `CharacterTextSplitter` 의존(`app/services/search.py:885`)이 남아 있음.
- 영향:
  - `requirements_lite` 환경에서 초기화/인덱싱 실패 가능.
- 원인:
  - “AI 비의존 경로”와 실제 처리 경로가 완전히 분리되지 않음.
- 권장 수정안:
  - BM25-only 모드 전용 캐시/분할 경로 분리.
  - `model_id` 없는 경우 안전한 BM25 캐시 경로 사용.
  - `CharacterTextSplitter` 대신 `DocumentSplitter` fallback 고정.
- 검증 테스트:
  - lite 의존성만 설치한 환경에서 초기화/업로드/검색 전 시나리오 통과.

### H-08. 태그 키 일관성 붕괴 및 동명이인 파일 충돌
- 증상:
  - basename 기반 탐색/태깅(`app/routes/api_files.py:77`, `app/routes/api_files.py:102`, `app/routes/api_files.py:321`)과
  - full path 기반 태그 저장(`app/routes/api_files.py:533`)이 혼재.
- 영향:
  - 동명 파일 다중 존재 시 태그/미리보기/필터 충돌 및 오동작.
- 원인:
  - 파일 식별자 정책(절대경로 vs basename vs UUID) 미정의.
- 권장 수정안:
  - 단일 식별자 체계 도입(권장: 내부 `file_id`), UI 표시명은 basename 분리.
  - 태그/리비전/preview/search 모두 동일 키로 통일.
- 검증 테스트:
  - 같은 파일명 2개 업로드 후 각각 태그/미리보기/필터 정확도 확인.

### H-09. 권한 경계 재검토 필요 엔드포인트
- 증상:
  - 상태 변경성/관리성 기능 중 일부가 인증 없이 열려 있음:
    - `/api/upload` (`app/routes/api_files.py:221`)
    - `/api/revisions` POST (`app/routes/api_files.py:375`)
    - `/api/tags/set` (`app/routes/api_files.py:517`)
    - `/api/tags/auto` (`app/routes/api_files.py:541`)
    - `/api/cache/clear` (`app/routes/api_search.py:156`)
- 영향:
  - 비관리자 사용자가 시스템 상태를 변경할 수 있음.
- 원인:
  - `admin_required` 적용 기준이 기능별로 일관되지 않음.
- 권장 수정안:
  - “읽기/조회” vs “상태변경” 정책 문서화 후 일괄 적용.
  - 최소: `upload`, `revisions POST`, `tags/set`, `cache/clear`는 관리자 제한 검토.
- 검증 테스트:
  - 비인증 세션에서 대상 API 호출 시 401 반환 확인.

---

## Medium

### M-01. 업로드 확장자 UX 불일치
- 증상:
  - 백엔드는 `.txt/.docx/.pdf/.xlsx/.xls/.hwp` 지원(`app/config.py:32`).
  - 프론트 필터는 `.txt/.docx/.pdf`만 허용(`static/app.js:2435`).
- 영향:
  - 사용자가 지원 파일도 업로드 불가로 오인.
- 원인:
  - 프론트 필터 로직이 최신 지원 확장자와 불일치.
- 권장 수정안:
  - 프론트 필터 목록을 백엔드와 동일하게 통일(또는 서버에서 목록 내려주기).
- 검증 테스트:
  - `.xlsx/.xls/.hwp` 업로드가 프론트 단계에서 차단되지 않는지 확인.

### M-02. PWA 아이콘 경로 불일치/누락
- 증상:
  - `templates/index.html`, `static/manifest.json`, `static/sw.js`는 `/static/icons/...` 참조.
  - 실제 파일은 `static/icon-192.png`만 존재(icons 디렉토리 없음).
- 영향:
  - PWA 설치 아이콘 누락, SW 캐시 실패/경고 발생 가능.
- 원인:
  - 리소스 경로 이관 누락.
- 권장 수정안:
  - `static/icons/icon-192.png`, `static/icons/icon-512.png` 실제 배치 또는 참조 경로 정정.
- 검증 테스트:
  - PWA 설치 시 아이콘 정상 표시, SW install 단계 에러 없음 확인.

### M-03. 파일명 sanitize 정규식 오동작 가능성
- 증상:
  - `app/utils.py:275`, `app/utils.py:276`의 정규식 이스케이프 패턴이 의도와 달리 일부 특수문자/공백 정규화를 놓칠 수 있음.
- 영향:
  - 파일명 정제 일관성 저하 및 예외 케이스 누락.
- 원인:
  - 문자 클래스/이스케이프 조합의 가독성 및 안정성 부족.
- 권장 수정안:
  - 허용 문자 정책을 명시하고 정규식 재정의 + 단위테스트 보강.
  - 예약어/특수문자/다중공백/유니코드 케이스 고정 테스트 추가.
- 검증 테스트:
  - `?`, `:`, 다중 공백, 예약어(`CON`, `LPT1`) 케이스 통과 여부 확인.

---

## Low

### L-01. 테스트 진입점 일관성 저하 (해결됨)
- 증상:
  - 사전 점검 시점에는 환경에 따라 `pytest -q` import path 이슈가 발생했음.
- 영향:
  - (과거) CI/개발자 로컬 실행 방식 불일치로 혼선.
- 원인:
  - (과거) 테스트 실행 진입점/환경 변수 의존.
- 권장 수정안:
  - `pytest.ini`에 `pythonpath = .` 설정 및 `README.md` 실행 명령 `pytest -q` 통일(완료).

---

## 5. 권장 API/인터페이스 정합화 변경안

### A. `/api/sync/status` 응답 스키마
- 권장 표준:
```json
{
  "success": true,
  "status": {
    "running": false,
    "current_folder": "",
    "progress": "",
    "error": ""
  }
}
```
- 호환 전략:
  - 1단계: 기존 필드(`is_syncing`, `current_folder`, `progress`, `error`) 유지 + `status` 병행 제공.
  - 2단계: 프론트 전환 완료 후 구필드 제거.

### B. `/api/status` 진행률 키
- 선택지:
  - 백엔드에서 `progress` alias 추가 (`load_progress` 유지).
  - 또는 프론트가 `load_progress`를 사용하도록 일괄 수정.

### C. `/api/revisions` GET 응답
- 선택지:
  - `history` + `revisions` 동시 제공.
  - 또는 프론트에서 `history` 사용으로 통일.

### D. `/api/tags/auto` 응답
- 선택지:
  - `suggested_tags` + `tags` 동시 제공.
  - 또는 프론트에서 `suggested_tags` 사용.

### E. 다운로드 API 정합성
- `/api/files/<path:filename>/download` GET 추가 또는 프론트 다운로드 액션 제거.
- 경로 검증/동명이인 파일 처리 정책(식별자) 함께 정의 필요.

---

## 6. 테스트 시나리오 (회귀 포함)

1. `/api/files/{filename}/download` 링크 클릭 시 200 + 파일 다운로드.
2. `/api/sync/status` 폴링에서 UI 상태 점등/버튼 활성화가 실제 동기화 상태와 일치.
3. `/api/revisions`/`/api/tags/auto` 호출 후 관리자 모달의 목록/자동태그 즉시 반영.
4. ZIP 업로드 버튼 클릭 시 JS 오류 없이 정상 처리 또는 기능 비활성화 안내.
5. 악성 HTML 포함 문서 검색 결과 렌더링 시 스크립트 실행 불가.
6. `filename="../x"` 형태로 리비전 저장 요청 시 경로 이탈 저장 차단.
7. BM25-only 환경(`requirements_lite`)에서 초기화/검색/업로드 후 검색까지 동작.
8. `.xlsx/.xls/.hwp` 업로드가 UI에서 차단되지 않고 백엔드 처리 성공.
9. PWA install/service worker install 시 아이콘/캐시 실패 없음.
10. 동명 파일 2개 존재 시 preview/tag/revision/filter 동작이 충돌 없이 식별 가능.
11. 비관리자 세션에서 상태변경성 엔드포인트 접근 차단 검증.
12. 파일명 sanitize 단위테스트(`?`, `:`, 다중 공백, 예약어) 통과.

---

## 7. 우선순위 실행 로드맵 (P0/P1/P2)

### P0 (즉시, 보안/장애)
- C-01 XSS 방어
- C-02 리비전 경로 주입 차단
- C-03 기본 관리자 비밀번호 fallback 제거
- H-01 다운로드 기능 복구(또는 버튼 제거)
- H-06 ZIP 업로드 런타임 오류 제거

### P1 (기능 정합성/운영 안정화)
- H-02/H-03/H-04/H-05 응답 스키마 정합화
- H-08 태그/파일 식별자 일관화
- H-09 권한 경계 재정의 및 적용
- M-01 업로드 확장자 UX 정합화

### P2 (품질/유지보수)
- M-02 PWA 리소스 경로 정비
- M-03 sanitize 정규식 리팩터링 + 테스트 강화
- L-01 테스트 실행 표준화 문서화
- H-07 BM25-only 경로 명시적 분리/안정화

---

## 8. 부록 (근거 파일/라인, 재현 절차)

### 근거 파일/라인
- 다운로드 링크 존재:
  - `static/app.js:2068`
- 다운로드 라우트 부재(모듈형 앱 URL 맵 기준):
  - `app/routes/api_files.py` 내 download route 없음
- 동기화 스키마 불일치:
  - 프론트 기대: `static/app.js:2379`
  - 백엔드 반환: `app/routes/api_system.py:171`, `app/routes/api_system.py:175`
- 상태 진행률 키 불일치:
  - 프론트: `static/app.js:1411`
  - 백엔드: `app/routes/api_system.py:82`
- 리비전 키 불일치:
  - 프론트: `static/app.js:2938`, `static/app.js:3614`
  - 백엔드: `app/routes/api_files.py:373`
- 자동태그 키 불일치:
  - 프론트: `static/app.js:2910`
  - 백엔드: `app/routes/api_files.py:567`
- ZIP 업로드 이슈(현재 해결):
  - 호출 경로: `static/app.js`
  - 구현 상태: `POST /api/upload/folder` + 클라이언트 연동 완료
- XSS 위험 경로:
  - 하이라이트 생성: `app/routes/api_search.py:81`, `app/routes/api_search.py:82`
  - 렌더링: `static/app.js:2035`, `static/app.js:2045`
- 리비전 경로 주입 경로:
  - 입력 수신: `app/routes/api_files.py:375`, `app/routes/api_files.py:378`
  - 파일 경로 조합: `app/services/file_manager.py:31`, `app/services/file_manager.py:32`
- BM25-lite 경로 취약 지점:
  - `app/services/search.py:681`, `app/services/search.py:811`, `app/services/search.py:885`
- 업로드 확장자 불일치:
  - 백엔드 지원: `app/config.py:32`
  - 프론트 필터: `static/app.js:2435`
- PWA 경로 불일치:
  - `templates/index.html:28`, `static/manifest.json:12`, `static/sw.js:16`
- 태그 키 혼재:
  - basename 기준: `app/routes/api_files.py:77`, `app/routes/api_files.py:102`, `app/routes/api_files.py:321`
  - full path 사용: `app/routes/api_files.py:533`
- 권한 경계 후보:
  - `/api/upload`: `app/routes/api_files.py:221`
  - `/api/revisions` POST: `app/routes/api_files.py:375`
  - `/api/tags/set`: `app/routes/api_files.py:517`
  - `/api/tags/auto`: `app/routes/api_files.py:541`
  - `/api/cache/clear`: `app/routes/api_search.py:156`
- 기본 관리자 fallback:
  - `app/services/settings_store.py:134`~`app/services/settings_store.py:135`
- sanitize 정규식:
  - `app/utils.py:275`, `app/utils.py:276`

### 재현 절차(요약)
- 다운로드 단절:
  - 검색 결과 카드에서 “원본 파일” 클릭 -> 404/405 또는 비정상 응답.
- 동기화 상태 UI 오동작:
  - 관리자 페이지에서 동기화 시작/중지 후 인디케이터/버튼 상태 확인.
- ZIP 업로드 오류:
  - 관리자 ZIP 업로드 탭에서 파일 업로드 -> 콘솔 오류(`API.uploadFolder is not a function`) 확인.
- 자동태그 미반영:
  - 태그 모달에서 “자동” 클릭 후 응답 성공이어도 태그 칩 미갱신 확인.
- XSS:
  - 악성 HTML 포함 문서 인덱싱 후 검색 결과 렌더링 동작 확인.

---

## 참고
- 본 문서는 **코드 수정이 아닌 점검/설계 문서**다.
- 구현 시에는 하위호환(응답 alias 병행) 전략을 먼저 적용해 프론트/백 배포 타이밍 차이를 흡수하는 것을 권장한다.

---

## 9. 반영 결과 업데이트 (2026-02-20)

본 섹션은 초기 점검 문서의 원문을 유지한 상태에서, 이후 코드 반영 결과를 추적하기 위해 추가되었다.

### 완료된 항목 (요약)
1. 다운로드 라우트 복구: by-id/legacy 다운로드 경로 구현.
2. `/api/status`, `/api/sync/status`, `/api/revisions`, `/api/tags/auto` 스키마 alias 병행 제공.
3. ZIP 폴더 업로드 구현(`POST /api/upload/folder`) 및 클라이언트 API 연결.
4. 검색 결과 렌더링 XSS 방어: 클라이언트 escape 기반 하이라이트로 전환.
5. 리비전 저장 경로 안전화: 안전 파일명 + revisions 디렉토리 경계 검증.
6. `file_id` 도입 및 by-id 파일 API 확장.
7. 상태변경성 API 관리자 권한 강화(`admin_required`).
8. 관리자 비밀번호 fallback 제거(미설정 시 fail-closed).
9. BM25-lite 경로 안정화(모델 미로드 분기/캐시 경로 보강).
10. 프론트 업로드 확장자 필터 정합화(`xlsx/xls/hwp` 포함).
11. PWA 아이콘 경로 정합화(`static/icons/*` 추가).
12. sanitize 정규식 수정 및 관련 테스트 보강.

### 검증
- 자동 테스트: `pytest -q` 통과.
- 신규 회귀 테스트 파일:
  - `tests/test_api_compat.py`
  - `tests/test_file_identity.py`
  - `tests/test_upload_folder_zip.py`
  - `tests/test_revision_path_safety.py`
