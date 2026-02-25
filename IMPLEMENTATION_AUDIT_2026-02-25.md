# IMPLEMENTATION_AUDIT_2026-02-25.md

## 0. 구현 완료 현황 (2026-02-25, Post-Implementation)

> 본 문서의 1~8장은 초기 감사 시점의 리스크 진단 기록입니다.  
> 아래는 A-01~A-15 개선 구현 완료 후의 최신 상태 요약입니다.

### A-01 ~ A-15 반영 결과
- [x] A-01 CORS allowlist 적용 (`CORS_ALLOWED_ORIGINS`, credentials 제한)
- [x] A-02 세션 쿠키 정책 명시 적용 (`HttpOnly`/`SameSite`/`Secure`)
- [x] A-03 `/api/search/history` `success` 추가
- [x] A-04 `popular` 표준 포맷(`[{query,count}]`) + `popular_legacy` 병행
- [x] A-05 검색 하이라이트 상태(`lastRenderedQuery`) 렌더 경로 연결
- [x] A-06 관리자 테마 아이콘 로직 `ThemeManager.getTheme()` 기준 정합화
- [x] A-07 Toast `innerHTML` 제거, `textContent` 기반 렌더로 XSS 방어
- [x] A-08 Service Worker API 캐시 정책 재설계(GET allowlist만 캐시)
- [x] A-09 파일 삭제 기본 정책 `index_only` 전환(`delete_source=true` 옵션)
- [x] A-10 ZIP 제한 도입(`max_entries`, `max_uncompressed_bytes`, `max_single_file_bytes`)
- [x] A-11 `/api/sync/stop` 취소 이벤트 기반 구현
- [x] A-12 모델 변경 시 `reindex` 기본 `true` + 트리거 결과 응답
- [x] A-13 단일/일괄 청크 정책 통일(`AppConfig.CHUNK_SIZE/OVERLAP`)
- [x] A-14 `APP_ENV=production` + waitress 미설치 시 fail-fast
- [x] A-15 테스트 진입점 통일(`pytest.ini` + `pytest -q`)

### 최신 검증 결과
- 테스트: `pytest -q` 기준 **54 passed**
- 하위호환: API alias(`popular_legacy` 등) 병행 유지
- 문서/설정: `README.md`, `CLAUDE.md`, `GEMINI.md`, `pytest.ini` 최신 정책 반영

## 1. 개요/목적/범위

### 목적
- `README.md`, `CLAUDE.md`, 실제 코드/테스트를 교차 검증해 사내 규정 검색기의 잠재 리스크를 우선순위화한다.
- 백엔드/프론트엔드 공통으로 즉시 실행 가능한 개선 백로그를 제공한다.

### 범위
- 기준 문서:
  - `README.md`
  - `CLAUDE.md`
- 점검 코드:
  - 백엔드: `app/__init__.py`, `app/routes/api_search.py`, `app/routes/api_files.py`, `app/routes/api_system.py`, `app/services/search.py`
  - 프론트엔드: `static/app.js`, `static/sw.js`, `templates/index.html`, `templates/admin.html`
- 심각도 체계:
  - `Critical`, `High`, `Medium`, `Low`

### 비범위
- 코드 수정/리팩터링 실행
- 운영 배포 변경

---

## 2. 점검 방법 (정적 점검 + API 스모크 + 테스트 결과)

### 정적 점검
- 라우트/서비스/클라이언트 코드에서 보안, 스키마 정합성, 상태관리, 캐싱, 동시성 관련 패턴 점검.
- 주요 점검 명령:
  - `rg -n "..." app static templates`
  - `Get-Content ...`

### API 스모크
- Flask `test_client()` 기반 확인:
  - URL 맵 등록 검증
  - `/api/search/history`, `/api/search/suggest`, `/api/status` 응답 구조 확인
  - CORS/쿠키 보안 헤더 동작 확인

### 테스트 실행
- 실행: `pytest -q`
  - 결과: `54 passed`

### 요약
- 기능 테스트는 fresh shell 기준 `pytest -q` 단일 명령으로 통과한다.
- 그러나 보안/정합성/운영 안정성 관점의 설계 리스크는 다수 존재한다.

---

## 3. 핵심 결론 (Top 10 리스크)

1. `Critical` CORS 설정이 임의 Origin + credentials 조합으로 동작 가능 (`app/__init__.py:85` + 스모크 결과)
2. `High` 세션 쿠키 SameSite/Secure 설정 의도와 실제 적용 불일치 (`app/__init__.py:57`, `app/__init__.py:58`)
3. `High` 검색 히스토리/자동완성 API 응답 스키마와 프론트 기대값 불일치 (`api_search` vs `static/app.js`)
4. `High` 인기검색어 데이터 타입 불일치(튜플/배열 vs 객체)로 UI 렌더 실패 가능 (`app/services/search.py:549`, `static/app.js:1657`)
5. `High` 토스트 렌더링 비이스케이프 경로로 DOM XSS 위험 (`static/app.js:1358`, `static/app.js:1362`)
6. `High` Service Worker가 API POST/인증 응답까지 캐시 대상이 되는 구조 (`static/sw.js:63`, `static/sw.js:68`, `static/sw.js:116`)
7. `High` 단일 파일 삭제가 원본 물리 삭제를 기본 동작으로 수행 (`app/routes/api_files.py:222`)
8. `High` ZIP 업로드에 압축폭탄/총용량/엔트리 제한이 없음 (`app/routes/api_files.py:382`, `app/routes/api_files.py:383`, `app/routes/api_files.py:420`)
9. `High` 모델 변경 후 인덱스 재정합성(강제 재처리) 미보장 (`app/routes/api_system.py:51`, `app/routes/api_system.py:59`)
10. `High` waitress 미설치 시 debug 서버 fallback으로 운영 위험 (`run.py:121`, `run.py:124`)

---

## 4. 상세 이슈 목록 (백엔드/프론트엔드/공통)

| ID | 영역 | 심각도 | 현상 | 근거(파일:라인) | 영향 | 개선안 | 검증 방법 | 예상 공수 |
|---|---|---|---|---|---|---|---|---|
| A-01 | 공통(보안) | Critical | CORS가 credentials 허용 상태에서 Origin 제한 없이 동작 | `app/__init__.py:85` (`CORS(app, supports_credentials=True)`), 스모크 응답 `Access-Control-Allow-Origin: https://evil.example` | 세션 기반 API가 교차 출처 요청에 노출될 수 있음 | 허용 Origin allowlist 명시, 환경별 CORS 정책 분리(개발/운영) | 악성 Origin 헤더로 요청 시 ACAO 미반환 확인 | 0.5일 |
| A-02 | 백엔드(보안) | High | 세션 쿠키 SameSite/Secure 의도 미적용 | `app/__init__.py:57`, `app/__init__.py:58` (`setdefault`), 런타임 확인: `SESSION_COOKIE_SAMESITE=None`, `SESSION_COOKIE_SECURE=False` | CSRF/세션 탈취 완화 수준 저하 | `setdefault` 대신 명시 설정(`app.config[...] = ...`), HTTPS 환경에서 `Secure=True` | 로그인 후 Set-Cookie에 SameSite/Secure 반영 확인 | 0.5일 |
| A-03 | 백엔드/프론트 정합 | High | `/api/search/history`, `/api/search/suggest`가 `success` 필드 없이 응답 | `app/routes/api_search.py:132`, `app/routes/api_search.py:136`, `app/routes/api_search.py:146`, `app/routes/api_search.py:153`, `app/routes/api_search.py:156` vs `static/app.js:1649` | 프론트 분기(`if (result.success)`)에서 데이터 무시 가능 | 두 API에 `success: true` 추가(하위호환 유지) | 자동완성/히스토리 UI 정상 노출 확인 | 0.5일 |
| A-04 | 백엔드/프론트 정합 | High | 인기검색어 포맷 불일치(배열 vs 객체) | `app/services/search.py:549` (`List[Tuple[str,int]]`), `static/app.js:1657` (`p.query`, `p.count`) | 인기검색어 목록이 빈값 또는 undefined로 표시 | API 표준을 `[{query,count}]`로 고정, 프론트 파서 하위호환 처리 | 검색 히스토리 데이터 생성 후 인기검색어 뱃지 정상 렌더 확인 | 0.5일 |
| A-05 | 프론트 | Medium | 하이라이트 쿼리 상태가 렌더 경로와 미연결 (`lastRenderedQuery` 미세팅) | `static/app.js:1811`, `static/app.js:2117` | 검색어 하이라이트가 누락/비일관 | `renderSearchResults()`에서 `lastRenderedQuery=query` 설정 | 검색 후 결과 본문 `<mark>` 포함 여부 확인 | 0.2일 |
| A-06 | 프론트 | Medium | 관리자 테마 버튼 아이콘 상태 로직 오류 (`ThemeManager.currentTheme` 미정의) | `static/app.js:2211`, `static/app.js:2218` | 아이콘/상태 표시 불일치 | `ThemeManager.getTheme()` 사용으로 치환 | 관리자 페이지 테마 전환 시 아이콘 즉시 일치 확인 | 0.2일 |
| A-07 | 프론트(보안) | High | Toast 컴포넌트가 메시지 비이스케이프 `innerHTML` 렌더 | `static/app.js:1341`, `static/app.js:1358`, `static/app.js:1362` | 서버/사용자 유래 메시지로 DOM XSS 가능 | `Toast.show`에서 `textContent` 기반 렌더 또는 escape 적용 | `<img onerror>` 문자열 입력 시 실행되지 않고 텍스트로 표시 확인 | 0.5일 |
| A-08 | 프론트(PWA) | High | SW가 `/api/*` 요청을 메서드 구분 없이 `networkFirst` 처리 + 캐시 저장 | `static/sw.js:63`, `static/sw.js:68`, `static/sw.js:109`, `static/sw.js:116` | POST/인증 응답 캐시로 상태 불일치/보안 리스크 | API 캐시는 GET allowlist만 허용, 인증/관리/POST는 캐시 제외 | `/api/admin/auth` 등 POST가 Cache Storage에 남지 않는지 확인 | 0.7일 |
| A-09 | 백엔드(안정성) | High | 단일 파일 삭제 시 원본 파일 물리 삭제가 기본 동작 | `app/routes/api_files.py:222` | 업로드 외 경로(동기화 폴더) 문서도 삭제될 수 있어 데이터 손실 위험 | 기본 정책을 `index_only`, 물리 삭제는 명시 옵션으로 분리 | 삭제 API 호출 후 원본 보존/인덱스 제거 정책 검증 | 1.0일 |
| A-10 | 백엔드(보안/안정성) | High | ZIP 업로드에 엔트리 수/압축해제 용량/개별 파일 크기 제한 없음 | `app/routes/api_files.py:382`, `app/routes/api_files.py:383`, `app/routes/api_files.py:420` | ZIP bomb/자원 고갈로 서비스 장애 가능 | `max_entries`, `max_uncompressed_bytes`, `max_single_file_bytes` 제한 도입 | 제한 초과 ZIP 업로드 시 400/413 및 무처리 확인 | 1.0일 |
| A-11 | 백엔드 | Medium | `/sync/stop`가 TODO 상태였으나 현재 취소 이벤트 기반으로 구현 완료 | `app/routes/api_system.py`, `app/services/search.py` | 동기화 작업 제어 가능 상태로 개선 | 취소 토큰/작업 상태기계 도입 및 API 연결 완료 | 대용량 동기화 중 중단 요청 시 진행 멈춤 확인 | 1.0일 |
| A-12 | 백엔드(검색품질) | High | 모델 변경 후 기존 벡터 인덱스와 임베딩 정합성 보장 없음 | `app/routes/api_system.py:51`, `app/routes/api_system.py:59` | 검색 결과 품질 저하/비결정성 | 모델 변경 시 강제 재인덱스 옵션 기본 활성, 상태 배지/작업 큐 연계 | 모델 교체 전후 동일 질의 품질 회귀 테스트 | 1.0일 |
| A-13 | 백엔드(일관성) | Medium | 단일 업로드와 일괄 처리의 청크 정책 불일치 | `app/services/search.py:714`(500/50), `app/services/search.py:912`(AppConfig) | 동일 문서라도 유입 경로에 따라 검색 품질 편차 | `process_single_file`도 `AppConfig.CHUNK_SIZE/OVERLAP` 사용 | 동일 문서 단일/일괄 업로드 결과 비교 | 0.5일 |
| A-14 | 운영/배포 | High | waitress 미설치 시 `debug=True` 내장 서버로 fallback | `run.py:121`, `run.py:124` | 운영환경에서 디버그 노출/성능 저하/안정성 리스크 | 운영모드에서는 시작 실패(fail-fast), 개발모드만 fallback | waitress 제거 후 운영 시작 시 명시적 실패 확인 | 0.5일 |
| A-15 | 테스트/개발경험 | Medium | 테스트 실행 진입점 불일치 이슈는 `pytest.ini` 도입으로 해결됨 | `pytest.ini`, `README.md` | CI/로컬 실행 방식 일치 | `pythonpath = .` 고정, 실행 명령 `pytest -q` 단일화 완료 | fresh shell에서 단일 명령으로 54 passing 확인 | 0.5일 |

---

## 5. 추가 구현 제안 (기능 확장 백로그)

### 5.1 API 계약 안정화
- OpenAPI(또는 JSON Schema) 기반 응답 계약 관리
- 핵심 엔드포인트 계약 테스트 자동화

### 5.2 운영 가시성
- `request_id`/`trace_id`를 API 응답 및 로그에 통일
- 슬로우쿼리/슬로우리퀘스트 대시보드화

### 5.3 보안 강화
- CSRF 방어(세션 기반 변경 API)
- 관리자 인증 실패 횟수 제한 및 지연 정책
- 업로드 파일 악성 패턴(매크로/PDF exploit) 사전 검사 훅

### 5.4 검색 품질 관리
- 모델/청킹/가중치 실험을 위한 오프라인 평가셋 구축
- 품질 지표(nDCG, Recall@k) 기반 회귀 테스트

---

## 6. 단계별 실행 로드맵

### P0 (즉시, 1~2주)
- A-01 CORS 제한
- A-02 쿠키 보안 속성 명시 적용
- A-07 Toast XSS 차단
- A-08 SW API 캐시 정책 재설계
- A-09 파일 삭제 정책 분리
- A-10 ZIP 제한 도입
- A-14 운영 fallback fail-fast

### P1 (단기, 2~4주)
- A-03, A-04 API 응답 스키마 표준화 + alias 유지
- A-11 sync stop 구현
- A-12 모델 변경 시 재인덱스 정책 자동화
- A-13 청크 정책 통일
- A-15 테스트 실행 기준 통일

### P2 (중기, 1~2개월)
- API 계약 자동 검증 파이프라인
- 검색 품질 회귀 프레임워크
- 운영 지표/알람 체계 고도화

---

## 7. 회귀 테스트 체크리스트

### 백엔드 단위/통합
- [x] `pytest -q` 통과
- [x] `/api/search/history` 응답 스키마 계약(`success`, `recent`, `popular[{query,count}]`, `popular_legacy`)
- [x] `/api/search/suggest` 응답 스키마 계약(`success`, `suggestions`)
- [x] `/api/status`, `/api/health` envelope 일관성 확인
- [x] 쿠키 보안 헤더(`SameSite`, `Secure`, `HttpOnly`) 검증
- [x] ZIP 제한 초과/우회 시나리오 차단 검증

### 프론트엔드 기능
- [x] 자동완성(최근/인기/제안) 정상 노출
- [x] 하이라이트 `<mark>` 정상 동작
- [x] 관리자 테마 토글 아이콘 상태 일치
- [x] Toast XSS 방어 동작 확인

### 서비스워커
- [x] POST 요청 미캐시
- [x] 인증 상태 응답 미캐시
- [x] 오프라인/온라인 전환 시 API 일관성 유지

### E2E 스모크
- [x] 검색 → 필터링 → 다운로드
- [x] 관리자 인증 → 업로드(파일/ZIP) → 재처리 → 버전비교

---

## 8. 부록 (실행 커맨드, 확인 전제)

### 8.1 실행 커맨드
- 파일/코드 탐색:
  - `rg --files app static templates tests`
  - `rg -n "..." app static templates`
- 테스트:
  - `pytest -q` (54 passed)
- 스모크 확인:
  - CORS 헤더 확인용 `test_client()` 요청
  - 세션 쿠키 설정값 확인
  - URL 맵 출력(`app.url_map.iter_rules()`)

### 8.2 확인 전제/가정
- 보고서 작성 기준 시점: `2026-02-25`
- 코드 변경은 수행하지 않았고 분석 결과만 정리함
- 라인 번호는 해당 시점 워킹트리 기준
- 하위호환을 유지하는 점진적 개선(alias 유지)을 기본 정책으로 가정

---

## 공개 API/인터페이스 변경 제안 (명시)

1. `/api/search/history`
- 변경: `success` 필드 추가, `popular`를 `[{query,count}]`로 표준화
- 호환: 기존 배열 포맷은 일정 기간 alias 지원

2. `/api/search/suggest`
- 변경: `success` 필드 추가
- 호환: 기존 `suggestions` 필드 유지

3. `/api/status`, `/api/health`
- 변경: 성공 응답 envelope 일관화
- 호환: 기존 필드 alias 유지

4. 파일 삭제 API
- 변경: 삭제 정책 명시 (`index_only` 기본, `delete_source` 옵션)
- 호환: 기존 호출은 `index_only`로 매핑

5. ZIP 업로드 API
- 변경: 제한 필드 도입
  - `max_entries`
  - `max_uncompressed_bytes`
  - `max_single_file_bytes`

6. Service Worker 캐시 인터페이스
- 변경: API는 GET allowlist만 캐시
- 인증/관리/POST 요청은 캐시 제외
