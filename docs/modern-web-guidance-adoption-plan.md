# Modern Web Guidance 전면 반영 계획서

> 대상: 사내 규정 검색기 v3.0 웹 프론트엔드 (`static/`, `templates/`)
> 도구: `modern-web-guidance` (skill-version 2026_11_10-47b4c80e)
> 작성일: 2026-11-22

## 1. 스캔 방법

광범위 모더니제이션 요청이므로 스킬의 `legacy-scan` 플로우를 사용했다.

```powershell
npx.cmd -y modern-web-guidance@latest legacy-scan static --json
npx.cmd -y modern-web-guidance@latest legacy-scan templates --json
npx.cmd -y modern-web-guidance@latest legacy-scan "static/js" --json
npx.cmd -y modern-web-guidance@latest retrieve "customize-scrollbar-color-and-thickness,dialog,dynamic-sibling-animations,js"
```

참고: `legacy-scan`은 결정적(deterministic) 레거시 규칙만 검사하므로 가이드 전체(약 20% 커버리지)가
아니다. 나머지 주제는 `search <topic>`으로 개별 조회한다.

## 2. 스캔 결과 요약

| # | guideId | 건수 | 위치 |
|---|---------|------|------|
| 1 | `customize-scrollbar-color-and-thickness` | 3 | `static/style.css` 36, 45, 50 |
| 2 | `dialog` | 1 | `static/js/legacy/app.js` 3834 |
| 3 | `dynamic-sibling-animations` | 1 | `static/style.css` 2254 (실제 10개 규칙, 2254~2292) |
| 4 | `js` (`replaceChildren-innerhtml`) | 7 | `app.js` 1541·2135, `chat.js` 72·100·121·123, `citations.js` 9 |

`templates/`는 finding 0건이다.

## 3. 검증 결과 (keepWhen / dropWhen 대조)

### 3.1 KEEP — 스크롤바 (3건)

- 현황: `::-webkit-scrollbar*` 벤더 슈도엘리먼트만 사용, 표준 `scrollbar-color`/`scrollbar-width` 없음
  (`static/` 전체 검색 결과 0건).
- 가이드 의무사항: 스크롤 가능한 컨테이너에 표준 속성 우선 적용, thumb/track 색상은
  CSS 변수로 분리, thumb 대비율 3:1 이상(WCAG 1.4.11).
- 기존 색상(`--accent-secondary` × `--bg-secondary`) 실측 대비율은 **1.49:1로 미달**이므로
  색상도 함께 교정한다(다크 thumb `#5b6b82` 3.43:1, 라이트 thumb `#64748b` 4.77:1,
  다크 hover `#ff6b8a` 6.85:1, 라이트 hover `#e11d48` 4.75:1 — 모두 3:1 이상 실측).
- WebKit 폴백은 `@supports not (scrollbar-color: auto)` 블록 안에 유지하고,
  같은 셀렉터에 `scrollbar-width: auto` 리셋을 포함한다(가이드 폴백 전략).

### 3.2 DROP — dialog (1건, 수정 없음)

- 해당 리스너(`VersionManager.init`, `app.js` 3834)는 네이티브 `<dialog>`가 아니라
  `div.modal-overlay`에 붙은 오버레이 클릭 닫기다(`templates/index.html` 383,
  `templates/admin.html` 291).
- `closedby="any"`는 `<dialog>` 전용 선언형dismiss이므로 div 오버레이에는 적용 불가.
  `e.target === this.modal` 검사는 div 모달의 정석 패턴이며 안전 코드를 건드리지 않는다.
- 네이티브 `<dialog>` 전환은 마크업·포커스·Esc·폼 동작까지 바꾸는 별도 작업이므로
  본 계획의 범위로 삼지 않고 후속 과제로 기록한다.

### 3.3 KEEP — stagger 애니메이션 (10개 규칙)

- `.result-card:nth-child(1~10)`이 `animation-delay`만 단계적으로 지정한다.
  keepWhen(3개 이상 연속 + delay 전용)에 정확히 해당한다.
- `sibling-index()` 단일 규칙으로 교체한다. 기존 오프셋(0s 시작, 0.05s 간격)을 보존하기 위해
  `calc((sibling-index() - 1) * var(--stagger-time))`을 사용한다.
- 정적/서버 렌더 CSS 권장 구조를 따른다: 기본 규칙 + `@supports (...sibling-index...)` 승격 +
  `@supports not (...)` 안에 기존 `:nth-child` 폴백 유지(높은 명시도가 표준 지원을 덮지 않도록).
- `@supports` 조건에는 리터럴 시간값을 사용한다(`var()` 참조 시 항상 true가 되는 파싱 함정 회피).
- `prefers-reduced-motion: reduce`에서 `.result-card` 애니메이션을 끄는 기존 대응(2297~2331)은
  가이드 의무사항을 이미 만족하므로 유지한다.
- 부수 효과: 기존 CSS는 10번째 이후 카드에 delay가 없었으나 `sibling-index()`는 동적 개수 전체에
  자동 적용된다(런타임에는 `StaggerAnimation.apply()` 인라인 delay가 우선하므로 동작 불변).

### 3.4 KEEP — `replaceChildren()` (7건)

- 7건 모두 일반 `Element`의 독립적인 `innerHTML = ''` 비우기이며, `<template>`·DocumentFragment·
  연쇄 대입 등 dropWhen에 해당하지 않는다.
- `container.innerHTML = ''; container.appendChild(fragment);` (`app.js` 2135~2136)는
  가이드가 권장하는 원자적 형태 `container.replaceChildren(fragment)`로 합친다.
  나머지는 비우기 호출만 `replaceChildren()`으로 교체한다(파서 미기동, Trusted Types CSP 대응,
  단일 MutationObserver 배치).

## 4. 구현 체크리스트

- [x] `static/style.css` 스크롤바: 표준 속성 + 테마별 변수 + WebKit 폴백 블록
- [x] `static/style.css` stagger: `sibling-index()` 승격 + `:nth-child` 폴백 격리
- [x] `static/js/legacy/app.js` 2건 (`createDropdown`, `renderResults`)
- [x] `static/js/rag/chat.js` 4건 (`loadConversations`, `loadConversation`, `startNewConversation`)
- [x] `static/js/rag/citations.js` 1건 (`renderList`)
- [x] dialog finding: DROP 판정, 코드 변경 없음

## 5. 검증

- `python -m pytest -q` (레포 자체 테스트, JS 미영향 확인)
- `python -m py_compile` 대상 아님(CSS/JS) — 변경 파일 육안 대조 + `legacy-scan` 재실행으로 finding 해소 확인
- 수동 확인 필요: 라이트/다크 테마 스크롤바 색상, 검색 결과 stagger, RAG 대화 목록/메시지/인용 렌더링

## 6. 2차 반영 (2026-11-22) — 카탈로그 전수 조사

`list`로 전체 카탈로그를 조회하고, 본 앱과 직접 관련 있는 4종을 추가 retrieve·검증했다.
(`ime-safe-enter-submit`, `dark-mode`, `highlight-text-ranges`, `prevent-page-scroll-under-modal-dialog`)

### 6.1 KEEP — IME Enter 오제출 방지 (4곳)

- 현황: 검색창·자동완성·관리자 비밀번호·태그 입력의 커스텀 Enter 핸들러가
  `isComposing` 검사를 하지 않아 한글 IME 확정키에 오제출된다.
  (RAG 컴포저의 Ctrl/Cmd+Enter는 가이드상 면제라 손대지 않는다.)
- 가이드 Strategy 1을 적용: `Enter` 게이트 안에 `isComposing || keyCode === 229` 검사를 둔다.
  ("Pairing `event.isComposing` with `event.keyCode === 229` reliably catches Safari's
  out-of-order confirming `Enter` keydown.")
- 명시적 검색 실행 버튼(`#search-btn`)과 입력 `aria-label`은 이미 있어 추가 작업 없음.

### 6.2 KEEP — 다크모드 네이티브 UI 동기화

- `<meta name="color-scheme" content="light dark">`을 3개 템플릿 head에 추가했다.
  ("place a `<meta>` tag in your `<head>` to ensure the browser knows which themes you
  support before it even starts rendering")
- `:root`에 `color-scheme: light dark; accent-color: var(--accent-primary)`를 선언했다.
  ("Apply the `color-scheme` property to the `html` element or the `:root` pseudo-class")
- 사이트 토글이 시스템값과 다를 수 있으므로 `setTheme()`에서
  `document.documentElement.style.colorScheme = theme`으로 네이티브 UI를 동기화한다.
- 기존 `matchMedia('(prefers-color-scheme: dark)')` + `change` 핸들러는 가이드 요구를
  이미 만족하므로 유지한다.

### 6.3 KEEP — highlightSafeText 복원 (버그 수정)

- 조사 중 `createResultCard`가 호출하는 `highlightSafeText`의 정의가 레포 어디에도 없고
  v3.0 리팩터 과정에서 누락된 것을 확인했다(호출 1건·정의 0건, 히스토리 대조 완료).
  검색 렌더링 진입 시 ReferenceError가 나는 상태였으므로 원본 로직을 복원했다.
- `highlight-text-ranges` 가이드 관점: 검색 결과 하이라이트는 Custom Highlight API의
  대표 용례이나, 가이드가 `<mark>` 래핑을 공식 폴백으로 인정하므로
  ("If the highlight is critical for the user experience, fall back to wrapping matched
  text in `<mark>` elements") 현재 escape-후-`<mark>` 방식을 유지한다.
  전체 Custom Highlight API 전환은 후속 과제로 남긴다.

### 6.4 부분 적용 — 모달 스크롤 체이닝 격리

- `prevent-page-scroll-under-modal-dialog`는 네이티브 `<dialog>` 전용 레시피라
  div 모달의 `body overflow` 잠금(`ReaderMode`)을 교체하지는 않는다.
- 다만 가이드의 체이닝 차단 취지에 따라 `.modal-overlay`에
  `overscroll-behavior: contain`을 progressive enhancement로 추가했다.

### 2차 구현 체크리스트

- [x] `highlightSafeText` 복원 (`static/js/legacy/app.js`, `export function`)
- [x] IME 가드 4곳 (검색 keypress, 자동완성 keydown, 관리자 비밀번호, 태그 입력)
- [x] `color-scheme` 메타 3종 + `:root color-scheme/accent-color` + `setTheme` 동기화
- [x] `.modal-overlay`에 `overscroll-behavior: contain`

### 2차 검증

- `node <추출테스트>`: 실제 파일에서 추출한 함수로 mark/escape/2자 필터 확인 → OK
- `node --check` 3파일, `python -m pytest -q` → 126 passed
- `legacy-scan` 재실행으로 finding 해소 확인

## 7. 남은 후속 과제

- `div.modal-overlay` → 네이티브 `<dialog>` + `closedby="any"` 전환 (가이드 `dialog` 전문 적용)
- 검색 하이라이트의 CSS Custom Highlight API 전환 검토 (현 `<mark>` 방식이 공식 폴백이므로 선택 사항)
- `innerHTML = <HTML문자열>` 대입부(자동완성 옵션, 스켈레톤, 카드 템플릿 등)에
  가이드 `sanitize-untrusted-html` 조회 후 살균 경로 점검 (현재는 `escapeHtml` 일관 적용 중)
- `search` 카탈로그의 접근성(`accessibility`)·성능(`performance`)·색상(`color`) 주제별 추가 감사

