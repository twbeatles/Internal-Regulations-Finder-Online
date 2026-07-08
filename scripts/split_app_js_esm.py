# -*- coding: utf-8 -*-
"""app.js → static/js ESM 모듈 분할 (1회성)."""
from __future__ import annotations

import os
import re
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "static", "app.js")
OUT = os.path.join(ROOT, "static", "js")

# (파일명, 섹션 시작 키워드, 섹션 끝 키워드 또는 None=다음 섹션까지)
SECTIONS = [
    ("core/logger.js", "로깅 유틸리티", "UX 유틸리티"),
    ("ui/ripple.js", "UX 유틸리티 - 리플", "공통 유틸리티 함수"),
    ("ui/toast-legacy.js", "토스트 알림 \\(레거시", "테마 관리"),
    ("core/performance.js", "성능 유틸리티", "사용자 설정"),
    ("core/prefs.js", "사용자 설정", "스켈레톤 로딩"),
    ("ui/skeleton.js", "스켈레톤 로딩", "네트워크 상태"),
    ("core/network.js", "네트워크 상태", "키보드 단축키"),
    ("search/keyboard.js", "키보드 단축키", "읽기 모드"),
    ("search/reader-mode.js", "읽기 모드", "하이라이트 탐색"),
    ("search/highlight-nav.js", "하이라이트 탐색", "검색 결과 탐색"),
    ("search/result-nav.js", "검색 결과 탐색", "순차 애니메이션"),
    ("ui/stagger.js", "순차 애니메이션", "결과보내기"),
    ("search/export-results.js", "결과보내기", "API 클라이언트"),
    ("services/api.js", "API 클라이언트", "북마크 관리"),
    ("search/bookmarks.js", "북마크 관리", "토스트 알림"),
    ("ui/toast.js", "토스트 알림", "테마 관리"),
    ("ui/theme.js", "테마 관리", "앱 상태 관리"),
    ("core/app-state.js", "앱 상태 관리", "자동완성 시스템"),
    ("search/autocomplete.js", "자동완성 시스템", "파일 목록 로드"),
    ("search/file-filter.js", "파일 목록 로드", "북마크 이벤트"),
    ("search/bookmark-events.js", "북마크 이벤트", "검색 페이지"),
    ("search/search-page.js", "검색 페이지", "관리자 폴링"),
    ("admin/polling.js", "관리자 폴링", "관리자 페이지"),
    ("admin/admin-page.js", "관리자 페이지", "PDF보내기"),
    ("search/pdf-export.js", "PDF보내기", "고급 검색 컨트롤러"),
    ("search/advanced-search.js", "고급 검색 컨트롤러", "버전 관리 매니저"),
    ("admin/version-manager.js", "버전 관리 매니저", "초기화"),
]


def read_src() -> str:
    with open(SRC, "r", encoding="utf-8") as f:
        return f.read()


def find_section(text: str, start_kw: str, end_kw: str | None) -> str | None:
    pattern = rf"// =+\n// {re.escape(start_kw)}"
    m = re.search(pattern, text)
    if not m:
        return None
    start = m.start()
    if end_kw:
        end_pat = rf"// =+\n// {re.escape(end_kw)}"
        m2 = re.search(end_pat, text[m.end():])
        if not m2:
            return text[start:]
        return text[start : m.end() + m2.start()]
    return text[start:]


def add_exports(chunk: str) -> str:
    lines = chunk.splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("const ") and "=" in stripped:
            name = stripped.split("const ", 1)[1].split("=", 1)[0].strip()
            if name and name[0].isalpha():
                line = line.replace(f"const {name}", f"export const {name}", 1)
        elif stripped.startswith("async function "):
            name = stripped.split("async function ", 1)[1].split("(", 1)[0].strip()
            line = line.replace(f"async function {name}", f"export async function {name}", 1)
        elif stripped.startswith("function ") and not stripped.startswith("function ("):
            name = stripped.split("function ", 1)[1].split("(", 1)[0].strip()
            if name:
                line = line.replace(f"function {name}", f"export function {name}", 1)
        out.append(line)
    return "\n".join(out) + "\n"


def main() -> None:
    text = read_src()
    os.makedirs(OUT, exist_ok=True)

    # 공통 유틸은 기존 core/utils.js 사용 — app.js 해당 구간은 search-page에서 import
    created: list[str] = []
    for rel, start_kw, end_kw in SECTIONS:
        chunk = find_section(text, start_kw, end_kw)
        if not chunk:
            print(f"SKIP (not found): {rel} <- {start_kw}")
            continue
        body = add_exports(chunk)
        header = f"/** Auto-split from app.js — {start_kw} */\n"
        if "escapeHtml" in body or "formatFileSize" in body:
            header += "import { escapeHtml, escapeRegExp, formatFileSize } from '../core/utils.js';\n"
        path = os.path.join(OUT, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(header + body)
        created.append(rel)
        print(f"Wrote {rel}")

    bootstrap = textwrap.dedent(
        '''\
        /**
         * 메인 페이지 ESM 부트스트랩 (app.js 분할 모듈)
         */
        import { escapeHtml, escapeRegExp, formatFileSize } from '../core/utils.js';
        import { RagStream } from '../rag/stream.js';
        import { CitationRenderer } from '../rag/citations.js';
        import { ModeToggle } from '../rag/mode-toggle.js';
        import { ChatComposer } from '../rag/composer.js';
        import { ChatManager } from '../rag/chat.js';

        // 검색·UI 모듈 (동적 import로 레거시 의존 순서 보장)
        const modules = await Promise.all([
            import('../core/logger.js'),
            import('../ui/ripple.js'),
            import('../core/performance.js'),
            import('../core/prefs.js'),
            import('../ui/skeleton.js'),
            import('../core/network.js'),
            import('../services/api.js'),
            import('../ui/toast.js'),
            import('../ui/theme.js'),
            import('../core/app-state.js'),
            import('../search/autocomplete.js'),
            import('../search/file-filter.js'),
            import('../search/bookmark-events.js'),
            import('../search/advanced-search.js'),
            import('../search/search-page.js'),
            import('../search/keyboard.js'),
            import('../search/reader-mode.js'),
            import('../search/highlight-nav.js'),
            import('../search/result-nav.js'),
            import('../search/export-results.js'),
            import('../search/bookmarks.js'),
            import('../search/bookmark-events.js'),
            import('../search/pdf-export.js'),
            import('../admin/version-manager.js'),
        ]);

        const flat = Object.assign({}, ...modules.map((m) => m));
        Object.assign(window, {
            escapeHtml,
            escapeRegExp,
            formatFileSize,
            RagStream,
            CitationRenderer,
            ModeToggle,
            ChatComposer,
            ChatManager,
            ...flat,
        });

        function bootstrapRag() {
            if (document.getElementById('rag-section')) ChatManager.init();
        }

        function bootstrapSearch() {
            if (document.querySelector('.search-section')) {
                flat.BookmarkPanel?.init?.();
                flat.AdvancedSearch?.init?.();
                flat.VersionManager?.init?.();
                flat.initSearch?.();
            }
        }

        function bootstrapAdmin() {
            if (document.querySelector('.admin-page') || document.getElementById('files-tbody')) {
                flat.initAdmin?.();
                flat.VersionManager?.init?.();
            }
        }

        function bootstrap() {
            const yearEl = document.getElementById('current-year');
            if (yearEl) yearEl.textContent = new Date().getFullYear();
            bootstrapRag();
            bootstrapSearch();
            bootstrapAdmin();
        }

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', bootstrap);
        } else {
            bootstrap();
        }
        '''
    )
    with open(os.path.join(OUT, "bootstrap", "main.js"), "w", encoding="utf-8", newline="\n") as f:
        f.write(bootstrap)

    # app.js를 얇은 deprecated shim으로 교체
    shim = textwrap.dedent(
        '''\
        /**
         * @deprecated static/js/bootstrap/main.js (ESM) 사용. 하위호환용 빈 shim.
         */
        console.warn('[RegSearch] static/app.js는 deprecated입니다. ESM bootstrap을 사용하세요.');
        '''
    )
    backup = SRC + ".bak"
    if not os.path.exists(backup):
        os.replace(SRC, backup)
    with open(SRC, "w", encoding="utf-8", newline="\n") as f:
        f.write(shim)

    print(f"Created {len(created)} modules; app.js -> shim, backup at app.js.bak")


if __name__ == "__main__":
    main()