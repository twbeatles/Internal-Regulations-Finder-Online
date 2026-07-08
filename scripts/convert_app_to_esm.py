# -*- coding: utf-8 -*-
"""app.js → static/js/legacy/app.js (ESM) + UI 패치."""
from __future__ import annotations

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "static", "app.js")
DST = os.path.join(ROOT, "static", "js", "legacy", "app.js")
BACKUP = os.path.join(ROOT, "static", "app.js.bak")


def read_src() -> str:
    if os.path.exists(BACKUP):
        with open(BACKUP, "r", encoding="utf-8") as f:
            return f.read()
    with open(SRC, "r", encoding="utf-8") as f:
        return f.read()


def patch_ui(text: str) -> str:
    # performSearch: AdvancedSearch 연동 + 취소 버튼
    old_perform = """async function performSearch() {
    const input = document.getElementById('search-input');
    const resultsContainer = document.getElementById('results-container');
    const resultCount = document.getElementById('result-count');
    const hybridCheck = document.getElementById('hybrid-search');
    const filterFile = document.getElementById('filter-file');
    const sortBy = document.getElementById('sort-by');

    // 필수 요소 존재 확인
    if (!input || !resultsContainer) {
        console.error('Required DOM elements not found');
        return;
    }

    const query = input.value.trim();
    if (!query) {
        Toast.warning('검색어 필요', '검색어를 입력해주세요');
        input.focus();
        return;
    }

    if (query.length < 2) {
        Toast.warning('검색어 짧음', '최소 2자 이상 입력해주세요');
        return;
    }

    // 네트워크 상태 확인
    if (!NetworkStatus.checkConnection()) {
        Toast.error('오프라인', '네트워크 연결을 확인해주세요');
        return;
    }

    // 스켈레톤 로딩 표시
    const k = parseInt(resultCount?.value || 5);
    resultsContainer.innerHTML = SkeletonLoading.createSearchSkeleton(k);

    const hybrid = hybridCheck?.checked !== false;
    const filterId = filterFile?.value || null;
    const sort = sortBy?.value || 'relevance';

    const result = await API.search(query, k, hybrid, true, filterId, sort, null);

    if (!result.success) {
        resultsContainer.innerHTML = `
            <div class="no-results">
                <div class="no-results-icon">😕</div>
                <h3>검색 실패</h3>
                <p>${escapeHtml(result.message)}</p>
                <button class="btn btn-primary" onclick="performSearch()" style="margin-top: 16px">
                    🔄 다시 시도
                </button>
            </div>
        `;
        Toast.error('검색 실패', result.message);
        return;
    }

    if (!result.results || result.results.length === 0) {
        resultsContainer.innerHTML = `
            <div class="no-results">
                <div class="no-results-icon">🔍</div>
                <h3>결과 없음</h3>
                <p>"${query}"에 대한 검색 결과가 없습니다</p>
            </div>
        `;
        return;
    }

    // 결과 표시
    renderSearchResults(result.results, query);
    input.value = '';
    input.focus();
}"""

    new_perform = """async function performSearch(options = {}) {
    const { preserveQuery = false } = options;
    const input = document.getElementById('search-input');
    const resultsContainer = document.getElementById('results-container');
    const resultCount = document.getElementById('result-count');
    const hybridCheck = document.getElementById('hybrid-search');
    const filterFile = document.getElementById('filter-file');
    const sortBy = document.getElementById('sort-by');
    const searchBtn = document.getElementById('search-btn');
    const cancelBtn = document.getElementById('cancel-search-btn');

    if (!input || !resultsContainer) {
        console.error('Required DOM elements not found');
        return;
    }

    const query = input.value.trim();
    if (!query) {
        Toast.warning('검색어 필요', '검색어를 입력해주세요');
        input.focus();
        return;
    }

    if (query.length < 2) {
        Toast.warning('검색어 짧음', '최소 2자 이상 입력해주세요');
        return;
    }

    if (!NetworkStatus.checkConnection()) {
        Toast.error('오프라인', '네트워크 연결을 확인해주세요');
        return;
    }

    const parsed = AdvancedSearch.parseQuery(query);
    const serverQuery = AdvancedSearch.isActive(query, parsed)
        ? AdvancedSearch.deriveServerQuery(parsed)
        : query;
    const fetchK = AdvancedSearch.isActive(query, parsed)
        ? Math.min(50, Math.max(parseInt(resultCount?.value || 5, 10) * 4, 20))
        : parseInt(resultCount?.value || 5, 10);

    resultsContainer.innerHTML = SkeletonLoading.createSearchSkeleton(fetchK);
    const hybrid = hybridCheck?.checked !== false;
    const filterId = filterFile?.value || null;
    const sort = sortBy?.value || 'relevance';

    if (cancelBtn) cancelBtn.hidden = false;
    if (searchBtn) searchBtn.disabled = true;

    const result = await API.search(serverQuery, fetchK, hybrid, true, filterId, sort, null);

    if (cancelBtn) cancelBtn.hidden = true;
    if (searchBtn) searchBtn.disabled = false;

    if (result.aborted) return;

    if (!result.success) {
        resultsContainer.innerHTML = `
            <div class="no-results">
                <div class="no-results-icon">😕</div>
                <h3>검색 실패</h3>
                <p>${escapeHtml(result.message)}</p>
                <button class="btn btn-primary" onclick="performSearch()" style="margin-top: 16px">
                    🔄 다시 시도
                </button>
            </div>
        `;
        Toast.error('검색 실패', result.message);
        return;
    }

    let results = Array.isArray(result.results) ? result.results : [];
    if (AdvancedSearch.isActive(query, parsed)) {
        results = AdvancedSearch.filterResults(results, parsed);
        const displayK = parseInt(resultCount?.value || 5, 10);
        results = results.slice(0, displayK);
    }

    if (!results.length) {
        resultsContainer.innerHTML = `
            <div class="no-results">
                <div class="no-results-icon">🔍</div>
                <h3>결과 없음</h3>
                <p>"${escapeHtml(query)}"에 대한 검색 결과가 없습니다</p>
            </div>
        `;
        return;
    }

    renderSearchResults(results, query);
    if (!preserveQuery) {
        input.value = '';
        const clearBtn = document.getElementById('clear-query-btn');
        if (clearBtn) clearBtn.hidden = true;
    }
    input.focus();
}"""

    if old_perform not in text:
        raise RuntimeError("performSearch block not found")
    text = text.replace(old_perform, new_perform)

    # initSearch: clear / instant / cancel / filters
    marker = "    // 이벤트 위임 설정 (북마크, 복사, 접기/펼치기)\n    setupBookmarkEventDelegation();\n}"
    injection = """    // 검색 UI 컨트롤 (지우기·실시간·취소·필터)
    const clearBtn = document.getElementById('clear-query-btn');
    const cancelBtn = document.getElementById('cancel-search-btn');
    const instantCheck = document.getElementById('instant-search');
    const filtersToggle = document.getElementById('filters-toggle');
    const searchOptions = document.getElementById('search-options');

    const updateClearBtn = () => {
        if (clearBtn && searchInput) clearBtn.hidden = !searchInput.value.length;
    };

    const runInstantSearch = PerformanceUtils.debounce(() => {
        if (!instantCheck?.checked || !searchInput) return;
        const q = searchInput.value.trim();
        if (q.length >= 2) performSearch({ preserveQuery: true });
    }, 450);

    if (searchInput) {
        searchInput.addEventListener('input', () => {
            updateClearBtn();
            if (searchBtn && searchInput.value.trim().length >= 2) searchBtn.disabled = false;
            runInstantSearch();
        });
        updateClearBtn();
    }

    if (clearBtn && searchInput) {
        clearBtn.addEventListener('click', () => {
            searchInput.value = '';
            updateClearBtn();
            searchInput.focus();
            Autocomplete.hide();
        });
    }

    if (cancelBtn) {
        cancelBtn.addEventListener('click', () => {
            API.abort('search');
            cancelBtn.hidden = true;
            if (searchBtn) searchBtn.disabled = false;
        });
    }

    if (filtersToggle && searchOptions) {
        filtersToggle.addEventListener('click', () => {
            const collapsed = searchOptions.dataset.collapsed !== 'false';
            searchOptions.dataset.collapsed = collapsed ? 'false' : 'true';
            filtersToggle.setAttribute('aria-expanded', collapsed ? 'true' : 'false');
        });
    }

    // 이벤트 위임 설정 (북마크, 복사, 접기/펼치기)
    setupBookmarkEventDelegation();
}"""

    if marker not in text:
        raise RuntimeError("initSearch marker not found")
    text = text.replace(marker, injection)
    return text


def strip_duplicate_utils(text: str) -> str:
    start = text.find("// ============================================================================\n// 공통 유틸리티 함수")
    end = text.find("// ============================================================================\n// 성능 유틸리티")
    if start == -1 or end == -1:
        return text
    return text[:start] + text[end:]


def add_esm_header(text: str) -> str:
    header = (
        "/** Legacy app bundle (ESM) — converted from static/app.js */\n"
        "import { escapeHtml, escapeRegExp, formatFileSize } from '../core/utils.js';\n\n"
    )
    return header + text


def add_exports(text: str) -> str:
    exports: list[str] = []

    def repl_const(m: re.Match) -> str:
        name = m.group(1)
        if name not in exports:
            exports.append(name)
        return f"export const {name} ="

    text = re.sub(r"(?m)^const ([A-Z][A-Za-z0-9_]*) =", repl_const, text)
    text = re.sub(r"(?m)^async function ([a-zA-Z_][a-zA-Z0-9_]*)", r"export async function \1", text)
    text = re.sub(r"(?m)^function ([a-zA-Z_][a-zA-z0-9_]*)", r"export function \1", text)

    # highlightSafeText 등 소문자 함수
    for fn in (
        "highlightSafeText", "showToast", "createToastContainer", "getToastIcon",
        "loadFileListForFilter", "toggleBookmarkByIndex", "setupBookmarkEventDelegation",
        "initSearch", "performSearch", "renderSearchResults", "createResultCard",
        "initAdmin", "setupUpload", "uploadFolderZip", "uploadFiles", "loadFiles",
        "deleteFile", "previewFile", "handlePreviewEsc", "closePreviewModal",
        "showAdminContent", "loadModels", "loadRagSettings", "loadStats",
        "escapeJs", "formatSize", "copyToClipboard", "showAuthModal", "hideAuthModal",
        "submitAdminAuth", "manageTags", "showRevisions",
    ):
        text = re.sub(rf"(?m)^function {fn}\b", f"export function {fn}", text)
        text = re.sub(rf"(?m)^async function {fn}\b", f"export async function {fn}", text)

    return text


def main() -> None:
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    if not os.path.exists(BACKUP):
        os.replace(SRC, BACKUP)
        print(f"Backed up to {BACKUP}")

    text = read_src()
    text = patch_ui(text)
    text = strip_duplicate_utils(text)
    text = add_esm_header(text)
    text = add_exports(text)

    with open(DST, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

    shim = (
        "/** @deprecated ESM: /static/js/bootstrap/main.js */\n"
        "export * from './js/legacy/app.js';\n"
    )
    with open(SRC, "w", encoding="utf-8", newline="\n") as f:
        f.write('/** Re-export shim — use type="module" bootstrap */\nimport "./js/legacy/app.js";\n')

    print(f"Wrote {DST}")


if __name__ == "__main__":
    main()