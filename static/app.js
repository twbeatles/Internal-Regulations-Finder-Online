/**
 * 사내 규정 검색기 - 클라이언트 JavaScript v1.6
 * API 통신 및 UI 상호작용 처리
 * 
 * Features:
 * - 하이브리드 검색 (Vector + BM25)
 * - 검색어 자동완성 및 히스토리
 * - 북마크 관리 (이벤트 위임 패턴)
 * - 파일 관리 (업로드/삭제/미리보기)
 * - 시스템 메트릭 모니터링
 * - 다크/라이트 테마
 * - 콘텐츠 접기/펼치기
 * - XSS 보안 강화
 */

// ============================================================================
// UX 유틸리티 - 리플 효과
// ============================================================================
const RippleEffect = {
    init() {
        document.addEventListener('click', (e) => {
            const btn = e.target.closest('.btn, .search-btn');
            if (btn) {
                this.create(btn, e);
            }
        });
    },

    create(element, event) {
        const rect = element.getBoundingClientRect();
        const ripple = document.createElement('span');
        const size = Math.max(rect.width, rect.height);
        const x = event.clientX - rect.left - size / 2;
        const y = event.clientY - rect.top - size / 2;

        ripple.className = 'ripple';
        ripple.style.cssText = `
            width: ${size}px;
            height: ${size}px;
            left: ${x}px;
            top: ${y}px;
        `;

        element.appendChild(ripple);
        ripple.addEventListener('animationend', () => ripple.remove());
    }
};

// ============================================================================
// UX 유틸리티 - 스켈레톤 로딩
// ============================================================================
const SkeletonLoading = {
    // 검색 결과 스켈레톤 생성
    createSearchSkeleton(count = 3) {
        let html = '<div class="skeleton-container">';
        for (let i = 0; i < count; i++) {
            html += `
                <div class="result-card skeleton-card-wrapper" style="animation-delay: ${i * 0.1}s">
                    <div class="result-header">
                        <div class="skeleton skeleton-text" style="width: 60%"></div>
                        <div class="skeleton skeleton-text-sm" style="width: 80px"></div>
                    </div>
                    <div class="skeleton skeleton-card" style="height: 100px"></div>
                    <div class="result-actions">
                        <div class="skeleton" style="width: 80px; height: 32px"></div>
                    </div>
                </div>
            `;
        }
        html += '</div>';
        return html;
    },

    // 파일 테이블 스켈레톤
    createTableSkeleton(rows = 5) {
        let html = '';
        for (let i = 0; i < rows; i++) {
            html += `
                <tr>
                    <td><div class="skeleton skeleton-text-sm" style="width: 60px"></div></td>
                    <td><div class="skeleton skeleton-text" style="width: 80%"></div></td>
                    <td><div class="skeleton skeleton-text-sm" style="width: 50px"></div></td>
                    <td><div class="skeleton skeleton-text-sm" style="width: 30px"></div></td>
                    <td><div class="skeleton" style="width: 60px; height: 24px"></div></td>
                </tr>
            `;
        }
        return html;
    },

    // 통계 카드 스켈레톤
    createStatsSkeleton() {
        return `
            <div class="stats-grid">
                ${Array(4).fill().map(() => `
                    <div class="stat-card">
                        <div class="skeleton" style="width: 40px; height: 40px; border-radius: 50%"></div>
                        <div class="stat-info">
                            <div class="skeleton skeleton-text" style="width: 50px"></div>
                            <div class="skeleton skeleton-text-sm" style="width: 30px"></div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }
};

// ============================================================================
// UX 유틸리티 - 네트워크 상태 감지
// ============================================================================
const NetworkStatus = {
    isOnline: navigator.onLine,
    listeners: [],

    init() {
        window.addEventListener('online', () => this.handleChange(true));
        window.addEventListener('offline', () => this.handleChange(false));
    },

    handleChange(online) {
        this.isOnline = online;

        if (online) {
            Toast.success('연결 복구', '네트워크 연결이 복구되었습니다');
        } else {
            Toast.error('연결 끊김', '네트워크 연결이 끊어졌습니다. 일부 기능이 제한될 수 있습니다.', 0);
        }

        this.listeners.forEach(cb => cb(online));
    },

    onStatusChange(callback) {
        this.listeners.push(callback);
    },

    checkConnection() {
        return this.isOnline;
    }
};

// ============================================================================
// v2.0 UI 기능 모듈
// ============================================================================

// PWA 서비스 워커 등록
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/static/sw.js')
            .then(reg => console.log('SW Registered:', reg))
            .catch(err => console.log('SW Fail:', err));
    });
}

// 키보드 단축키 매니저
const KeyboardShortcuts = {
    modal: null,

    init() {
        this.modal = document.getElementById('shortcuts-modal');

        // 단축키 이벤트 리스너
        document.addEventListener('keydown', (e) => {
            // 입력 중일 때는 단축키 무시 (Esc는 제외)
            if (e.target.matches('input, textarea') && e.key !== 'Escape') {
                return;
            }

            // Ctrl/Cmd + K: 검색창 포커스
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                document.getElementById('search-input').focus();
            }

            // Ctrl/Cmd + /: 도움말 표시 (한글 키보드 대응 '?' 키)
            if ((e.ctrlKey || e.metaKey) && (e.key === '/' || e.key === '?')) {
                e.preventDefault();
                this.toggleHelp();
            }

            // J/K/Down/Up: 검색 결과 탐색 (입력 중 아닐 때)
            if (!this.isInputActive()) {
                if (e.key === 'j' || e.key === 'ArrowDown') {
                    // e.preventDefault(); // 스크롤 방해 금지
                    SearchResultNavigator.next();
                }
                if (e.key === 'k' || e.key === 'ArrowUp') {
                    // e.preventDefault();
                    SearchResultNavigator.prev();
                }

                // N/P: 하이라이트 탐색
                if (e.key === 'n') HighlightNavigator.next();
                if (e.key === 'p') HighlightNavigator.prev();

                // R: 읽기 모드
                if (e.key === 'r') ReaderMode.toggleCurrent();

                // T: 테마 토글
                if (e.key === 't') ThemeManager.toggle();
            }

            // Esc: 모든 모달/읽기모드/네비게이션 닫기
            if (e.key === 'Escape') {
                this.closeAll();
            }
        });

        // 도움말 모달 닫기 버튼
        const closeBtn = document.getElementById('shortcuts-close');
        if (closeBtn) closeBtn.addEventListener('click', () => this.toggleHelp(false));
    },

    isInputActive() {
        const active = document.activeElement;
        return active.tagName === 'INPUT' || active.tagName === 'TEXTAREA';
    },

    toggleHelp(show) {
        if (!this.modal) return;
        const isShown = this.modal.style.display !== 'none';
        const shouldShow = show !== undefined ? show : !isShown;
        this.modal.style.display = shouldShow ? 'flex' : 'none';
    },

    closeAll() {
        this.toggleHelp(false);
        ReaderMode.close();
        HighlightNavigator.close();
        // 기타 모달 닫기
    }
};

// 읽기 모드
const ReaderMode = {
    modal: null,
    body: null,
    fontSize: 16,

    init() {
        this.modal = document.getElementById('reader-modal');
        this.body = document.getElementById('reader-body');

        if (!this.modal) return;

        // 버튼 이벤트
        document.getElementById('reader-close').addEventListener('click', () => this.close());

        document.getElementById('reader-font-up').addEventListener('click', () => {
            this.fontSize = Math.min(this.fontSize + 2, 32);
            this.updateFont();
        });

        document.getElementById('reader-font-down').addEventListener('click', () => {
            this.fontSize = Math.max(this.fontSize - 2, 12);
            this.updateFont();
        });
    },

    open(title, content) {
        if (!this.modal) return;

        document.getElementById('reader-title').textContent = title;
        // 마크다운 변환 또는 줄바꿈 처리
        this.body.innerHTML = content.replace(/\n/g, '<br>');
        this.modal.style.display = 'flex';
        document.body.style.overflow = 'hidden'; // 배경 스크롤 방지
        this.updateFont();
    },

    close() {
        if (this.modal) this.modal.style.display = 'none';
        document.body.style.overflow = '';
    },

    updateFont() {
        if (this.body) {
            this.body.style.fontSize = `${this.fontSize}px`;
            document.getElementById('reader-font-size').textContent = `${this.fontSize}px`;
        }
    },

    toggleCurrent() {
        // 현재 선택된 결과가 있으면 읽기 모드로 열기
        const selected = document.querySelector('.result-card.selected');
        if (selected) {
            const title = selected.querySelector('.result-source').textContent;
            const content = selected.querySelector('.result-content').textContent;
            this.open(title, content);
        }
    }
};

// 하이라이트 네비게이터
const HighlightNavigator = {
    container: null,
    highlights: [],
    currentIndex: -1,

    init() {
        this.container = document.getElementById('highlight-nav');
        if (!this.container) return;

        document.getElementById('highlight-prev').addEventListener('click', () => this.prev());
        document.getElementById('highlight-next').addEventListener('click', () => this.next());
        document.getElementById('highlight-close').addEventListener('click', () => this.close());
    },

    scan() {
        this.highlights = Array.from(document.querySelectorAll('mark.highlight'));
        this.currentIndex = -1;
        this.updateUI();

        if (this.highlights.length > 0) {
            this.container.style.display = 'flex';
        } else {
            this.container.style.display = 'none';
        }
    },

    next() {
        if (this.highlights.length === 0) return;
        this.currentIndex = (this.currentIndex + 1) % this.highlights.length;
        this.focus();
    },

    prev() {
        if (this.highlights.length === 0) return;
        this.currentIndex = (this.currentIndex - 1 + this.highlights.length) % this.highlights.length;
        this.focus();
    },

    focus() {
        const el = this.highlights[this.currentIndex];
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });

        // 현재 활성화 표시
        this.highlights.forEach(h => h.classList.remove('active'));
        el.classList.add('active');

        this.updateUI();
    },

    updateUI() {
        const countEl = document.getElementById('highlight-count');
        if (countEl) {
            countEl.textContent = this.highlights.length > 0 ?
                `${this.currentIndex + 1} / ${this.highlights.length}` : '0 / 0';
        }
    },

    close() {
        if (this.container) this.container.style.display = 'none';
        this.highlights.forEach(h => h.classList.remove('active'));
    }
};

// 검색 결과 키보드 탐색
const SearchResultNavigator = {
    cards: [],

    scan() {
        this.cards = Array.from(document.querySelectorAll('.result-card'));
    },

    next() {
        this.scan();
        if (this.cards.length === 0) return;

        const current = document.querySelector('.result-card.selected');
        let nextIndex = 0;

        if (current) {
            current.classList.remove('selected');
            const idx = this.cards.indexOf(current);
            nextIndex = (idx + 1) % this.cards.length;
        }

        this.select(nextIndex);
    },

    prev() {
        this.scan();
        if (this.cards.length === 0) return;

        const current = document.querySelector('.result-card.selected');
        let prevIndex = this.cards.length - 1;

        if (current) {
            current.classList.remove('selected');
            const idx = this.cards.indexOf(current);
            prevIndex = (idx - 1 + this.cards.length) % this.cards.length;
        }

        this.select(prevIndex);
    },

    select(index) {
        const card = this.cards[index];
        if (card) {
            card.classList.add('selected');
            card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }
};


// ============================================================================
// UX 유틸리티 - 스태거드 애니메이션
// ============================================================================
const StaggerAnimation = {
    apply(container, selector = '.result-card', baseDelay = 0.08) {
        const items = container.querySelectorAll(selector);
        items.forEach((item, index) => {
            item.classList.add('stagger-item');
            item.style.animationDelay = `${index * baseDelay}s`;
        });
    }
};

// ============================================================================
// UX 유틸리티 - 검색 결과 내보내기
// ============================================================================
const ExportResults = {
    lastResults: [],  // 마지막 검색 결과 저장
    lastQuery: '',

    // 결과 저장 (검색 후 호출)
    saveResults(results, query) {
        this.lastResults = results;
        this.lastQuery = query;
    },

    // 텍스트 형식으로 내보내기
    exportAsText() {
        if (!this.lastResults.length) {
            Toast.warning('내보내기 실패', '검색 결과가 없습니다');
            return;
        }

        let text = `검색어: "${this.lastQuery}"\n`;
        text += `검색 결과: ${this.lastResults.length}개\n`;
        text += '='.repeat(50) + '\n\n';

        this.lastResults.forEach((item, index) => {
            const score = Math.round((item.score || 0) * 100);
            text += `[${index + 1}] ${item.source || '알 수 없음'} (${score}%)\n`;
            text += '-'.repeat(40) + '\n';
            text += (item.content || '') + '\n\n';
        });

        this.download(text, `검색결과_${this.lastQuery}.txt`, 'text/plain');
        Toast.success('내보내기 완료', '텍스트 파일로 저장되었습니다');
    },

    // 마크다운 형식으로 내보내기
    exportAsMarkdown() {
        if (!this.lastResults.length) {
            Toast.warning('내보내기 실패', '검색 결과가 없습니다');
            return;
        }

        let md = `# 검색 결과: "${this.lastQuery}"\n\n`;
        md += `> 총 ${this.lastResults.length}개 결과\n\n`;

        this.lastResults.forEach((item, index) => {
            const score = Math.round((item.score || 0) * 100);
            md += `## ${index + 1}. ${item.source || '알 수 없음'}\n\n`;
            md += `**관련도:** ${score}%\n\n`;
            md += '```\n' + (item.content || '') + '\n```\n\n';
            md += '---\n\n';
        });

        this.download(md, `검색결과_${this.lastQuery}.md`, 'text/markdown');
        Toast.success('내보내기 완료', '마크다운 파일로 저장되었습니다');
    },

    // JSON 형식으로 내보내기
    exportAsJSON() {
        if (!this.lastResults.length) {
            Toast.warning('내보내기 실패', '검색 결과가 없습니다');
            return;
        }

        const data = {
            query: this.lastQuery,
            timestamp: new Date().toISOString(),
            resultCount: this.lastResults.length,
            results: this.lastResults.map((item, index) => ({
                rank: index + 1,
                source: item.source,
                score: Math.round((item.score || 0) * 100),
                content: item.content
            }))
        };

        const json = JSON.stringify(data, null, 2);
        this.download(json, `검색결과_${this.lastQuery}.json`, 'application/json');
        Toast.success('내보내기 완료', 'JSON 파일로 저장되었습니다');
    },

    // 파일 다운로드 헬퍼
    download(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType + ';charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    },

    // 내보내기 드롭다운 메뉴 표시/숨김
    toggleMenu(button) {
        const menu = document.getElementById('export-menu');
        if (menu) {
            menu.classList.toggle('visible');
            // 외부 클릭 시 닫기
            if (menu.classList.contains('visible')) {
                setTimeout(() => {
                    document.addEventListener('click', this.closeMenu, { once: true });
                }, 0);
            }
        }
    },

    closeMenu(e) {
        const menu = document.getElementById('export-menu');
        if (menu && !e.target.closest('.export-dropdown')) {
            menu.classList.remove('visible');
        }
    }
};

// ============================================================================
// API 클라이언트
// ============================================================================
const API = {
    baseUrl: '',
    pendingRequests: new Map(),  // 진행 중인 요청 추적
    maxRetries: 3,  // 최대 재시도 횟수

    async fetch(endpoint, options = {}) {
        // 중복 요청 방지 (POST 요청에 대해서만)
        const requestKey = `${options.method || 'GET'}-${endpoint}-${JSON.stringify(options.body || '')}`;

        if (options.method === 'POST' && this.pendingRequests.has(requestKey)) {
            console.log('Duplicate request prevented:', endpoint);
            return this.pendingRequests.get(requestKey);
        }

        const controller = new AbortController();
        const timeout = options.timeout || 30000; // 30초 기본 타임아웃
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        const requestPromise = this._executeRequest(endpoint, options, controller, timeoutId);

        if (options.method === 'POST') {
            this.pendingRequests.set(requestKey, requestPromise);
            requestPromise.finally(() => {
                this.pendingRequests.delete(requestKey);
            });
        }

        return requestPromise;
    },

    async _executeRequest(endpoint, options, controller, timeoutId, retryCount = 0) {
        try {
            const response = await fetch(this.baseUrl + endpoint, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                signal: controller.signal,
                ...options
            });
            clearTimeout(timeoutId);

            // Rate Limit 처리 (429)
            if (response.status === 429) {
                const data = await response.json().catch(() => ({}));
                const retryAfter = data.retry_after || 60;
                Toast.warning('요청 제한', `잠시 후 다시 시도해주세요 (${retryAfter}초)`);
                return { success: false, message: '요청이 너무 많습니다. 잠시 후 다시 시도해주세요.', retry_after: retryAfter };
            }

            // 서버 과부하 (503) - 재시도
            if (response.status === 503 && retryCount < this.maxRetries) {
                console.log(`Server busy, retrying... (${retryCount + 1}/${this.maxRetries})`);
                await new Promise(resolve => setTimeout(resolve, 1000 * (retryCount + 1)));
                return this._executeRequest(endpoint, options, controller, timeoutId, retryCount + 1);
            }

            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                const data = await response.json();
                return data;
            } else {
                // JSON이 아닌 응답 (예: HTML 에러 페이지)
                const text = await response.text();
                // <!doctype ... 로 시작하면 HTML로 간주
                if (text.trim().toLowerCase().startsWith('<!doctype') || text.trim().toLowerCase().startsWith('<html')) {
                    console.error('API Returned HTML:', text.substring(0, 200));
                    return { success: false, message: '서버 오류: 올바르지 않은 응답 형식입니다 (HTML)' };
                }
                return { success: false, message: '서버 응답 오류 (Non-JSON)' };
            }
        } catch (error) {
            clearTimeout(timeoutId);
            console.error('API Error:', error);

            // 네트워크 오류 시 재시도
            if (error.name !== 'AbortError' && retryCount < this.maxRetries) {
                console.log(`Network error, retrying... (${retryCount + 1}/${this.maxRetries})`);
                await new Promise(resolve => setTimeout(resolve, 1000 * (retryCount + 1)));
                return this._executeRequest(endpoint, options, controller, timeoutId, retryCount + 1);
            }

            if (error.name === 'AbortError') {
                return { success: false, message: '요청 시간이 초과되었습니다' };
            }
            return { success: false, message: error.message || '서버 연결 실패' };
        }
    },


    getStatus() {
        return this.fetch('/api/status');
    },

    search(query, k = 5, hybrid = true, highlight = true, filterFile = null, sortBy = 'relevance') {
        return this.fetch('/api/search', {
            method: 'POST',
            body: JSON.stringify({ query, k, hybrid, highlight, filter_file: filterFile, sort_by: sortBy })
        });
    },

    getSearchHistory(limit = 10) {
        return this.fetch(`/api/search/history?limit=${limit}`);
    },

    getSuggestions(query, limit = 8) {
        return this.fetch(`/api/search/suggest?q=${encodeURIComponent(query)}&limit=${limit}`);
    },

    getFiles() {
        return this.fetch('/api/files');
    },

    async uploadFiles(files) {
        const formData = new FormData();
        for (const file of files) {
            formData.append('files', file);
        }

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            return await response.json();
        } catch (error) {
            console.error('Upload Error:', error);
            return { success: false, message: '업로드 실패' };
        }
    },

    reprocessFiles() {
        return this.fetch('/api/process', { method: 'POST' });
    },

    clearCache() {
        return this.fetch('/api/cache', { method: 'DELETE' });
    },

    getModels() {
        return this.fetch('/api/models');
    },

    setModel(modelName) {
        return this.fetch('/api/models', {
            method: 'POST',
            body: JSON.stringify({ model: modelName })
        });
    },

    deleteFile(filename) {
        return this.fetch(`/api/files/${encodeURIComponent(filename)}`, { method: 'DELETE' });
    },

    getFilePreview(filename, length = 2000) {
        return this.fetch(`/api/files/${encodeURIComponent(filename)}/preview?length=${length}`);
    },

    // 관리자 인증 API
    checkAdminAuth() {
        return this.fetch('/api/admin/check', { credentials: 'include' });
    },

    adminAuth(password) {
        return this.fetch('/api/admin/auth', {
            method: 'POST',
            body: JSON.stringify({ password }),
            credentials: 'include'
        });
    },

    adminLogout() {
        return this.fetch('/api/admin/logout', { method: 'POST', credentials: 'include' });
    },

    // 파일명 목록 (필터용)
    getFileNames() {
        return this.fetch('/api/files/names');
    },

    // 검색 통계
    getSearchStats(limit = 10) {
        return this.fetch(`/api/stats/search?limit=${limit}`);
    },

    // v2.0 API 메소드
    getSyncStatus() {
        return this.fetch('/api/sync/status');
    },

    startSync(folder) {
        return this.fetch('/api/sync/start', { method: 'POST', body: JSON.stringify({ folder }) });
    },

    stopSync() {
        return this.fetch('/api/sync/stop', { method: 'POST' });
    },

    async uploadFolder(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload/folder', {
                method: 'POST',
                body: formData
            });
            return await response.json();
        } catch (error) {
            console.error('Folder Upload Error:', error);
            return { success: false, message: '업로드 실패' };
        }
    },

    getRevisions(filename) {
        return this.fetch(`/api/files/${encodeURIComponent(filename)}/revisions`);
    },

    // 태그 관리
    getFileTags(filename) {
        return this.fetch(`/api/files/structure`).then(res => {
            if (!res.success) return { success: false, tags: [] };
            const file = res.files.find(f => f.name === filename);
            return { success: true, tags: file ? file.tags : [] };
        });
    },

    setFileTags(filename, tags) {
        return this.fetch('/api/tags/set', {
            method: 'POST',
            body: JSON.stringify({ filename, tags })
        });
    },

    autoTagFile(filename) {
        return this.fetch('/api/tags/auto', {
            method: 'POST',
            body: JSON.stringify({ filename })
        });
    },

    getAllTags() {
        return this.fetch('/api/tags');
    }
};

// ============================================================================
// 북마크 매니저 (로컬스토리지 기반)
// ============================================================================
const BookmarkManager = {
    STORAGE_KEY: 'regulation_bookmarks',

    getAll() {
        try {
            const data = localStorage.getItem(this.STORAGE_KEY);
            return data ? JSON.parse(data) : [];
        } catch {
            return [];
        }
    },

    save(bookmarks) {
        localStorage.setItem(this.STORAGE_KEY, JSON.stringify(bookmarks));
    },

    add(item) {
        const bookmarks = this.getAll();
        // 중복 체크 (content 기준)
        const exists = bookmarks.some(b => b.content.substring(0, 100) === item.content.substring(0, 100));
        if (!exists) {
            bookmarks.unshift({
                id: Date.now(),
                source: item.source,
                content: item.content,
                score: item.score,
                addedAt: new Date().toISOString()
            });
            this.save(bookmarks.slice(0, 50)); // 최대 50개 유지
            return true;
        }
        return false;
    },

    remove(id) {
        const bookmarks = this.getAll().filter(b => b.id !== id);
        this.save(bookmarks);
    },

    isBookmarked(content) {
        return this.getAll().some(b => b.content.substring(0, 100) === content.substring(0, 100));
    }
};

// ============================================================================
// 토스트 알림
// ============================================================================
const Toast = {
    container: null,

    init() {
        this.container = document.getElementById('toast-container');
    },

    show(type, title, message, duration = 4000) {
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <span class="toast-icon">${icons[type]}</span>
            <div class="toast-content">
                <div class="toast-title">${title}</div>
                ${message ? `<div class="toast-message">${message}</div>` : ''}
            </div>
            <button class="toast-close" onclick="Toast.close(this)">✕</button>
        `;

        if (this.container) {
            this.container.appendChild(toast);
        }

        if (duration > 0) {
            setTimeout(() => this.remove(toast), duration);
        }

        return toast;
    },

    close(btn) {
        const toast = btn.closest('.toast');
        this.remove(toast);
    },

    remove(toast) {
        if (!toast) return;
        toast.classList.add('toast-out');
        setTimeout(() => toast.remove(), 300);
    },

    success(title, message) { return this.show('success', title, message); },
    error(title, message) { return this.show('error', title, message); },
    warning(title, message) { return this.show('warning', title, message); },
    info(title, message) { return this.show('info', title, message); }
};

// ============================================================================
// 테마 관리
// ============================================================================
const ThemeManager = {
    storageKey: 'theme-preference',

    init() {
        // 저장된 테마 또는 시스템 테마 적용
        const savedTheme = localStorage.getItem(this.storageKey);
        if (savedTheme) {
            this.setTheme(savedTheme, false);
        } else {
            // 시스템 테마 감지
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            this.setTheme(prefersDark ? 'dark' : 'light', false);
        }

        // 시스템 테마 변경 감지
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            if (!localStorage.getItem(this.storageKey)) {
                this.setTheme(e.matches ? 'dark' : 'light', false);
            }
        });

        // 테마 토글 버튼 초기화
        this.setupToggleButton();
    },

    setTheme(theme, save = true) {
        document.documentElement.setAttribute('data-theme', theme);

        if (save) {
            localStorage.setItem(this.storageKey, theme);
        }

        // 버튼 아이콘 업데이트
        this.updateToggleButton(theme);
    },

    getTheme() {
        return document.documentElement.getAttribute('data-theme') || 'dark';
    },

    toggle() {
        const current = this.getTheme();
        const newTheme = current === 'dark' ? 'light' : 'dark';
        this.setTheme(newTheme);
        return newTheme;
    },

    setupToggleButton() {
        const btn = document.getElementById('theme-toggle');
        if (btn) {
            btn.addEventListener('click', () => {
                const newTheme = this.toggle();
                Toast.info('테마 변경', newTheme === 'dark' ? '다크 모드' : '라이트 모드');
            });
        }
    },

    updateToggleButton(theme) {
        const btn = document.getElementById('theme-toggle');
        if (btn) {
            btn.innerHTML = theme === 'dark' ? '☀️' : '🌙';
            btn.title = theme === 'dark' ? '라이트 모드로 전환' : '다크 모드로 전환';
        }
    }
};

// ============================================================================
// 상태 관리
// ============================================================================
const AppState = {
    ready: false,
    loading: false,
    refreshInterval: null,

    async checkStatus() {
        const result = await API.getStatus();
        this.updateStatusUI(result);
        return result;
    },

    updateStatusUI(result) {
        const badge = document.getElementById('status-badge');
        const text = document.getElementById('status-text');

        if (!badge || !text) return;

        badge.classList.remove('loading', 'ready', 'error');

        if (result.loading) {
            badge.classList.add('loading');
            text.textContent = result.progress || '로딩 중...';
            this.ready = false;
        } else if (result.ready) {
            badge.classList.add('ready');
            text.textContent = '준비 완료';
            this.ready = true;
            this.stopRefresh();  // 준비 완료 시 자동 새로고침 중지
            this.enableSearch();
        } else {
            badge.classList.add('error');
            text.textContent = '오류';
            this.ready = false;
        }
    },

    enableSearch() {
        const searchBtn = document.getElementById('search-btn');
        const searchInput = document.getElementById('search-input');

        if (searchBtn) searchBtn.disabled = false;
        if (searchInput) searchInput.disabled = false;
    },

    startRefresh(interval = 3000) {
        this.stopRefresh();
        this.refreshInterval = setInterval(() => {
            this.checkStatus();
        }, interval);
    },

    stopRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }
};

// ============================================================================
// 자동완성 시스템
// ============================================================================
const Autocomplete = {
    container: null,
    input: null,
    dropdown: null,
    debounceTimer: null,
    selectedIndex: -1,
    suggestions: [],
    isVisible: false,

    init(inputElement) {
        this.input = inputElement;
        if (!this.input) return;

        // 드롭다운 컨테이너 생성
        this.createDropdown();

        // 이벤트 바인딩
        this.input.addEventListener('input', () => this.onInput());
        this.input.addEventListener('focus', () => this.onFocus());
        this.input.addEventListener('blur', () => setTimeout(() => this.hide(), 200));
        this.input.addEventListener('keydown', (e) => this.onKeydown(e));
    },

    createDropdown() {
        // 기존 드롭다운 제거
        const existing = document.getElementById('autocomplete-dropdown');
        if (existing) existing.remove();

        this.dropdown = document.createElement('div');
        this.dropdown.id = 'autocomplete-dropdown';
        this.dropdown.className = 'autocomplete-dropdown';
        this.dropdown.innerHTML = '';

        // 검색 박스 컨테이너에 추가
        const searchBox = this.input.closest('.search-box');
        if (searchBox) {
            searchBox.style.position = 'relative';
            searchBox.appendChild(this.dropdown);
        }
    },

    async onInput() {
        const query = this.input.value.trim();

        // 디바운싱
        clearTimeout(this.debounceTimer);
        this.debounceTimer = setTimeout(async () => {
            if (query.length < 1) {
                this.showHistory();
                return;
            }
            await this.fetchSuggestions(query);
        }, 300);
    },

    async onFocus() {
        const query = this.input.value.trim();
        if (query.length < 1) {
            this.showHistory();
        } else {
            await this.fetchSuggestions(query);
        }
    },

    onKeydown(e) {
        if (!this.isVisible) return;

        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                this.selectNext();
                break;
            case 'ArrowUp':
                e.preventDefault();
                this.selectPrev();
                break;
            case 'Enter':
                if (this.selectedIndex >= 0) {
                    e.preventDefault();
                    this.selectCurrent();
                }
                break;
            case 'Escape':
                this.hide();
                break;
        }
    },

    async fetchSuggestions(query) {
        const result = await API.getSuggestions(query);
        if (result.success && result.suggestions) {
            this.suggestions = result.suggestions;
            this.render(result.suggestions, 'suggestions');
        }
    },

    async showHistory() {
        const result = await API.getSearchHistory(5);
        if (result.success) {
            const items = [];
            if (result.recent && result.recent.length > 0) {
                items.push({ type: 'header', text: '최근 검색어' });
                result.recent.forEach(q => items.push({ type: 'recent', text: q }));
            }
            if (result.popular && result.popular.length > 0) {
                items.push({ type: 'header', text: '인기 검색어' });
                result.popular.forEach(p => items.push({ type: 'popular', text: p.query, count: p.count }));
            }
            this.renderHistory(items);
        }
    },

    renderHistory(items) {
        if (!this.dropdown || items.length === 0) {
            this.hide();
            return;
        }

        let html = '';
        items.forEach((item, index) => {
            if (item.type === 'header') {
                html += `<div class="autocomplete-header">${escapeHtml(item.text)}</div>`;
            } else {
                const icon = item.type === 'recent' ? '🕐' : '🔥';
                const countBadge = item.count ? `<span class="count-badge">${item.count}</span>` : '';
                html += `<div class="autocomplete-item" data-query="${escapeHtml(item.text)}">
                    <span class="item-icon">${icon}</span>
                    <span class="item-text">${escapeHtml(item.text)}</span>
                    ${countBadge}
                </div>`;
            }
        });

        this.dropdown.innerHTML = html;
        this.bindItemEvents();
        this.show();
    },

    render(suggestions, type = 'suggestions') {
        if (!this.dropdown || suggestions.length === 0) {
            this.hide();
            return;
        }

        this.suggestions = suggestions;
        this.selectedIndex = -1;

        let html = '';
        suggestions.forEach((text, index) => {
            html += `<div class="autocomplete-item" data-index="${index}" data-query="${escapeHtml(text)}">
                <span class="item-icon">🔍</span>
                <span class="item-text">${escapeHtml(text)}</span>
            </div>`;
        });

        this.dropdown.innerHTML = html;
        this.bindItemEvents();
        this.show();
    },

    bindItemEvents() {
        const items = this.dropdown.querySelectorAll('.autocomplete-item');
        items.forEach(item => {
            item.addEventListener('click', () => {
                const query = item.dataset.query;
                if (query) {
                    this.input.value = query;
                    this.hide();
                    performSearch();
                }
            });
            item.addEventListener('mouseenter', () => {
                items.forEach(i => i.classList.remove('selected'));
                item.classList.add('selected');
            });
        });
    },

    selectNext() {
        const items = this.dropdown.querySelectorAll('.autocomplete-item');
        if (items.length === 0) return;
        this.selectedIndex = (this.selectedIndex + 1) % items.length;
        this.updateSelection(items);
    },

    selectPrev() {
        const items = this.dropdown.querySelectorAll('.autocomplete-item');
        if (items.length === 0) return;
        this.selectedIndex = this.selectedIndex <= 0 ? items.length - 1 : this.selectedIndex - 1;
        this.updateSelection(items);
    },

    updateSelection(items) {
        items.forEach((item, index) => {
            item.classList.toggle('selected', index === this.selectedIndex);
        });
    },

    selectCurrent() {
        const items = this.dropdown.querySelectorAll('.autocomplete-item');
        if (this.selectedIndex >= 0 && items[this.selectedIndex]) {
            const query = items[this.selectedIndex].dataset.query;
            if (query) {
                this.input.value = query;
                this.hide();
                performSearch();
            }
        }
    },

    show() {
        if (this.dropdown) {
            this.dropdown.classList.add('visible');
            this.isVisible = true;
        }
    },

    hide() {
        if (this.dropdown) {
            this.dropdown.classList.remove('visible');
            this.isVisible = false;
            this.selectedIndex = -1;
        }
    }
};

// ============================================================================
// 파일 목록 로드 (검색 필터용)
// ============================================================================
async function loadFileListForFilter() {
    const filterSelect = document.getElementById('filter-file');
    if (!filterSelect) return;

    const result = await API.getFileNames();
    if (result.success && result.files) {
        // 기존 옵션 유지 (전체 파일)
        const existingOptions = filterSelect.innerHTML;
        let html = '<option value="">전체 파일</option>';

        result.files.forEach(filename => {
            html += `<option value="${escapeHtml(filename)}">${escapeHtml(filename)}</option>`;
        });

        filterSelect.innerHTML = html;
    }
}

// ============================================================================
// 북마크 토글 (XSS 방지를 위해 인덱스 기반으로 변경)
// ============================================================================
// 현재 검색 결과 저장 (이벤트 위임용)
let currentSearchResults = [];

function toggleBookmarkByIndex(index, buttonElement) {
    const item = currentSearchResults[index];
    if (!item) {
        Toast.error('오류', '북마크 대상을 찾을 수 없습니다');
        return;
    }

    const isBookmarked = BookmarkManager.isBookmarked(item.content);

    if (isBookmarked) {
        // 북마크 제거
        const bookmarks = BookmarkManager.getAll();
        const bookmark = bookmarks.find(b => b.content.substring(0, 100) === item.content.substring(0, 100));
        if (bookmark) {
            BookmarkManager.remove(bookmark.id);
            buttonElement.textContent = '☆';
            buttonElement.title = '북마크 추가';
            buttonElement.classList.remove('bookmarked');
            Toast.info('북마크 해제', '북마크가 제거되었습니다');
        }
    } else {
        // 북마크 추가
        if (BookmarkManager.add(item)) {
            buttonElement.textContent = '⭐';
            buttonElement.title = '북마크 해제';
            buttonElement.classList.add('bookmarked');
            Toast.success('북마크 저장', '북마크에 추가되었습니다');
        }
    }
}

// 이벤트 위임으로 북마크 버튼 처리
function setupBookmarkEventDelegation() {
    const container = document.getElementById('results-container');
    if (!container) return;

    container.addEventListener('click', (e) => {
        const bookmarkBtn = e.target.closest('.btn-bookmark');
        if (bookmarkBtn) {
            const index = parseInt(bookmarkBtn.dataset.index, 10);
            if (!isNaN(index)) {
                toggleBookmarkByIndex(index, bookmarkBtn);
            }
        }

        // 복사 버튼 처리
        const copyBtn = e.target.closest('.btn-copy');
        if (copyBtn) {
            const index = parseInt(copyBtn.dataset.index, 10);
            if (!isNaN(index) && currentSearchResults[index]) {
                copyToClipboard(currentSearchResults[index].content || '');
            }
        }

        // 콘텐츠 접기/펼치기 처리
        const toggleBtn = e.target.closest('.btn-toggle-content');
        if (toggleBtn) {
            const card = toggleBtn.closest('.result-card');
            if (card) {
                card.classList.toggle('collapsed');
                toggleBtn.textContent = card.classList.contains('collapsed') ? '펼치기 ▼' : '접기 ▲';
            }
        }
    });
}

// ============================================================================
// 검색 페이지 (메인)
// ============================================================================
async function initSearch() {
    Toast.init();
    ThemeManager.init();
    RippleEffect.init();
    NetworkStatus.init();

    // 초기 상태 확인
    const status = await AppState.checkStatus();

    if (!status.ready) {
        AppState.startRefresh(2000);
    }

    // 파일 목록 로드 (필터 드롭다운용)
    loadFileListForFilter();

    // 검색 이벤트
    const searchBtn = document.getElementById('search-btn');
    const searchInput = document.getElementById('search-input');

    if (searchBtn) {
        searchBtn.addEventListener('click', () => {
            Autocomplete.hide();
            performSearch();
        });
    }

    if (searchInput) {
        // 자동완성 초기화
        Autocomplete.init(searchInput);

        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && Autocomplete.selectedIndex < 0) {
                Autocomplete.hide();
                performSearch();
            }
        });

        // 포커스
        if (status.ready) {
            searchInput.focus();
        }
    }

    // v2.0 UI 모듈 초기화
    KeyboardShortcuts.init();
    ReaderMode.init();
    HighlightNavigator.init();

    // 이벤트 위임 설정 (북마크, 복사, 접기/펼치기)
    setupBookmarkEventDelegation();
}

async function performSearch() {
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
    const filter = filterFile?.value || null;
    const sort = sortBy?.value || 'relevance';

    const result = await API.search(query, k, hybrid, true, filter, sort);

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
}

function renderSearchResults(results, query) {
    const container = document.getElementById('results-container');
    if (!container) return;

    // 결과 저장 (내보내기용)
    ExportResults.saveResults(results, query);

    // XSS 방지를 위해 query를 이스케이프
    const safeQuery = escapeHtml(query);

    let html = `
        <div class="results-header">
            <span class="results-query">🔎 "${safeQuery}"</span>
            <div class="results-actions-header">
                <span class="results-count">${results.length}개 결과</span>
                <div class="export-dropdown">
                    <button class="btn btn-secondary btn-sm" onclick="ExportResults.toggleMenu(this)" aria-haspopup="true" aria-expanded="false">
                        📥 내보내기 ▾
                    </button>
                    <div id="export-menu" class="export-menu" role="menu">
                        <button class="export-item" onclick="ExportResults.exportAsText()" role="menuitem">
                            📄 텍스트 (.txt)
                        </button>
                        <button class="export-item" onclick="ExportResults.exportAsMarkdown()" role="menuitem">
                            📝 마크다운 (.md)
                        </button>
                        <button class="export-item" onclick="ExportResults.exportAsJSON()" role="menuitem">
                            📋 JSON (.json)
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

    // 검색 결과 저장 (이벤트 위임용)
    currentSearchResults = results;

    results.forEach((item, index) => {
        const score = Math.round((item.score || 0) * 100);
        const scoreClass = score >= 70 ? 'high' : score >= 40 ? 'medium' : 'low';
        const isBookmarked = BookmarkManager.isBookmarked(item.content || '');
        const bookmarkIcon = isBookmarked ? '⭐' : '☆';
        const bookmarkTitle = isBookmarked ? '북마크 해제' : '북마크 추가';
        const bookmarkClass = isBookmarked ? 'bookmarked' : '';

        // 서버에서 하이라이트된 컨텐츠 사용 (없으면 일반 컨텐츠)
        const displayContent = item.content_highlighted || escapeHtml(item.content || '');

        // 콘텐츠 길이에 따라 접기 버튼 표시
        const contentLength = (item.content || '').length;
        const showToggle = contentLength > 300;

        html += `
            <div class="result-card" style="animation-delay: ${index * 0.1}s">
                <div class="result-header">
                    <div class="result-title">
                        <span class="result-index">${index + 1}</span>
                        <span class="result-source">${escapeHtml(item.source || '알 수 없음')}</span>
                        <button class="btn-bookmark ${bookmarkClass}" 
                                data-index="${index}"
                                title="${bookmarkTitle}"
                                aria-label="${bookmarkTitle}">${bookmarkIcon}</button>
                    </div>
                    <div class="result-score">
                        <span class="score-value ${scoreClass}">${score}%</span>
                        <div class="score-bar" role="progressbar" aria-valuenow="${score}" aria-valuemin="0" aria-valuemax="100">
                            <div class="score-fill ${scoreClass}" style="width: ${score}%"></div>
                        </div>
                    </div>
                </div>
                <div class="result-content">${displayContent}</div>
                <div class="result-actions">
                    ${showToggle ? '<button class="btn btn-sm btn-toggle-content">접기 ▲</button>' : ''}
                    <button class="btn btn-secondary btn-copy" data-index="${index}">
                        📋 복사
                    </button>
                    <a href="/api/files/${encodeURIComponent(item.source || '')}/download" 
                       class="btn btn-primary" 
                       download
                       title="원본 파일 다운로드">
                        📥 원본 파일
                    </a>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;

    // 스타거 애니메이션 적용
    StaggerAnimation.apply(container, '.result-card', 0.08);

    // v2.0 네비게이션 갱신
    SearchResultNavigator.scan();
    HighlightNavigator.scan();
}

// ============================================================================
// 관리자 페이지
// ============================================================================
async function initAdmin() {
    Toast.init();
    ThemeManager.init();
    RippleEffect.init();
    NetworkStatus.init();

    // 테마 토글
    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
        themeBtn.addEventListener('click', () => {
            ThemeManager.toggle();
            themeBtn.textContent = ThemeManager.currentTheme === 'dark' ? '🌙' : '☀️';
            // 버튼 애니메이션
            themeBtn.style.transform = 'rotate(360deg)';
            setTimeout(() => themeBtn.style.transform = '', 300);
        });
        // 초기 아이콘 설정
        themeBtn.textContent = ThemeManager.currentTheme === 'dark' ? '🌙' : '☀️';
    }

    // 관리자 인증 확인
    const authResult = await API.checkAdminAuth();
    if (authResult.success && authResult.required && !authResult.authenticated) {
        // 인증 필요 - 모달 표시
        showAuthModal();
        return; // 인증 전까지 나머지 초기화 중단
    }

    // 인증 완료 처리 (콘텐츠 표시)
    showAdminContent();

    // 초기 상태 확인
    await AppState.checkStatus();
    await loadFiles();
    await loadStats();
    await loadModels();

    // 파일 업로드 설정
    setupUpload();

    // v2.0 관리자 기능 초기화
    await FolderSync.init();

    // 업로드 탭 전환
    const btnFile = document.getElementById('btn-upload-file');
    const btnFolder = document.getElementById('btn-upload-folder');
    const areaFile = document.getElementById('upload-area');
    const areaFolder = document.getElementById('folder-upload-area');

    if (btnFile && btnFolder && areaFile && areaFolder) {
        btnFile.addEventListener('click', () => {
            btnFile.classList.add('active');
            btnFolder.classList.remove('active');
            areaFile.style.display = 'block';
            areaFolder.style.display = 'none';
        });

        btnFolder.addEventListener('click', () => {
            btnFolder.classList.add('active');
            btnFile.classList.remove('active');
            areaFolder.style.display = 'block';
            areaFile.style.display = 'none';
        });
    }

    // 버튼 이벤트
    document.getElementById('refresh-btn')?.addEventListener('click', async () => {
        await loadFiles();
        await loadStats();
        Toast.success('새로고침', '파일 목록을 갱신했습니다');
    });

    document.getElementById('reprocess-btn')?.addEventListener('click', async () => {
        const btn = document.getElementById('reprocess-btn');
        btn.disabled = true;
        btn.textContent = '처리 중...';

        const result = await API.reprocessFiles();

        if (result.success) {
            Toast.success('재처리 완료', result.message);
            await loadFiles();
            await loadStats();
        } else {
            Toast.error('재처리 실패', result.message);
        }

        btn.disabled = false;
        btn.textContent = '⚡ 재처리';
    });

    document.getElementById('clear-cache-btn')?.addEventListener('click', async () => {
        if (!confirm('캐시를 삭제하시겠습니까?\n다음 검색 시 인덱스가 재생성됩니다.')) return;

        const result = await API.clearCache();

        if (result.success) {
            Toast.success('캐시 삭제', result.message);
        } else {
            Toast.error('실패', result.message);
        }
    });

    // 주기적 상태 갱신
    setInterval(async () => {
        await AppState.checkStatus();
        await loadStats();
    }, 10000);
}

function setupUpload() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');

    if (!uploadArea || !fileInput) return;

    // 클릭 업로드
    uploadArea.addEventListener('click', () => fileInput.click());

    // 파일 선택
    fileInput.addEventListener('change', async (e) => {
        if (e.target.files.length > 0) {
            await uploadFiles(e.target.files);
        }
    });

    // 드래그 앤 드롭
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', async (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            await uploadFiles(files);
        }
    });

    // 폴더 업로드 (ZIP) 영역 설정
    const folderArea = document.getElementById('folder-upload-area');
    const folderInput = document.getElementById('folder-input');

    if (folderArea && folderInput) {
        folderArea.addEventListener('click', () => folderInput.click());

        folderInput.addEventListener('change', async (e) => {
            if (e.target.files.length > 0) {
                await uploadFolderZip(e.target.files[0]);
            }
        });

        folderArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            folderArea.classList.add('dragover');
        });

        folderArea.addEventListener('dragleave', () => {
            folderArea.classList.remove('dragover');
        });

        folderArea.addEventListener('drop', async (e) => {
            e.preventDefault();
            folderArea.classList.remove('dragover');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                // ZIP 파일인지 확인
                if (files[0].name.toLowerCase().endsWith('.zip')) {
                    await uploadFolderZip(files[0]);
                } else {
                    Toast.error('형식 오류', 'ZIP 파일만 지원됩니다');
                }
            }
        });
    }
}

async function uploadFolderZip(file) {
    const progressDiv = document.getElementById('upload-progress');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');

    progressDiv.classList.remove('hidden');
    progressFill.style.width = '0%';
    progressText.textContent = 'ZIP 파일 업로드 및 처리 중...';

    const result = await API.uploadFolder(file);

    if (result.success) {
        progressFill.style.width = '100%';
        progressText.textContent = '완료!';
        Toast.success('폴더 업로드 완료', result.message);
        await loadFiles();
        await loadStats();
    } else {
        progressFill.style.width = '0%';
        Toast.error('업로드 실패', result.message);
    }

    setTimeout(() => {
        progressDiv.classList.add('hidden');
    }, 3000);
}

// 자동 동기화 관리자
const FolderSync = {
    isRunning: false,
    timer: null,

    async init() {
        const startBtn = document.getElementById('btn-start-sync');
        const stopBtn = document.getElementById('btn-stop-sync');

        if (startBtn && stopBtn) {
            startBtn.addEventListener('click', () => this.start());
            stopBtn.addEventListener('click', () => this.stop());
        }

        await this.checkStatus();
    },

    async checkStatus() {
        const result = await API.getSyncStatus();
        if (result.success && result.status) {
            this.updateUI(result.status.running);
        }
    },

    async start() {
        const folderPath = document.getElementById('sync-folder-path');
        // 값이 없으면 서버 기본값 사용
        const folder = folderPath ? folderPath.value : null;

        const result = await API.startSync(folder);
        if (result.success) {
            Toast.success('동기화 시작', result.message);
            this.updateUI(true);
        } else {
            Toast.error('실패', result.message);
        }
    },

    async stop() {
        const result = await API.stopSync();
        if (result.success) {
            Toast.success('동기화 중지', result.message);
            this.updateUI(false);
        }
    },

    updateUI(running) {
        this.isRunning = running;

        const indicator = document.getElementById('sync-status-indicator');
        const statusText = document.getElementById('sync-status-text');
        const startBtn = document.getElementById('btn-start-sync');
        const stopBtn = document.getElementById('btn-stop-sync');
        const folderInput = document.getElementById('sync-folder-path');

        if (indicator) {
            indicator.className = running ? 'status-dot running' : 'status-dot';
            indicator.style.backgroundColor = running ? 'var(--success)' : 'var(--text-muted)';
        }

        if (statusText) statusText.textContent = running ? '실시간 동기화 중' : '동기화 중지됨';

        if (startBtn) startBtn.disabled = running;
        if (stopBtn) stopBtn.disabled = !running;
        if (folderInput) folderInput.disabled = running;
    }
};

async function uploadFiles(files) {
    const progressDiv = document.getElementById('upload-progress');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');

    // 파일 필터링
    const validFiles = Array.from(files).filter(f => {
        const ext = f.name.split('.').pop().toLowerCase();
        return ['txt', 'docx', 'pdf'].includes(ext);
    });

    if (validFiles.length === 0) {
        Toast.warning('지원되지 않는 형식', '.txt, .docx, .pdf 파일만 지원됩니다');
        return;
    }

    // 프로그레스 표시
    progressDiv.classList.remove('hidden');
    progressFill.style.width = '0%';
    progressText.textContent = `${validFiles.length}개 파일 업로드 중...`;

    // 업로드 시뮬레이션 (진행률)
    let progress = 0;
    const progressInterval = setInterval(() => {
        if (progress < 90) {
            progress += 10;
            progressFill.style.width = progress + '%';
        }
    }, 200);

    const result = await API.uploadFiles(validFiles);

    clearInterval(progressInterval);
    progressFill.style.width = '100%';

    setTimeout(() => {
        progressDiv.classList.add('hidden');
    }, 1000);

    if (result.success) {
        Toast.success('업로드 완료', result.message);
        await loadFiles();
        await loadStats();

        if (result.failed && result.failed.length > 0) {
            Toast.warning('일부 실패', `${result.failed.length}개 파일 처리 실패`);
        }
    } else {
        Toast.error('업로드 실패', result.message);
    }

    // 파일 입력 리셋
    document.getElementById('file-input').value = '';
}

async function loadFiles() {
    const result = await API.getFiles();
    const tbody = document.getElementById('files-tbody');

    if (!tbody) return;

    if (!result.success || !result.files || result.files.length === 0) {
        tbody.innerHTML = `
            <tr class="empty-row">
                <td colspan="5">로드된 파일이 없습니다</td>
            </tr>
        `;
        return;
    }

    const statusIcons = {
        '완료': '✅',
        '캐시': '💾',
        '실패': '❌',
        '처리중': '⏳',
        '대기': '⏸️'
    };

    tbody.innerHTML = result.files.map(file => `
        <tr>
            <td>${statusIcons[file.status] || '?'} ${file.status}</td>
            <td>
                <span class="file-name-link" onclick="previewFile('${escapeJs(file.name)}')" title="클릭하여 미리보기">
                    ${escapeHtml(file.name)}
                </span>
            </td>
            <td>${formatSize(file.size)}</td>
            <td>${file.chunks}</td>
            <td class="file-actions">
                <button class="btn btn-secondary btn-sm" onclick="previewFile('${escapeJs(file.name)}')" title="미리보기">
                    👁️
                </button>
                <button class="btn btn-secondary btn-sm" onclick="manageTags('${escapeJs(file.name)}')" title="태그 관리">
                    🏷️
                </button>
                <button class="btn btn-secondary btn-sm" onclick="showRevisions('${escapeJs(file.name)}')" title="변경 이력">
                    🕒
                </button>
                <button class="btn btn-danger btn-sm" onclick="deleteFile('${escapeJs(file.name)}')" title="삭제">
                    🗑️
                </button>
            </td>
        </tr>
    `).join('');
}

// 파일 삭제
async function deleteFile(filename) {
    if (!confirm(`"${filename}" 파일을 삭제하시겠습니까?\n\n주의: 삭제 후 인덱스 재처리가 필요할 수 있습니다.`)) {
        return;
    }

    const result = await API.deleteFile(filename);

    if (result.success) {
        Toast.success('파일 삭제', result.message);
        await loadFiles();
        await loadStats();

        if (result.reindex_required && result.remaining_files > 0) {
            Toast.info('안내', '인덱스 재처리를 권장합니다');
        }
    } else {
        Toast.error('삭제 실패', result.message);
    }
}

// 파일 미리보기
async function previewFile(filename) {
    // 기존 모달 제거
    const existingModal = document.getElementById('preview-modal');
    if (existingModal) existingModal.remove();

    // 로딩 표시
    Toast.info('로딩', '파일 내용을 불러오는 중...');

    const result = await API.getFilePreview(filename);

    if (!result.success) {
        Toast.error('미리보기 실패', result.message);
        return;
    }

    // 모달 생성
    const modal = document.createElement('div');
    modal.id = 'preview-modal';
    modal.className = 'modal-overlay';
    modal.innerHTML = `
        <div class="modal-content preview-modal">
            <div class="modal-header">
                <h3>📄 ${escapeHtml(filename)}</h3>
                <button class="modal-close" onclick="closePreviewModal()">✕</button>
            </div>
            <div class="modal-info">
                <span class="info-badge">상태: ${result.status}</span>
                <span class="info-badge">청크: ${result.chunks}개</span>
                <span class="info-badge">전체 길이: ${result.total_length.toLocaleString()}자</span>
                ${result.is_truncated ? '<span class="info-badge warning">일부만 표시됨</span>' : ''}
            </div>
            <div class="modal-body">
                <pre class="preview-content">${escapeHtml(result.content)}</pre>
            </div>
            <div class="modal-footer">
                <button class="btn btn-secondary" onclick="copyToClipboard(\`${escapeJs(result.content)}\`)">
                    📋 복사
                </button>
                <button class="btn btn-primary" onclick="closePreviewModal()">닫기</button>
            </div>
        </div>
    `;

    document.body.appendChild(modal);

    // ESC 키로 닫기
    modal.addEventListener('click', (e) => {
        if (e.target === modal) closePreviewModal();
    });
    document.addEventListener('keydown', handlePreviewEsc);
}

function handlePreviewEsc(e) {
    if (e.key === 'Escape') closePreviewModal();
}

function closePreviewModal() {
    const modal = document.getElementById('preview-modal');
    if (modal) {
        modal.classList.add('modal-closing');
        setTimeout(() => modal.remove(), 200);
    }
    document.removeEventListener('keydown', handlePreviewEsc);
}

// 관리자용 콘텐츠 표시
function showAdminContent() {
    document.querySelectorAll('.admin-only').forEach(el => {
        el.style.display = 'block';
        // 부드러운 등장을 위한 애니메이션
        el.style.opacity = '0';
        el.style.transform = 'translateY(10px)';
        el.style.transition = 'opacity 0.3s, transform 0.3s';

        // requestAnimationFrame을 사용하여 display 변경 후 트랜지션 적용
        requestAnimationFrame(() => {
            el.style.opacity = '1';
            el.style.transform = 'translateY(0)';
        });
    });
}

// 모델 목록 로드
async function loadModels() {
    const result = await API.getModels();
    const select = document.getElementById('model-select');

    if (!select) return;

    if (result.success && result.models) {
        select.innerHTML = result.models.map(model =>
            `<option value="${model}" ${model === result.current ? 'selected' : ''}>${model}</option>`
        ).join('');

        // 현재 선택된 모델이 목록에 없으면 추가 (커스텀 모델 등)
        if (result.current && !result.models.includes(result.current)) {
            const option = document.createElement('option');
            option.value = result.current;
            option.textContent = result.current;
            option.selected = true;
            select.appendChild(option);
        }
    } else {
        select.innerHTML = '<option value="" disabled>모델 목록 로드 실패</option>';
    }

    // 모델 변경 버튼 이벤트
    const changeBtn = document.getElementById('change-model-btn');
    if (changeBtn) {
        // 기존 리스너 제거 방식 대신 onclick 재정의 (간단하게)
        changeBtn.onclick = async () => {
            const selectedModel = select.value;
            if (!selectedModel) return;

            if (!confirm(`모델을 '${selectedModel}'(으)로 변경하시겠습니까?\n서버가 잠시 재시작될 수 있습니다.`)) return;

            changeBtn.disabled = true;
            changeBtn.textContent = '변경 중...';

            const setRes = await API.setModel(selectedModel);

            if (setRes.success) {
                Toast.success('모델 변경', '모델이 성공적으로 변경되었습니다. 서버가 초기화됩니다.');
                // 잠시 후 새로고침
                setTimeout(() => window.location.reload(), 2000);
            } else {
                Toast.error('변경 실패', setRes.message);
                changeBtn.disabled = false;
                changeBtn.textContent = '변경 적용';
            }
        };
    }
}

async function loadStats() {
    const result = await API.getStatus();

    if (!result.success) return;

    const stats = result.stats || {};

    // 기본 통계
    const filesEl = document.getElementById('stat-files');
    const chunksEl = document.getElementById('stat-chunks');
    const sizeEl = document.getElementById('stat-size');

    if (filesEl) filesEl.textContent = stats.files || 0;
    if (chunksEl) chunksEl.textContent = stats.chunks || 0;
    if (sizeEl) sizeEl.textContent = stats.size_formatted || '0 B';

    // 모델 정보
    const modelText = result.model || '-';
    const modelEl = document.getElementById('stat-model');
    if (modelEl) {
        // 모델명이 길면 줄임
        modelEl.textContent = modelText.length > 15 ? modelText.substring(0, 12) + '...' : modelText;
        modelEl.title = modelText;
    }

    // 시스템 메트릭 (CPU, 메모리, 활성 검색)
    const cpuEl = document.getElementById('stat-cpu');
    const memoryEl = document.getElementById('stat-memory');
    const activeEl = document.getElementById('stat-active');

    if (cpuEl && result.cpu_percent !== undefined) {
        cpuEl.textContent = Math.round(result.cpu_percent) + '%';
    }
    if (memoryEl && result.memory_percent !== undefined) {
        memoryEl.textContent = Math.round(result.memory_percent) + '%';
    }
    if (activeEl && result.search_queue) {
        activeEl.textContent = result.search_queue.active || 0;
    }
}

// ============================================================================
// 유틸리티
// ============================================================================
function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function escapeJs(str) {
    if (!str) return '';
    return str.replace(/\\/g, '\\\\')
        .replace(/`/g, '\\`')
        .replace(/\$/g, '\\$')
        .replace(/\n/g, '\\n')
        .replace(/\r/g, '\\r')
        .replace(/'/g, "\\'")
        .replace(/"/g, '\\"');
}

function formatSize(bytes) {
    if (!bytes) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex++;
    }
    return `${size.toFixed(1)} ${units[unitIndex]}`;
}

async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        Toast.success('복사됨', '클립보드에 복사되었습니다');
    } catch (err) {
        // Fallback
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        Toast.success('복사됨', '클립보드에 복사되었습니다');
    }
}

// ============================================================================
// 관리자 인증 모달
// ============================================================================
function showAuthModal() {
    const modal = document.getElementById('auth-modal');
    if (modal) {
        modal.style.display = 'flex';
        const input = document.getElementById('auth-password');
        if (input) {
            input.value = '';
            input.focus();
        }
        // 오류 메시지 숨기기
        const errorEl = document.getElementById('auth-error');
        if (errorEl) errorEl.style.display = 'none';
    }
}

function hideAuthModal() {
    const modal = document.getElementById('auth-modal');
    if (modal) {
        modal.style.display = 'none';
    }
}

async function submitAdminAuth() {
    const passwordInput = document.getElementById('auth-password');
    const errorEl = document.getElementById('auth-error');

    if (!passwordInput) return;

    const password = passwordInput.value;

    const result = await API.adminAuth(password);

    if (result.success) {
        hideAuthModal();
        Toast.success('인증 성공', '관리자 페이지에 접근합니다');

        // 인증 후 콘텐츠 표시
        showAdminContent();

        // 페이지 초기화 계속
        await AppState.checkStatus();
        await loadFiles();
        await loadStats();
        await loadModels();
        setupUpload();

        // 버튼 이벤트 설정
        document.getElementById('refresh-btn')?.addEventListener('click', async () => {
            await loadFiles();
            await loadStats();
            Toast.success('새로고침', '파일 목록을 갱신했습니다');
        });

        document.getElementById('reprocess-btn')?.addEventListener('click', async () => {
            const btn = document.getElementById('reprocess-btn');
            btn.disabled = true;
            btn.textContent = '처리 중...';
            const reprocessResult = await API.reprocessFiles();
            if (reprocessResult.success) {
                Toast.success('재처리 완료', reprocessResult.message);
                await loadFiles();
                await loadStats();
            } else {
                Toast.error('재처리 실패', reprocessResult.message);
            }
            btn.disabled = false;
            btn.textContent = '⚡ 재처리';
        });

        document.getElementById('clear-cache-btn')?.addEventListener('click', async () => {
            if (!confirm('캐시를 삭제하시겠습니까?\n다음 검색 시 인덱스가 재생성됩니다.')) return;
            const cacheResult = await API.clearCache();
            if (cacheResult.success) {
                Toast.success('캐시 삭제', cacheResult.message);
            } else {
                Toast.error('실패', cacheResult.message);
            }
        });

        // 주기적 상태 갱신
        setInterval(async () => {
            await AppState.checkStatus();
            await loadStats();
        }, 10000);
    } else {
        // 오류 표시
        if (errorEl) {
            errorEl.textContent = result.message || '인증 실패';
            errorEl.style.display = 'block';
        }
        passwordInput.value = '';
        passwordInput.focus();
    }
}


// ============================================================================
// 태그 및 리비전 관리
// ============================================================================

async function manageTags(filename) {
    Toast.info('로딩', '태그 정보를 불러오는 중...');
    const result = await API.getFileTags(filename);
    const existingTags = result.success ? result.tags : [];

    // 모달 생성
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.innerHTML = `
        <div class="modal-content" style="max-width: 500px;">
            <div class="modal-header">
                <h3>🏷️ 태그 관리: ${escapeHtml(filename)}</h3>
                <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">✕</button>
            </div>
            <div class="modal-body" style="padding: 20px;">
                <div class="tag-input-container" style="display: flex; gap: 8px; margin-bottom: 16px;">
                    <input type="text" id="tag-input" class="form-input" placeholder="새 태그 입력 (Enter)" style="flex: 1;">
                    <button class="btn btn-secondary" id="btn-auto-tag" title="자동 태그 생성">🤖 자동</button>
                </div>
                <div id="tag-list" style="display: flex; flex-wrap: wrap; gap: 8px; min-height: 50px; padding: 10px; background: var(--bg-input); border-radius: 8px;">
                    <!-- 태그 렌더링 영역 -->
                </div>
            </div>
            <div class="modal-footer" style="padding: 20px; border-top: 1px solid var(--glass-border); display: flex; justify-content: flex-end; gap: 8px;">
                <button class="btn btn-primary" onclick="this.closest('.modal-overlay').remove()">닫기</button>
            </div>
        </div>
    `;

    document.body.appendChild(modal);

    const tagListEl = modal.querySelector('#tag-list');
    const tagInput = modal.querySelector('#tag-input');
    const autoTagBtn = modal.querySelector('#btn-auto-tag');

    let currentTags = [...existingTags];

    function renderTags() {
        tagListEl.innerHTML = currentTags.map(tag => `
            <span class="search-tag" style="background: var(--accent-secondary);">
                ${escapeHtml(tag)}
                <span class="remove-tag" onclick="removeTag('${escapeJs(tag)}')" style="cursor: pointer; margin-left: 6px; opacity: 0.7;">✕</span>
            </span>
        `).join('');

        if (currentTags.length === 0) {
            tagListEl.innerHTML = '<span style="color: var(--text-muted); font-size: 13px;">등록된 태그가 없습니다.</span>';
        }
    }

    window.removeTag = async (tag) => {
        currentTags = currentTags.filter(t => t !== tag);
        renderTags();
        await API.setFileTags(filename, currentTags);
    };

    tagInput.addEventListener('keypress', async (e) => {
        if (e.key === 'Enter') {
            const newTag = tagInput.value.trim();
            if (newTag && !currentTags.includes(newTag)) {
                currentTags.push(newTag);
                tagInput.value = '';
                renderTags();
                await API.setFileTags(filename, currentTags);
            }
        }
    });

    autoTagBtn.addEventListener('click', async () => {
        autoTagBtn.disabled = true;
        autoTagBtn.textContent = '생성 중...';

        const res = await API.autoTagFile(filename);
        if (res.success && res.tags) {
            // 중복 제거 후 병합
            const newTags = res.tags.filter(t => !currentTags.includes(t));
            if (newTags.length > 0) {
                currentTags = [...currentTags, ...newTags];
                renderTags();
                // 실제로 태그를 저장해야 함
                await API.setFileTags(filename, currentTags);
                Toast.success('자동 태그', `${newTags.length}개 태그가 추가되었습니다`);
            } else {
                Toast.info('자동 태그', '추가할 새로운 태그가 없습니다');
            }
        } else {
            Toast.error('실패', res.message || '자동 태그 생성 실패');
        }

        autoTagBtn.disabled = false;
        autoTagBtn.textContent = '🤖 자동';
    });

    renderTags();
    tagInput.focus();
}

async function showRevisions(filename) {
    Toast.info('로딩', '리비전 정보를 불러오는 중...');
    const result = await API.getRevisions(filename);

    const revisions = result.success ? result.revisions : [];

    const modal = document.createElement('div');
    modal.className = 'modal-overlay';

    let rowsHtml = '';
    if (revisions.length === 0) {
        rowsHtml = '<tr><td colspan="3" style="text-align: center; padding: 20px;">변경 이력이 없습니다.</td></tr>';
    } else {
        rowsHtml = revisions.map(rev => `
            <tr>
                <td>${rev.version}</td>
                <td>${rev.date}</td>
                <td>${escapeHtml(rev.comment || '-')}</td>
            </tr>
        `).join('');
    }

    modal.innerHTML = `
        <div class="modal-content" style="max-width: 700px;">
            <div class="modal-header">
                <h3>🕒 변경 이력: ${escapeHtml(filename)}</h3>
                <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">✕</button>
            </div>
            <div class="modal-body" style="padding: 0;">
                <div class="files-table-container">
                    <table class="files-table">
                        <thead>
                            <tr>
                                <th style="width: 80px;">버전</th>
                                <th style="width: 180px;">날짜</th>
                                <th>코멘트</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${rowsHtml}
                        </tbody>
                    </table>
                </div>
            </div>
            <div class="modal-footer" style="padding: 20px; text-align: right;">
                <button class="btn btn-primary" onclick="this.closest('.modal-overlay').remove()">닫기</button>
            </div>
        </div>
    `;

    document.body.appendChild(modal);
}

// ============================================================================
// 초기화
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    // 동적 년도 업데이트
    const yearEl = document.getElementById('current-year');
    if (yearEl) {
        yearEl.textContent = new Date().getFullYear();
    }

    // 메인 페이지인지 관리자 페이지인지 확인
    if (document.querySelector('.search-section')) {
        initSearch();
    } else if (document.querySelector('.admin-section') || document.getElementById('files-tbody')) {
        // 관리자 페이지 초기화
        initAdmin();
    }
});
