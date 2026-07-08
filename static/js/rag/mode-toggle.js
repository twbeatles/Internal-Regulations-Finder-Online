/**
 * RAG / 레거시 검색 모드 전환
 */
export const ModeToggle = {
    STORAGE_KEY: 'search_mode_override',
    current: 'rag',

    async init() {
        const ragSection = document.getElementById('rag-section');
        const legacySection = document.querySelector('.search-section');
        const toggle = document.getElementById('mode-toggle');
        if (!ragSection || !legacySection) return;

        let mode = localStorage.getItem(this.STORAGE_KEY);
        if (!mode) {
            try {
                const res = await fetch('/api/settings/search-mode');
                const data = await res.json();
                mode = data?.data?.search_mode || data?.search_mode || 'rag';
            } catch {
                mode = 'rag';
            }
        }
        this.apply(mode, ragSection, legacySection, toggle);

        if (toggle) {
            toggle.addEventListener('click', () => {
                const next = this.current === 'rag' ? 'legacy' : 'rag';
                localStorage.setItem(this.STORAGE_KEY, next);
                this.apply(next, ragSection, legacySection, toggle);
            });
        }
    },

    apply(mode, ragSection, legacySection, toggle) {
        this.current = mode === 'legacy' ? 'legacy' : 'rag';
        const isRag = this.current === 'rag';
        ragSection.hidden = !isRag;
        legacySection.hidden = isRag;
        if (toggle) {
            toggle.textContent = isRag ? '문서 검색' : 'AI 질문';
            toggle.title = isRag ? '키워드 검색 모드로 전환' : 'RAG 질문 모드로 전환';
            toggle.setAttribute('aria-pressed', String(isRag));
        }
    }
};