/**
 * 인용 렌더링
 */
import { escapeHtml } from '../core/utils.js';

export const CitationRenderer = {
    renderList(citations, container) {
        if (!container) return;
        container.innerHTML = '';
        if (!citations || !citations.length) {
            container.hidden = true;
            return;
        }
        container.hidden = false;
        const title = document.createElement('h4');
        title.className = 'citations-title';
        title.textContent = '참고 규정';
        container.appendChild(title);

        citations.forEach((c) => {
            const item = document.createElement('button');
            item.type = 'button';
            item.className = 'citation-item';
            const label = `[${c.index}] ${c.source || ''} ${c.article_no || ''}`.trim();
            item.textContent = label;
            item.title = (c.excerpt || '').slice(0, 200);
            item.addEventListener('click', () => this.openCitation(c));
            container.appendChild(item);
        });
    },

    linkifyAnswer(text, citations) {
        if (!text) return '';
        const safe = escapeHtml(text);
        return safe.replace(/\[(\d+)\]/g, (match, num) => {
            const idx = parseInt(num, 10);
            const cite = (citations || []).find((c) => c.index === idx);
            if (!cite) return match;
            return `<button type="button" class="cite-link" data-cite-index="${idx}">${match}</button>`;
        });
    },

    bindCitationLinks(root, citations) {
        if (!root) return;
        root.querySelectorAll('.cite-link').forEach((btn) => {
            btn.addEventListener('click', () => {
                const idx = parseInt(btn.dataset.citeIndex, 10);
                const cite = (citations || []).find((c) => c.index === idx);
                if (cite) this.openCitation(cite);
            });
        });
    },

    openCitation(citation) {
        if (typeof ReaderMode !== 'undefined' && ReaderMode.openCitation) {
            ReaderMode.openCitation(citation);
            return;
        }
        if (typeof ReaderMode !== 'undefined' && citation.file_id && ReaderMode.togglePreview) {
            ReaderMode.openItem({
                source: citation.source,
                file_id: citation.file_id,
                content: citation.excerpt || ''
            });
            ReaderMode.togglePreview();
        }
    }
};