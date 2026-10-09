/**
 * 메인 페이지 ESM 부트스트랩
 */
import * as App from '../legacy/app.js';
import { escapeHtml, escapeRegExp, formatFileSize } from '../core/utils.js';
import { RagStream } from '../rag/stream.js';
import { CitationRenderer } from '../rag/citations.js';
import { ModeToggle } from '../rag/mode-toggle.js';
import { ChatComposer } from '../rag/composer.js';
import { ChatManager } from '../rag/chat.js';

// 레거시 번들 전역 노출 (onclick·인라인 핸들러 호환)
Object.assign(window, App, {
    escapeHtml: window.escapeHtml || escapeHtml,
    escapeRegExp: window.escapeRegExp || escapeRegExp,
    formatFileSize: window.formatFileSize || formatFileSize,
    RagStream,
    CitationRenderer,
    ModeToggle,
    ChatComposer,
    ChatManager,
});

function bootstrapRag() {
    if (document.getElementById('rag-section')) {
        ChatManager.init();
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bootstrapRag);
} else {
    bootstrapRag();
}