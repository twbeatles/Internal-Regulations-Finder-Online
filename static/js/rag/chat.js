/**
 * RAG 채팅 UI
 */
import { RagStream } from './stream.js';
import { CitationRenderer } from './citations.js';
import { ChatComposer } from './composer.js';
import { ModeToggle } from './mode-toggle.js';

export const ChatManager = {
    conversationId: null,
    abortController: null,
    citations: [],
    history: [],
    fileFilterId: '',

    init() {
        const root = document.getElementById('rag-section');
        if (!root) return;

        this.messagesEl = document.getElementById('chat-messages');
        this.citationsEl = document.getElementById('chat-citations');
        this.welcomeEl = document.getElementById('chat-welcome');
        this.conversationListEl = document.getElementById('chat-conversation-list');
        this.fileFilterEl = document.getElementById('chat-file-filter');
        this.verificationBadgeEl = document.getElementById('chat-verification-badge');

        ChatComposer.init({
            onSend: (text) => this.sendMessage(text),
            onStop: () => this.stopStreaming()
        });

        document.querySelectorAll('.chat-suggestion').forEach((chip) => {
            chip.addEventListener('click', () => {
                const q = chip.dataset.query || chip.textContent;
                if (q) this.sendMessage(q.trim());
            });
        });

        document.getElementById('chat-new-btn')?.addEventListener('click', () => this.startNewConversation());
        this.fileFilterEl?.addEventListener('change', () => {
            this.fileFilterId = this.fileFilterEl.value || '';
        });

        ModeToggle.init();
        this.loadFileOptions();
        this.loadConversations();
    },

    async loadFileOptions() {
        if (!this.fileFilterEl) return;
        try {
            const res = await fetch('/api/files', { credentials: 'same-origin' });
            const data = await res.json();
            const files = data?.data?.files || data?.files || [];
            for (const file of files) {
                const opt = document.createElement('option');
                opt.value = file.file_id || '';
                opt.textContent = file.filename || file.name || file.file_id || '문서';
                if (opt.value) this.fileFilterEl.appendChild(opt);
            }
        } catch (_) {
            /* 파일 목록 없음 */
        }
    },

    async loadConversations() {
        if (!this.conversationListEl) return;
        try {
            const res = await fetch('/api/rag/conversations?limit=30', { credentials: 'same-origin' });
            const data = await res.json();
            const items = data?.data?.conversations || data?.conversations || [];
            this.conversationListEl.innerHTML = '';
            for (const conv of items) {
                const li = document.createElement('li');
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'chat-conversation-item';
                btn.textContent = conv.title || conv.id?.slice(0, 8) || '대화';
                btn.dataset.id = conv.id;
                if (conv.id === this.conversationId) btn.classList.add('active');
                btn.addEventListener('click', () => this.loadConversation(conv.id));
                li.appendChild(btn);
                this.conversationListEl.appendChild(li);
            }
        } catch (_) {
            /* 대화 목록 없음 */
        }
    },

    async loadConversation(conversationId) {
        if (!conversationId) return;
        try {
            const res = await fetch(`/api/rag/conversations/${conversationId}`, { credentials: 'same-origin' });
            const data = await res.json();
            const conv = data?.data || data;
            if (!conv?.messages) return;

            this.conversationId = conversationId;
            this.history = [];
            this.messagesEl.innerHTML = '';
            this.citations = [];
            if (this.citationsEl) this.citationsEl.hidden = true;
            if (this.welcomeEl) this.welcomeEl.hidden = true;

            for (const msg of conv.messages) {
                if (msg.role === 'user' || msg.role === 'assistant') {
                    this.appendMessage(msg.role, msg.content || '', { citations: msg.citations || [] });
                    this.history.push({ role: msg.role, content: msg.content || '' });
                }
            }
            this.loadConversations();
        } catch (err) {
            window.Toast?.error?.(err.message || '대화 불러오기 실패');
        }
    },

    startNewConversation() {
        this.conversationId = null;
        this.history = [];
        this.citations = [];
        if (this.messagesEl) this.messagesEl.innerHTML = '';
        if (this.citationsEl) {
            this.citationsEl.innerHTML = '';
            this.citationsEl.hidden = true;
        }
        if (this.welcomeEl) this.welcomeEl.hidden = false;
        if (this.verificationBadgeEl) this.verificationBadgeEl.hidden = true;
        this.loadConversations();
    },

    appendMessage(role, content, { streaming = false, citations = [] } = {}) {
        if (this.welcomeEl) this.welcomeEl.hidden = true;
        const wrap = document.createElement('div');
        wrap.className = `chat-message chat-message-${role}`;
        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble';
        if (role === 'assistant') {
            bubble.innerHTML = CitationRenderer.linkifyAnswer(content, citations);
            CitationRenderer.bindCitationLinks(bubble, citations);
        } else {
            bubble.textContent = content;
        }
        wrap.appendChild(bubble);
        if (streaming) wrap.dataset.streaming = 'true';
        this.messagesEl.appendChild(wrap);
        this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
        return bubble;
    },

    showVerificationBadge(donePayload) {
        if (!this.verificationBadgeEl) return;
        const score = donePayload?.verification_score;
        if (score == null) {
            this.verificationBadgeEl.hidden = true;
            return;
        }
        const pct = Math.round(Number(score) * 100);
        this.verificationBadgeEl.hidden = false;
        this.verificationBadgeEl.textContent = `근거 일치 ${pct}%`;
        this.verificationBadgeEl.dataset.level = pct >= 60 ? 'good' : pct >= 35 ? 'warn' : 'low';
    },

    async sendMessage(text) {
        if (!text || text.length < 2) {
            window.Toast?.warning?.('질문은 2자 이상 입력해주세요.');
            return;
        }
        this.appendMessage('user', text);
        this.history.push({ role: 'user', content: text });
        ChatComposer.clear();
        ChatComposer.setBusy(true);

        const bubble = this.appendMessage('assistant', '', { streaming: true });
        let answer = '';
        this.citations = [];
        this.abortController = new AbortController();

        const body = {
            message: text,
            conversation_id: this.conversationId,
            stream: true,
            history: this.history.slice(0, -1),
        };
        if (this.fileFilterId) body.filter_file_id = this.fileFilterId;

        try {
            const res = await fetch('/api/rag/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'same-origin',
                signal: this.abortController.signal,
                body: JSON.stringify(body)
            });

            await RagStream.consume(res, {
                meta: (data) => {
                    if (data.conversation_id) this.conversationId = data.conversation_id;
                },
                token: (data) => {
                    answer += data.text || '';
                    bubble.innerHTML = CitationRenderer.linkifyAnswer(answer, this.citations);
                    CitationRenderer.bindCitationLinks(bubble, this.citations);
                    this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
                },
                citation: (data) => {
                    this.citations.push(data);
                    CitationRenderer.renderList(this.citations, this.citationsEl);
                },
                done: (data) => {
                    if (data.conversation_id) this.conversationId = data.conversation_id;
                    if (data.answer) answer = data.answer;
                    this.citations = data.citations || this.citations;
                    bubble.innerHTML = CitationRenderer.linkifyAnswer(answer, this.citations);
                    CitationRenderer.bindCitationLinks(bubble, this.citations);
                    CitationRenderer.renderList(this.citations, this.citationsEl);
                    bubble.closest('.chat-message')?.removeAttribute('data-streaming');
                    this.history.push({ role: 'assistant', content: answer });
                    this.showVerificationBadge(data);
                    this.loadConversations();
                },
                error: (data) => {
                    const msg = data.message || '응답 생성 중 오류가 발생했습니다.';
                    bubble.textContent = msg;
                    window.Toast?.error?.(msg);
                }
            });
        } catch (err) {
            if (err.name !== 'AbortError') {
                bubble.textContent = '질문 처리에 실패했습니다.';
                window.Toast?.error?.(err.message || '네트워크 오류');
            }
        } finally {
            ChatComposer.setBusy(false);
            this.abortController = null;
        }
    },

    stopStreaming() {
        if (this.abortController) {
            this.abortController.abort();
            this.abortController = null;
        }
        ChatComposer.setBusy(false);
    }
};