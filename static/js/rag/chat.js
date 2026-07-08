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

    init() {
        const root = document.getElementById('rag-section');
        if (!root) return;

        this.messagesEl = document.getElementById('chat-messages');
        this.citationsEl = document.getElementById('chat-citations');
        this.welcomeEl = document.getElementById('chat-welcome');

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

        ModeToggle.init();
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

    async sendMessage(text) {
        if (!text || text.length < 2) {
            window.Toast?.warning?.('질문은 2자 이상 입력해주세요.');
            return;
        }
        this.appendMessage('user', text);
        ChatComposer.clear();
        ChatComposer.setBusy(true);

        const bubble = this.appendMessage('assistant', '', { streaming: true });
        let answer = '';
        this.citations = [];
        this.abortController = new AbortController();

        try {
            const res = await fetch('/api/rag/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'same-origin',
                signal: this.abortController.signal,
                body: JSON.stringify({
                    message: text,
                    conversation_id: this.conversationId,
                    stream: true
                })
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