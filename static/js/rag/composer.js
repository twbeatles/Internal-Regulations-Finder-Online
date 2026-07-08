/**
 * 채팅 입력 컴포저
 */
export const ChatComposer = {
    init({ onSend, onStop }) {
        this.input = document.getElementById('chat-input');
        this.sendBtn = document.getElementById('chat-send-btn');
        this.stopBtn = document.getElementById('chat-stop-btn');
        this.onSend = onSend;
        this.onStop = onStop;
        if (!this.input || !this.sendBtn) return;

        this.sendBtn.addEventListener('click', () => this._submit());
        this.input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                this._submit();
            }
        });
        if (this.stopBtn) {
            this.stopBtn.addEventListener('click', () => {
                if (typeof this.onStop === 'function') this.onStop();
            });
        }
    },

    _submit() {
        const text = (this.input?.value || '').trim();
        if (!text || typeof this.onSend !== 'function') return;
        this.onSend(text);
    },

    setBusy(busy) {
        if (this.sendBtn) this.sendBtn.disabled = !!busy;
        if (this.stopBtn) this.stopBtn.hidden = !busy;
        if (this.input) this.input.disabled = !!busy;
    },

    clear() {
        if (this.input) this.input.value = '';
    }
};