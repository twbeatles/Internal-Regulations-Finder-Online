/**
 * SSE 스트림 파서
 */
export const RagStream = {
    async consume(response, handlers = {}) {
        if (!response.ok || !response.body) {
            const text = await response.text();
            throw new Error(text || `HTTP ${response.status}`);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const parts = buffer.split('\n\n');
            buffer = parts.pop() || '';
            for (const part of parts) {
                this._dispatch(part, handlers);
            }
        }
        if (buffer.trim()) {
            this._dispatch(buffer, handlers);
        }
    },

    _dispatch(block, handlers) {
        const lines = block.split('\n');
        let event = 'message';
        let data = '';
        for (const line of lines) {
            if (line.startsWith('event:')) event = line.slice(6).trim();
            if (line.startsWith('data:')) data += line.slice(5).trim();
        }
        if (!data) return;
        let payload;
        try {
            payload = JSON.parse(data);
        } catch {
            payload = data;
        }
        const handler = handlers[event];
        if (typeof handler === 'function') handler(payload);
    }
};