/**
 * 관리자 페이지 ESM 부트스트랩
 */
import * as App from '../legacy/app.js';
import { escapeHtml, formatFileSize } from '../core/utils.js';

Object.assign(window, App, {
    escapeHtml: window.escapeHtml || escapeHtml,
    formatFileSize: window.formatFileSize || formatFileSize,
});