/**
 * 사내 규정 검색기 - Service Worker v2.0
 * PWA 오프라인 지원 및 캐싱
 */

const CACHE_NAME = 'regulation-search-v2';
const STATIC_CACHE = 'static-v2';
const API_CACHE = 'api-v2';

// 캐시할 정적 리소스
const STATIC_ASSETS = [
    '/',
    '/static/style.css',
    '/static/app.js',
    '/static/manifest.json',
    '/static/icons/icon-192.png',
    '/static/icons/icon-512.png'
];

// API 캐시 패턴 (검색 결과 등)
const API_CACHE_PATTERNS = [
    '/api/status',
    '/api/files/names',
    '/api/categories'
];

// 설치 이벤트 - 정적 리소스 캐싱
self.addEventListener('install', (event) => {
    console.log('[SW] Installing...');
    event.waitUntil(
        caches.open(STATIC_CACHE)
            .then((cache) => {
                console.log('[SW] Caching static assets');
                return cache.addAll(STATIC_ASSETS);
            })
            .then(() => self.skipWaiting())
    );
});

// 활성화 이벤트 - 이전 캐시 정리
self.addEventListener('activate', (event) => {
    console.log('[SW] Activating...');
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames
                    .filter((name) => name !== STATIC_CACHE && name !== API_CACHE)
                    .map((name) => {
                        console.log('[SW] Deleting old cache:', name);
                        return caches.delete(name);
                    })
            );
        }).then(() => self.clients.claim())
    );
});

// Fetch 이벤트 - 네트워크 우선 전략
self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);

    // API 요청 처리
    if (url.pathname.startsWith('/api/')) {
        // 검색은 네트워크만 사용, 기타 API는 네트워크 우선
        if (url.pathname === '/api/search') {
            event.respondWith(fetch(request));
        } else {
            event.respondWith(networkFirst(request, API_CACHE));
        }
        return;
    }

    // 정적 리소스 - 캐시 우선
    if (request.method === 'GET') {
        event.respondWith(cacheFirst(request, STATIC_CACHE));
    }
});

// 캐시 우선 전략
async function cacheFirst(request, cacheName) {
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
        return cachedResponse;
    }

    try {
        const networkResponse = await fetch(request);

        // 성공적인 응답만 캐시
        if (networkResponse.ok) {
            const cache = await caches.open(cacheName);
            cache.put(request, networkResponse.clone());
        }

        return networkResponse;
    } catch (error) {
        console.error('[SW] Fetch failed:', error);

        // 오프라인 폴백 페이지 (옵션)
        if (request.destination === 'document') {
            return caches.match('/');
        }

        throw error;
    }
}

// 네트워크 우선 전략
async function networkFirst(request, cacheName) {
    try {
        const networkResponse = await fetch(request);

        // 성공적인 응답은 캐시 업데이트
        if (networkResponse.ok) {
            const cache = await caches.open(cacheName);
            cache.put(request, networkResponse.clone());
        }

        return networkResponse;
    } catch (error) {
        console.log('[SW] Network failed, trying cache:', request.url);
        const cachedResponse = await caches.match(request);
        if (cachedResponse) {
            return cachedResponse;
        }
        throw error;
    }
}

// 푸시 알림 수신
self.addEventListener('push', (event) => {
    if (!event.data) return;

    const data = event.data.json();
    const options = {
        body: data.body || '새 업데이트가 있습니다',
        icon: '/static/icons/icon-192.png',
        badge: '/static/icons/icon-192.png',
        tag: data.tag || 'regulation-notification',
        data: data.url || '/'
    };

    event.waitUntil(
        self.registration.showNotification(
            data.title || '사내 규정 검색기',
            options
        )
    );
});

// 알림 클릭 처리
self.addEventListener('notificationclick', (event) => {
    event.notification.close();

    event.waitUntil(
        clients.matchAll({ type: 'window' }).then((windowClients) => {
            // 이미 열린 창이 있으면 포커스
            for (const client of windowClients) {
                if (client.url === event.notification.data && 'focus' in client) {
                    return client.focus();
                }
            }
            // 새 창 열기
            if (clients.openWindow) {
                return clients.openWindow(event.notification.data);
            }
        })
    );
});

console.log('[SW] Service Worker loaded');
