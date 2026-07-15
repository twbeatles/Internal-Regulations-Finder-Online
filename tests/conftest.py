# -*- coding: utf-8 -*-
"""공통 pytest fixture."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_shared_limiters():
    """모듈 전역 RateLimiter 상태를 테스트 간 격리."""
    try:
        from app.routes.api_system import _admin_auth_limiter

        _admin_auth_limiter.reset()
    except Exception:
        pass
    try:
        from app.services.search import rate_limiter, search_queue

        rate_limiter.reset()
    except Exception:
        pass
    yield
    try:
        from app.routes.api_system import _admin_auth_limiter

        _admin_auth_limiter.reset()
    except Exception:
        pass
