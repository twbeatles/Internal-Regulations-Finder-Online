# -*- coding: utf-8 -*-
from __future__ import annotations

from functools import wraps
from typing import Any, Callable, TypeVar, cast

from flask import jsonify, session

TFunc = TypeVar("TFunc", bound=Callable[..., Any])


def admin_required(fn: TFunc) -> TFunc:
    """Require session-based admin authentication for state-changing endpoints."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any):
        if not session.get("admin_authenticated", False):
            return jsonify(
                {
                    "success": False,
                    "message": "관리자 인증이 필요합니다",
                    "error_code": "ADMIN_AUTH_REQUIRED",
                }
            ), 401
        return fn(*args, **kwargs)

    return cast(TFunc, wrapper)

