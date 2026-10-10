"""HTTP JSON 請求：逾時、指數退避重試、清楚的錯誤訊息。"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable

import requests

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}

_local = threading.local()


class ApiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _session() -> requests.Session:
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        _local.session = s
    return s


def post_json(
    url: str,
    body: dict,
    headers: dict | None = None,
    timeout: float = 120.0,
    max_retries: int = 5,
    check_cancel: Callable[[], None] | None = None,
) -> dict | list:
    headers = {"Content-Type": "application/json", **(headers or {})}
    attempt = 0
    while True:
        if check_cancel:
            check_cancel()
        try:
            resp = _session().post(url, json=body, headers=headers, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt >= max_retries:
                raise ApiError(f"無法連線到 {url}：{e}") from e
            _sleep(attempt, None, check_cancel)
            attempt += 1
            continue
        if resp.status_code < 300:
            try:
                return resp.json()
            except ValueError as e:
                raise ApiError(f"{url} 回傳的不是 JSON：{resp.text[:300]}", resp.status_code) from e
        if resp.status_code in RETRY_STATUS and attempt < max_retries:
            _sleep(attempt, resp.headers.get("Retry-After"), check_cancel)
            attempt += 1
            continue
        raise ApiError(f"{url} 回應 HTTP {resp.status_code}：{resp.text[:500]}", resp.status_code)


def _sleep(attempt: int, retry_after: str | None, check_cancel: Callable[[], None] | None) -> None:
    delay = min(60.0, 2.0**attempt) + random.random()
    if retry_after:
        try:
            delay = max(delay, min(120.0, float(retry_after)))
        except ValueError:
            pass
    end = time.monotonic() + delay
    while time.monotonic() < end:
        if check_cancel:
            check_cancel()
        time.sleep(min(0.5, end - time.monotonic()) if end > time.monotonic() else 0)


def bearer(api_key: str | None) -> dict:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}
