"""Rate-limited HTTP client with retries."""

from __future__ import annotations

import time

import httpx


class RateLimitedClient:
    """Synchronous httpx wrapper enforcing a minimum interval between requests.

    Retries on 5xx and connection errors with simple linear backoff.
    """

    def __init__(
        self,
        rate_per_sec: float = 1.0,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        timeout: float = 30.0,
    ) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be > 0")
        self._min_interval = 1.0 / rate_per_sec
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)
        self._last_call = 0.0

    def get(self, url: str, **kwargs) -> httpx.Response:
        return self._request("GET", url, **kwargs)

    def stream_text(self, url: str, **kwargs) -> httpx.Response:
        return self._request("GET", url, **kwargs)

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        self._throttle()
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                r = self._client.request(method, url, **kwargs)
                if r.status_code >= 500:
                    r.raise_for_status()
                r.raise_for_status()
                return r
            except (httpx.HTTPStatusError, httpx.TransportError) as e:
                last_exc = e
                if attempt + 1 < self._max_retries:
                    time.sleep(self._retry_backoff * (attempt + 1))
                    continue
                raise
        raise RuntimeError("unreachable") from last_exc

    def _throttle(self) -> None:
        now = time.monotonic()
        wait = self._min_interval - (now - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def close(self) -> None:
        self._client.close()
