"""Rate-limited HTTP client tests."""

import time

import httpx
import pytest
import respx

from eonet_cascades.data.http import RateLimitedClient


@respx.mock
def test_client_calls_url():
    respx.get("https://example.com/x").mock(return_value=httpx.Response(200, json={"ok": True}))
    client = RateLimitedClient(rate_per_sec=10.0)
    r = client.get("https://example.com/x")
    assert r.json() == {"ok": True}


@respx.mock
def test_client_throttles_to_rate():
    respx.get("https://example.com/y").mock(return_value=httpx.Response(200, json={}))
    client = RateLimitedClient(rate_per_sec=4.0)  # min 0.25s between requests
    t0 = time.monotonic()
    for _ in range(3):
        client.get("https://example.com/y")
    elapsed = time.monotonic() - t0
    # 3 calls at 4/sec: first is free, then 2 * 0.25s = ~0.5s
    assert elapsed >= 0.45, f"expected throttle, got {elapsed:.3f}s"


@respx.mock
def test_client_retries_on_5xx():
    route = respx.get("https://example.com/z").mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(500),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    client = RateLimitedClient(rate_per_sec=100.0, max_retries=3, retry_backoff=0.01)
    r = client.get("https://example.com/z")
    assert r.status_code == 200
    assert route.call_count == 3


@respx.mock
def test_client_raises_after_max_retries():
    respx.get("https://example.com/q").mock(return_value=httpx.Response(503))
    client = RateLimitedClient(rate_per_sec=100.0, max_retries=2, retry_backoff=0.01)
    with pytest.raises(httpx.HTTPStatusError):
        client.get("https://example.com/q")
