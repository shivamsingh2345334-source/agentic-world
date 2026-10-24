"""
llm_clients.py - Shared plumbing for talking to Groq and Gemini safely.

WHY THIS FILE EXISTS
  * Free-tier LLM APIs enforce requests-per-minute (RPM) caps. Without client
    side pacing, a viral spike turns into a wall of HTTP 429s. A tiny async
    rate limiter lets the queue absorb the spike while LLM calls drain at a
    sustainable pace.
  * Transient failures (429/5xx/timeouts) are normal. Retrying with
    exponential backoff + jitter keeps the pipeline self-healing.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Awaitable, Callable, TypeVar

from app.config import Settings

log = logging.getLogger(__name__)
T = TypeVar("T")


class AsyncRateLimiter:
    """Evenly spaces calls at `rpm` requests/minute (a leaky bucket).

    NOTE: Per-process. If you run N worker processes, set RPM = quota / N,
    or move the limiter to Redis (e.g. a Lua token bucket) for a global cap.
    """

    def __init__(self, rpm: int) -> None:
        self._interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_slot = 0.0

    async def acquire(self) -> None:
        if not self._interval:
            return
        # Reserve a time slot under the lock, but SLEEP outside it so other
        # coroutines can reserve their own (later) slots concurrently.
        async with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._interval
        delay = slot - now
        if delay > 0:
            await asyncio.sleep(delay)


_limiters: dict[str, AsyncRateLimiter] = {}


def limiter_for(name: str, rpm: int) -> AsyncRateLimiter:
    return _limiters.setdefault(name, AsyncRateLimiter(rpm))


async def call_with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    limiter: AsyncRateLimiter | None = None,
    attempts: int = 4,
    base_delay: float = 1.0,
    name: str = "llm",
) -> T:
    """Run `fn` with pacing + exponential backoff. Re-raises after the last try."""
    for attempt in range(1, attempts + 1):
        try:
            if limiter:
                await limiter.acquire()
            return await fn()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise many types
            if attempt == attempts:
                log.error("%s failed after %d attempts: %s", name, attempts, exc)
                raise
            # Jitter avoids all workers retrying in lock-step (thundering herd).
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            log.warning("%s attempt %d/%d failed (%s); retrying in %.1fs", name, attempt, attempts, exc, delay)
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable")


def build_groq(settings: Settings):
    """Tier-1 model. max_tokens is tiny because the answer is ONE word -
    fewer output tokens = lower latency and cost."""
    from langchain_groq import ChatGroq

    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set (or use LLM_MODE=mock)")
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0,
        max_tokens=5,
        max_retries=0,  # we own retries (see call_with_retry)
        timeout=15,
    )


def build_gemini(settings: Settings):
    """Tier-2 / summariser model."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set (or use LLM_MODE=mock)")
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0.2,
        max_retries=0,
        timeout=60,
    )
