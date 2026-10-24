"""
worker.py - asyncio worker pool doing fair-share processing.

    python -m app.worker --workers 5

HOW FAIRNESS WORKS HERE
  Each worker loops: fetch_fair() -> up to `quantum` messages from EVERY
  tenant (see queue_manager.py) -> process them in interleaved order -> ack.
  Because all 5 workers pull the same way, every tenant gets an equal slice
  of total capacity no matter how deep Vercel's stream is.

WHY asyncio (not threads/processes)?
  The work is ~100% network wait (Groq, Gemini, Redis). One event loop
  multiplexes dozens of in-flight calls cheaply. Scale out by running more
  processes (set GROQ_RPM / GEMINI_RPM = quota / processes).

RELIABILITY (at-least-once)
  Messages are ACKed only AFTER success. On failure they stay in the Pending
  list; a reclaimer re-delivers them after `reclaim_idle_ms`; after
  `max_delivery_attempts` they move to the dead-letter stream (fsr:dlq:*).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import socket
import time

from redis.exceptions import RedisError

from app.agentic_router import AgenticRouter, RouterResult
from app.config import Settings, get_settings, setup_logging
from app.queue_manager import QueueManager, QueueMessage, create_redis
from app.slack_batcher import SlackBatcher

log = logging.getLogger("worker")


class FairShareWorker:
    def __init__(self, name: str, queue: QueueManager, router: AgenticRouter, batcher: SlackBatcher,
                 settings: Settings, stop: asyncio.Event, reclaimer: bool = False) -> None:
        self.name, self.queue, self.router, self.batcher = name, queue, router, batcher
        self.s, self.stop, self.reclaimer = settings, stop, reclaimer
        self._last_reclaim = time.monotonic()

    async def _sleep(self, seconds: float) -> None:
        """Interruptible sleep so Ctrl+C doesn't wait out the idle backoff."""
        try:
            await asyncio.wait_for(self.stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def run(self) -> None:
        idle = 0.2
        log.info("%s up", self.name)
        while not self.stop.is_set():
            try:
                msgs = await self._next_batch()
            except RedisError as exc:
                log.error("%s: Redis error (%s); backing off", self.name, exc)
                await self._sleep(2.0)
                continue
            if not msgs:
                # Exponential idle backoff: an idle fleet of 5 workers polling every 200ms
                # would burn ~25 commands/sec of the Upstash quota for nothing.
                await self._sleep(idle)
                idle = min(idle * 2, self.s.idle_max_sleep_s)
                continue
            idle = 0.2
            for msg in msgs:
                await self._handle(msg)
        log.info("%s stopped", self.name)

    async def _next_batch(self) -> list[QueueMessage]:
        if self.reclaimer and time.monotonic() - self._last_reclaim > self.s.reclaim_interval_s:
            self._last_reclaim = time.monotonic()
            stale = await self.queue.reclaim_stale(self.name)
            if stale:
                log.warning("%s reclaimed %d orphaned messages", self.name, len(stale))
                return stale
        return await self.queue.fetch_fair(self.name)

    async def _handle(self, msg: QueueMessage) -> None:
        try:
            result: RouterResult = await self.router.process(msg.post)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            await self._on_failure(msg, exc)
            return
        try:
            if result.is_signal:
                a = result.analysis
                await self.batcher.add_alert(msg.tenant, {
                    "post_id": msg.post.get("id"),
                    "url": msg.post.get("url"),
                    "author": msg.post.get("author"),
                    "sentiment": a.sentiment,
                    "intent": a.intent,
                    "urgency": a.urgency,
                    "summary": a.summary,
                })
            # Buffer the alert BEFORE acking: a crash in between causes a duplicate
            # alert (acceptable) rather than a lost one (not acceptable).
            await self.queue.ack(msg)
            await self.queue.record_result(msg, "signal" if result.is_signal else "noise")
        except RedisError as exc:
            log.error("%s: could not finalise %s (%s); will be redelivered", self.name, msg.msg_id, exc)

    async def _on_failure(self, msg: QueueMessage, exc: Exception) -> None:
        try:
            attempts = await self.queue.delivery_count(msg)
            if attempts >= self.s.max_delivery_attempts:
                log.error("%s: %s failed %d times -> DLQ (%s)", self.name, msg.msg_id, attempts, exc)
                await self.queue.dead_letter(msg, repr(exc))
                await self.queue.incr_metrics(msg.tenant, failed=1)
            else:
                log.warning("%s: %s failed (attempt %d): %s - left pending for retry", self.name, msg.msg_id, attempts, exc)
        except RedisError:
            log.exception("Failure bookkeeping failed for %s", msg.msg_id)


class WorkerPool:
    def __init__(self, settings: Settings, num_workers: int) -> None:
        self.s, self.n = settings, num_workers

    async def run(self) -> None:
        redis = create_redis(self.s.redis_url)
        await redis.ping()  # fail fast on bad credentials
        queue = QueueManager(redis, self.s)
        router = AgenticRouter(self.s)
        batcher = SlackBatcher(queue, self.s)

        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set)  # graceful: finish in-flight messages
            except NotImplementedError:  # Windows
                signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))

        prefix = f"{socket.gethostname()}-{os.getpid()}"
        workers = [
            FairShareWorker(f"{prefix}-w{i}", queue, router, batcher, self.s, stop, reclaimer=(i == 0))
            for i in range(self.n)
        ]
        log.info("Starting %d workers (mode=%s, quantum=%d)", self.n, self.s.llm_mode, self.s.fair_quantum)
        try:
            await asyncio.gather(*(w.run() for w in workers), batcher.run(stop))
        finally:
            await redis.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair-Share Event Router worker pool")
    parser.add_argument("--workers", type=int, default=None, help="override NUM_WORKERS")
    args = parser.parse_args()
    settings = get_settings()
    setup_logging(settings.log_level)
    asyncio.run(WorkerPool(settings, args.workers or settings.num_workers).run())


if __name__ == "__main__":
    main()
