"""
queue_manager.py - Tenant-partitioned queueing on Redis Streams (Upstash).

THE CORE IDEA (solves multi-tenant queue starvation)
  A single shared queue is FIFO: when Vercel enqueues 800 posts in a burst,
  Railway's 5 posts wait behind all of them. Instead every tenant owns an
  ISOLATED stream:

        fsr:stream:vercel    [########################...]   <- viral spike
        fsr:stream:railway   [##]                            <- unaffected
        fsr:stream:posthog   [###]                           <- unaffected

  Workers read ALL tenant streams in one XREADGROUP call with COUNT=quantum,
  i.e. "up to <quantum> messages from EACH tenant" per fetch. That is a
  round-robin with a fixed quantum (a simple Deficit-Round-Robin variant):
  a small tenant is served at the same rate as the huge one, and when it
  drains, its share automatically flows to the busy tenant (max-min fairness).

WHY REDIS STREAMS (not LIST/BLPOP)
  Consumer groups give at-least-once delivery: a message stays in the Pending
  Entries List (PEL) until ACKed, so a crashed worker never loses a mention.
  Orphaned messages are recovered with XAUTOCLAIM.

UPSTASH NOTES
  Upstash bills per command, so: tenant list is cached for a few seconds,
  fetches cover all tenants in ONE command, and idle workers back off.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass

import redis.asyncio as aioredis
from redis.exceptions import ResponseError

from app.config import Settings

log = logging.getLogger(__name__)

TENANTS_KEY = "fsr:tenants"
GROUP = "fsr-workers"
_STREAM_PREFIX = "fsr:stream:"


def stream_key(tenant: str) -> str:
    return f"{_STREAM_PREFIX}{tenant}"


def dlq_key(tenant: str) -> str:
    return f"fsr:dlq:{tenant}"


def metrics_key(tenant: str) -> str:
    return f"fsr:metrics:{tenant}"


def create_redis(url: str) -> aioredis.Redis:
    """One shared async client (internally a connection pool) per process."""
    return aioredis.from_url(
        url,
        decode_responses=True,  # str in / str out; we store JSON text anyway
        health_check_interval=30,  # detect half-dead TCP connections to Upstash
        socket_keepalive=True,
    )


@dataclass(frozen=True)
class QueueMessage:
    tenant: str
    msg_id: str
    post: dict
    enqueued_at: float


class QueueManager:
    def __init__(self, redis: aioredis.Redis, settings: Settings) -> None:
        self.redis = redis
        self.settings = settings
        self._groups_ready: set[str] = set()
        self._tenant_cache: list[str] = []
        self._tenant_cache_at = 0.0
        self._rr_offset = 0  # rotates who is listed first, so no tenant is permanently "first"

    # ------------------------------------------------------------------ ingest
    async def enqueue_many(self, posts: list[dict]) -> tuple[int, dict[str, int]]:
        """Append posts to their tenant streams. Returns (accepted, rejected_by_tenant).

        Backpressure: if a tenant's backlog is already at the cap we reject the
        overflow (API -> HTTP 429) instead of letting Redis memory grow without
        bound. Per-tenant caps mean one tenant's flood can't exhaust capacity
        for the others either.
        """
        by_tenant: dict[str, list[dict]] = defaultdict(list)
        for p in posts:
            by_tenant[p["tenant"]].append(p)

        # Round trip 1: current backlog per tenant.
        async with self.redis.pipeline(transaction=False) as pipe:
            for tenant in by_tenant:
                pipe.xlen(stream_key(tenant))
            lengths = await pipe.execute()

        accepted = 0
        rejected: dict[str, int] = {}
        now = f"{time.time():.6f}"

        # Round trip 2: write everything (pipelined => one network hop for N posts).
        async with self.redis.pipeline(transaction=False) as pipe:
            for (tenant, items), current in zip(by_tenant.items(), lengths):
                room = max(0, self.settings.max_backlog_per_tenant - int(current))
                take = items[:room]
                if len(items) > len(take):
                    rejected[tenant] = len(items) - len(take)
                for p in take:
                    pipe.xadd(stream_key(tenant), {"data": json.dumps(p), "enq": now})
                if take:
                    pipe.sadd(TENANTS_KEY, tenant)
                accepted += len(take)
            await pipe.execute()
        return accepted, rejected

    # ---------------------------------------------------------------- tenants
    async def list_tenants(self, use_cache: bool = True) -> list[str]:
        if use_cache and time.monotonic() - self._tenant_cache_at < 2.0:
            return self._tenant_cache
        self._tenant_cache = sorted(await self.redis.smembers(TENANTS_KEY))
        self._tenant_cache_at = time.monotonic()
        return self._tenant_cache

    async def _ensure_group(self, tenant: str) -> None:
        if tenant in self._groups_ready:
            return
        try:
            # id="0" => a group created AFTER messages arrived still sees them.
            await self.redis.xgroup_create(stream_key(tenant), GROUP, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):  # BUSYGROUP = another worker already made it
                raise
        self._groups_ready.add(tenant)

    # ------------------------------------------------------------------ fetch
    async def fetch_fair(self, consumer: str, quantum: int | None = None) -> list[QueueMessage]:
        """Fair-share fetch: up to `quantum` messages from EVERY tenant stream.

        The result is interleaved (vercel, railway, posthog, vercel, ...) so a
        worker also *processes* in round-robin order, not tenant-by-tenant.
        """
        quantum = quantum or self.settings.fair_quantum
        tenants = await self.list_tenants()
        if not tenants:
            return []
        for t in tenants:
            await self._ensure_group(t)

        self._rr_offset = (self._rr_offset + 1) % len(tenants)
        order = tenants[self._rr_offset :] + tenants[: self._rr_offset]
        streams = {stream_key(t): ">" for t in order}  # ">" = only never-delivered msgs

        try:
            # COUNT applies PER STREAM in Redis => this single command IS the round robin.
            result = await self.redis.xreadgroup(GROUP, consumer, streams, count=quantum)
        except ResponseError as exc:
            if "NOGROUP" in str(exc):  # stream/group deleted under us (e.g. admin reset)
                self._groups_ready.clear()
                return []
            raise

        per_tenant: dict[str, list[QueueMessage]] = {}
        for key, entries in result or []:
            tenant = key[len(_STREAM_PREFIX) :]
            msgs: list[QueueMessage] = []
            for mid, fields in entries:  # explicit loop: _parse is async (may dead-letter)
                parsed = await self._parse(tenant, mid, fields)
                if parsed:
                    msgs.append(parsed)
            if msgs:
                per_tenant[tenant] = msgs
        interleaved = itertools.chain.from_iterable(itertools.zip_longest(*per_tenant.values()))
        return [m for m in interleaved if m is not None]

    async def _parse(self, tenant: str, msg_id: str, fields: dict | None) -> QueueMessage | None:
        """Decode one stream entry. Poison messages are dead-lettered, never retried forever."""
        if not fields:  # entry was deleted but still in PEL
            return None
        try:
            return QueueMessage(tenant, msg_id, json.loads(fields["data"]), float(fields.get("enq", time.time())))
        except (KeyError, ValueError, TypeError):
            log.error("Malformed message %s on %s -> DLQ", msg_id, tenant)
            await self.redis.xadd(dlq_key(tenant), {"raw": json.dumps(fields), "reason": "malformed"})
            await self._ack_ids(tenant, [msg_id])
            return None

    # --------------------------------------------------------------- ack / DLQ
    async def _ack_ids(self, tenant: str, ids: list[str]) -> None:
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.xack(stream_key(tenant), GROUP, *ids)
            pipe.xdel(stream_key(tenant), *ids)  # XDEL keeps XLEN == real backlog
            await pipe.execute()

    async def ack(self, msg: QueueMessage) -> None:
        await self._ack_ids(msg.tenant, [msg.msg_id])

    async def dead_letter(self, msg: QueueMessage, reason: str) -> None:
        await self.redis.xadd(dlq_key(msg.tenant), {"data": json.dumps(msg.post), "reason": reason[:500]})
        await self.ack(msg)

    async def delivery_count(self, msg: QueueMessage) -> int:
        """Times Redis has delivered this message (the PEL is our retry counter)."""
        rows = await self.redis.xpending_range(stream_key(msg.tenant), GROUP, min=msg.msg_id, max=msg.msg_id, count=1)
        return int(rows[0]["times_delivered"]) if rows else 1

    async def reclaim_stale(self, consumer: str, count: int = 10) -> list[QueueMessage]:
        """Take over messages whose worker died or failed (idle in PEL too long)."""
        out: list[QueueMessage] = []
        for tenant in await self.list_tenants():
            await self._ensure_group(tenant)
            try:
                res = await self.redis.xautoclaim(
                    stream_key(tenant), GROUP, consumer,
                    min_idle_time=self.settings.reclaim_idle_ms, start_id="0-0", count=count,
                )
            except ResponseError as exc:
                log.warning("XAUTOCLAIM unavailable for %s: %s", tenant, exc)
                continue
            for mid, fields in res[1]:  # res = [next_cursor, entries, (deleted_ids)]
                msg = await self._parse(tenant, mid, fields)
                if msg:
                    out.append(msg)
        return out

    # ---------------------------------------------------------------- metrics
    async def incr_metrics(self, tenant: str, **fields: int) -> None:
        async with self.redis.pipeline(transaction=False) as pipe:
            for name, amount in fields.items():
                pipe.hincrby(metrics_key(tenant), name, int(amount))
            await pipe.execute()

    async def record_result(self, msg: QueueMessage, outcome: str) -> None:
        """outcome: 'signal' | 'noise'. Latency is enqueue -> done: the number
        that exposes starvation (it explodes for the small tenant in a FIFO design)."""
        latency_ms = int((time.time() - msg.enqueued_at) * 1000)
        await self.incr_metrics(msg.tenant, processed=1, **{outcome: 1}, latency_ms_sum=latency_ms)

    async def stats(self) -> dict:
        tenants = await self.list_tenants(use_cache=False)
        async with self.redis.pipeline(transaction=False) as pipe:
            for t in tenants:
                pipe.xlen(stream_key(t))
                pipe.hgetall(metrics_key(t))
            raw = await pipe.execute()
        out: dict[str, dict] = {}
        for i, t in enumerate(tenants):
            backlog, m = raw[2 * i], {k: int(v) for k, v in raw[2 * i + 1].items()}
            processed = m.get("processed", 0)
            out[t] = {
                "backlog": int(backlog),
                "processed": processed,
                "signal": m.get("signal", 0),
                "noise": m.get("noise", 0),
                "alerts_sent": m.get("alerts_sent", 0),
                "summaries_sent": m.get("summaries_sent", 0),
                "avg_latency_ms": round(m.get("latency_ms_sum", 0) / processed) if processed else 0,
            }
        return {"tenants": out, "total_backlog": sum(v["backlog"] for v in out.values())}

    async def reset(self) -> int:
        """Delete every fsr:* key. Demo helper only (guarded by ALLOW_ADMIN_RESET)."""
        keys = [k async for k in self.redis.scan_iter(match="fsr:*", count=500)]
        if keys:
            await self.redis.delete(*keys)
        self._groups_ready.clear()
        self._tenant_cache, self._tenant_cache_at = [], 0.0
        return len(keys)
