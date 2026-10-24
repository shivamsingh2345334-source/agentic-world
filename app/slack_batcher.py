"""
slack_batcher.py - Smart batching + AI summarisation in front of Slack.

THE PROBLEM
  A viral spike produces hundreds of alerts in seconds. Posting each one to a
  Slack webhook (limit ~1 msg/sec) means (a) a Slack channel nobody can read
  and (b) HTTP 429s that can escalate to webhook bans.

THE FIX
  Processed alerts are buffered per tenant in Redis (a LIST). Every
  `batch_window_s` seconds a flusher drains each tenant's buffer:

      alerts <= threshold  ->  send individually (throttled, <= 1/sec)
      alerts >  threshold  ->  ONE Gemini-written summary message

WHY THE BUFFER LIVES IN REDIS
  Any worker process can append, any flusher can drain. LPOP-with-count is
  atomic, so two flushers can never send the same alert twice.

WHY PYTHON WRITES THE NUMBERS, NOT THE LLM
  Counts and the headline are computed deterministically. The LLM only
  writes the "top issues" prose, so it cannot hallucinate "500 users" when
  there were 312.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter

import httpx
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import Settings
from app.llm_clients import build_gemini, call_with_retry, limiter_for
from app.queue_manager import QueueManager

log = logging.getLogger(__name__)

MAX_BUFFER = 20_000  # hard cap per tenant so a stuck flusher can't grow Redis forever
PROMPT_SAMPLE = 40  # alerts shown to Gemini; the rest are represented by the stats
SLACK_TEXT_LIMIT = 3000


def alerts_key(tenant: str) -> str:
    return f"fsr:alerts:{tenant}"


# ------------------------------------------------------------------ Slack I/O
class SlackSender:
    """Mock mode (default): log + append to slack_outbox.jsonl.
    Real mode: POST to an Incoming Webhook with throttling and 429 handling.

    In production each customer has their OWN webhook, so throttling is
    per-tenant. (Here one URL is reused, which is fine for a POC.)
    """

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.mock = settings.slack_webhook_url.strip().lower() in ("", "mock")
        self._client = None if self.mock else httpx.AsyncClient(timeout=10)
        self._locks: dict[str, asyncio.Lock] = {}
        self._next_ok: dict[str, float] = {}
        self.sent = 0

    async def send(self, tenant: str, text: str) -> None:
        text = text[:SLACK_TEXT_LIMIT]
        self.sent += 1
        if self.mock:
            log.info("\n┌─ MOCK SLACK → #%s ─────────────\n%s\n└────────────────────────────────", tenant, text)
            line = json.dumps({"ts": time.time(), "tenant": tenant, "text": text})
            await asyncio.to_thread(self._append, line)
            return
        await self._throttle(tenant)
        for attempt in range(4):
            resp = await self._client.post(self.s.slack_webhook_url, json={"text": text})
            if resp.status_code == 200:
                return
            if resp.status_code == 429:  # Slack tells us exactly how long to back off
                wait = float(resp.headers.get("Retry-After", 2 ** attempt))
                log.warning("Slack 429 for %s; sleeping %.1fs", tenant, wait)
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
        log.error("Slack delivery gave up for tenant %s", tenant)

    @staticmethod
    def _append(line: str) -> None:
        with open("slack_outbox.jsonl", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    async def _throttle(self, tenant: str) -> None:
        lock = self._locks.setdefault(tenant, asyncio.Lock())
        async with lock:
            wait = self._next_ok.get(tenant, 0.0) - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_ok[tenant] = time.monotonic() + self.s.slack_min_interval_s

    async def aclose(self) -> None:
        if self._client:
            await self._client.aclose()


# --------------------------------------------------------------- summarisation
SUMMARY_SYSTEM = """You write incident digests for a customer-success team in Slack mrkdwn.
You are given statistics and a sample of mentions about "{tenant}" from a short time window.
Write ONLY the following (no preamble, max 900 characters):
*Top 3 issues:*
1. <issue> (<approx. how many mentions, based on the sample>)
2. ...
3. ...
*Recommended action:* <one sentence>
Never invent numbers beyond the statistics provided."""


class Summarizer:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.mock = settings.llm_mode == "mock"
        self._gemini = None if self.mock else build_gemini(settings)
        self._limiter = limiter_for("gemini", settings.gemini_rpm)  # shared with Tier 2

    @staticmethod
    def headline(tenant: str, alerts: list[dict], window_s: float) -> str:
        n = len(alerts)
        sent = Counter(a["sentiment"] for a in alerts)
        return (
            f"🚨 *VIRAL SPIKE: {tenant.title()}* — {n} relevant mentions in the last ~{int(window_s)}s\n"
            f"🐞 {sent.get('Bug', 0)} bugs/outages · 👎 {sent.get('Negative', 0)} negative · "
            f"👍 {sent.get('Positive', 0)} positive · avg urgency "
            f"{sum(a['urgency'] for a in alerts) / n:.1f}/5"
        )

    @staticmethod
    def top_sample(alerts: list[dict]) -> list[dict]:
        return sorted(alerts, key=lambda a: a["urgency"], reverse=True)[:PROMPT_SAMPLE]

    def fallback_body(self, alerts: list[dict]) -> str:
        """Deterministic summary used when Gemini is unavailable (and in mock mode).
        We NEVER fall back to sending N individual messages - that's the DDoS we're avoiding."""
        intents = Counter(a["intent"] for a in alerts).most_common(3)
        lines = ["*Top 3 issues:*"]
        for i, (intent, count) in enumerate(intents, 1):
            example = next(a["summary"] for a in self.top_sample(alerts) if a["intent"] == intent)
            lines.append(f"{i}. {intent.replace('_', ' ')} ×{count} — e.g. “{example[:100]}”")
        return "\n".join(lines)

    async def summarise(self, tenant: str, alerts: list[dict]) -> str:
        head = self.headline(tenant, alerts, self.s.batch_window_s)
        if self.mock:
            await asyncio.sleep(self.s.mock_latency_ms / 1000)
            return f"{head}\n{self.fallback_body(alerts)}"
        sample = "\n".join(
            f"- [{a['sentiment']}/{a['intent']}/u{a['urgency']}] {a['summary']}" for a in self.top_sample(alerts)
        )
        msgs = [
            SystemMessage(content=SUMMARY_SYSTEM.format(tenant=tenant)),
            HumanMessage(content=f"Total mentions: {len(alerts)}\nSample (highest urgency first):\n{sample}"),
        ]
        try:
            resp = await call_with_retry(lambda: self._gemini.ainvoke(msgs), limiter=self._limiter, name="gemini-summary")
            content = resp.content
            body = content if isinstance(content, str) else "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
            return f"{head}\n{body.strip()}"
        except Exception:  # noqa: BLE001
            log.exception("Summary LLM failed; using deterministic fallback")
            return f"{head}\n{self.fallback_body(alerts)}"


# -------------------------------------------------------------------- batcher
class SlackBatcher:
    def __init__(self, queue: QueueManager, settings: Settings) -> None:
        self.queue = queue
        self.redis = queue.redis
        self.s = settings
        self.sender = SlackSender(settings)
        self.summarizer = Summarizer(settings)

    async def add_alert(self, tenant: str, alert: dict) -> None:
        """Called by workers for every SIGNAL. Cheap: one pipelined round trip."""
        async with self.redis.pipeline(transaction=False) as pipe:
            pipe.rpush(alerts_key(tenant), json.dumps(alert))
            pipe.ltrim(alerts_key(tenant), -MAX_BUFFER, -1)  # keep newest if ever over cap
            await pipe.execute()

    async def flush_tenant(self, tenant: str) -> int:
        key = alerts_key(tenant)
        n = await self.redis.llen(key)
        if not n:
            return 0
        raw = await self.redis.lpop(key, n)  # atomic: no double-sends across flushers
        alerts = [json.loads(r) for r in (raw or [])]
        if not alerts:
            return 0

        if len(alerts) > self.s.batch_threshold:
            log.info("[%s] %d alerts > %d -> batching into ONE summary", tenant, len(alerts), self.s.batch_threshold)
            await self.sender.send(tenant, await self.summarizer.summarise(tenant, alerts))
            await self.queue.incr_metrics(tenant, summaries_sent=1)
        else:
            for a in alerts:
                link = f" <{a['url']}|source>" if a.get("url") else ""
                await self.sender.send(
                    tenant,
                    f"🔔 *{tenant.title()}* · {a['sentiment']} · {a['intent'].replace('_', ' ')} · "
                    f"urgency {a['urgency']}/5\n{a['summary']}{link}",
                )
            await self.queue.incr_metrics(tenant, alerts_sent=len(alerts))
        return len(alerts)

    async def flush_all(self) -> None:
        tenants = await self.queue.list_tenants(use_cache=False)
        # Tenants flush concurrently: a slow Gemini summary for Vercel must not delay Railway's alerts.
        results = await asyncio.gather(*(self.flush_tenant(t) for t in tenants), return_exceptions=True)
        for t, r in zip(tenants, results):
            if isinstance(r, Exception):
                log.error("Flush failed for %s: %r", t, r)

    async def run(self, stop: asyncio.Event) -> None:
        """Flusher loop. Sleeps in an interruptible way so shutdown is instant."""
        log.info("Batcher started: window=%.0fs threshold=%d", self.s.batch_window_s, self.s.batch_threshold)
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.s.batch_window_s)
            except asyncio.TimeoutError:
                pass
            try:
                await self.flush_all()
            except Exception:  # noqa: BLE001 - the flusher must never die
                log.exception("Flush cycle failed")
        await self.sender.aclose()
