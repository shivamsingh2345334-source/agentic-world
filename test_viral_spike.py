#!/usr/bin/env python3
"""
test_viral_spike.py - Simulates a viral spike and PROVES the fairness claim.

Injects N posts (default 1200): 80% Vercel, 10% Railway, 10% PostHog, shuffled
so they arrive interleaved like real traffic. ~80% of all posts are noise.
Then it watches /v1/stats and reports when each tenant fully drained.

WHAT TO LOOK FOR: Railway and PostHog finish in a fraction of the time Vercel
needs, and their avg latency stays low. In a single shared FIFO queue they
would finish LAST (right after Vercel's 960 posts).

Usage:
    python test_viral_spike.py                       # 1200 posts, then watch
    python test_viral_spike.py --total 300 --reset   # smaller, from a clean slate
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time

import httpx

DISTRIBUTION = {"vercel": 0.80, "railway": 0.10, "posthog": 0.10}

SIGNAL_TEMPLATES = {
    "vercel": [
        "{b} is DOWN again?! My production deploy returns 502 since 10 minutes #outage",
        "Builds on {b} are failing with a timeout error, anyone else seeing this?",
        "Edge functions on {b} have gotten so slow this week, thinking about switching to Cloudflare",
        "Love the new {b} dashboard, deploys are really fast now, thanks!",
        "How do I configure custom domains on {b}? The docs are confusing",
        "{b} pricing for bandwidth is getting expensive, we may cancel our team plan",
    ],
    "railway": [
        "{b} deploy stuck on 'building' for 20 minutes, is something broken?",
        "Moved our Postgres to {b} and it was painless. Great product!",
        "Is there a way to get static outbound IPs on {b}? Need it for a firewall allowlist",
    ],
    "posthog": [
        "{b} session replay is not loading for our project, getting an error",
        "Switching our analytics to {b}, the feature flags are awesome",
        "Feature request: {b} should support custom retention windows for events",
    ],
}
NOISE_TEMPLATES = [
    "GIVEAWAY!! follow me and RT to win free {b} swag + crypto airdrop",
    "dm me for free followers, promo code {b}50 #nft #crypto",
    "Check out my new {b} NFT collection!!! follow me",
    "Hiring now!!! earn $5000/week from home, promo code {b}",
]


def make_posts(total: int, signal_ratio: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    posts: list[dict] = []
    for tenant, share in DISTRIBUTION.items():
        for i in range(round(total * share)):
            is_signal = rng.random() < signal_ratio
            template = rng.choice(SIGNAL_TEMPLATES[tenant] if is_signal else NOISE_TEMPLATES)
            posts.append({
                "tenant": tenant,
                "text": template.format(b=tenant.title()) + f" (#{rng.randint(1000, 9999)})",
                "author": f"user_{rng.randint(1, 50_000)}",
                "platform": rng.choice(["x", "reddit", "hackernews", "linkedin"]),
                "url": f"https://example.com/{tenant}/{i}",
            })
    rng.shuffle(posts)  # arrival order is interleaved, as in the real world
    return posts


async def inject(client: httpx.AsyncClient, posts: list[dict], batch_size: int, concurrency: int) -> dict[str, int]:
    """Send posts in parallel chunks. Returns accepted count per tenant."""
    sem = asyncio.Semaphore(concurrency)
    accepted: dict[str, int] = {t: 0 for t in DISTRIBUTION}

    async def send(chunk: list[dict]) -> None:
        async with sem:
            resp = await client.post("/v1/ingest/batch", json={"posts": chunk})
        if resp.status_code not in (202, 429):
            resp.raise_for_status()
        body = resp.json()
        rejected = body.get("rejected_by_tenant", {})
        for p in chunk:
            accepted[p["tenant"]] += 1
        for t, n in rejected.items():
            accepted[t] -= n

    await asyncio.gather(*(send(posts[i:i + batch_size]) for i in range(0, len(posts), batch_size)))
    return accepted


def print_table(elapsed: float, stats: dict, base: dict, sent: dict) -> None:
    print(f"\n[t+{elapsed:5.1f}s] {'tenant':<9}{'sent':>6}{'done':>7}{'backlog':>9}{'signal':>8}{'noise':>7}{'avg latency':>13}")
    for t, s in sorted(stats["tenants"].items()):
        done = s["processed"] - base.get(t, 0)
        print(f"          {t:<9}{sent.get(t, 0):>6}{done:>7}{s['backlog']:>9}{s['signal']:>8}{s['noise']:>7}{s['avg_latency_ms']:>11}ms")


async def watch(client: httpx.AsyncClient, base: dict, sent: dict, start: float, timeout: float, interval: float) -> None:
    drained_at: dict[str, float] = {}
    while time.monotonic() - start < timeout:
        stats = (await client.get("/v1/stats")).json()
        elapsed = time.monotonic() - start
        for t, n in sent.items():
            done = stats["tenants"].get(t, {}).get("processed", 0) - base.get(t, 0)
            if n and done >= n and t not in drained_at:
                drained_at[t] = elapsed
        print_table(elapsed, stats, base, sent)
        if all(t in drained_at for t, n in sent.items() if n):
            break
        await asyncio.sleep(interval)
    else:
        print("\n⚠️  Timed out waiting for drain (live LLM free tiers are rate limited - try --total 200 or LLM_MODE=mock)")

    print("\n================ FAIRNESS REPORT ================")
    for t, secs in sorted(drained_at.items(), key=lambda kv: kv[1]):
        print(f"  {t:<9} fully processed after {secs:6.1f}s  ({sent[t]} posts)")
    print("=================================================")
    print("Check slack_outbox.jsonl (or worker logs) for the batched summaries.")


async def main() -> int:
    ap = argparse.ArgumentParser(description="Inject a viral spike into the Fair-Share Event Router")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--total", type=int, default=1200)
    ap.add_argument("--signal-ratio", type=float, default=0.20, help="fraction of posts that are relevant")
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reset", action="store_true", help="wipe queue state first (needs ALLOW_ADMIN_RESET=true)")
    ap.add_argument("--no-watch", action="store_true")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--interval", type=float, default=3.0)
    args = ap.parse_args()

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    async with httpx.AsyncClient(base_url=args.url, headers=headers, timeout=30) as client:
        try:
            (await client.get("/health")).raise_for_status()
        except httpx.HTTPError as exc:
            print(f"❌ API not reachable at {args.url}: {exc}\n   Start it with: uvicorn app.main:app --port 8000")
            return 1

        if args.reset:
            r = await client.post("/v1/admin/reset")
            print("🧹 reset:", r.status_code, r.text)

        base = {t: s["processed"] for t, s in (await client.get("/v1/stats")).json()["tenants"].items()}
        posts = make_posts(args.total, args.signal_ratio, args.seed)
        counts = {t: sum(p["tenant"] == t for p in posts) for t in DISTRIBUTION}
        print(f"🌊 Injecting {len(posts)} posts: {counts}")

        start = time.monotonic()
        sent = await inject(client, posts, args.batch_size, args.concurrency)
        took = time.monotonic() - start
        print(f"✅ Ingested {sum(sent.values())} posts in {took:.2f}s ({sum(sent.values()) / took:,.0f} posts/s) — API never touched an LLM")

        if not args.no_watch:
            await watch(client, base, sent, start, args.timeout, args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
