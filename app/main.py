"""
main.py - FastAPI ingestion edge.

DESIGN RULE: the API does ZERO LLM work. It validates, appends to a Redis
stream and returns 202. That keeps ingestion latency in single-digit
milliseconds even during a viral spike, and it means API pods and worker
pods scale independently.

Run:  uvicorn app.main:app --port 8000
"""
from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError

from app.config import Settings, get_settings, setup_logging
from app.models import BatchIngestRequest, IngestResponse, Post
from app.queue_manager import QueueManager, create_redis

log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    redis = create_redis(settings.redis_url)
    await redis.ping()  # crash at boot (not at first request) if credentials are wrong
    app.state.queue = QueueManager(redis, settings)
    log.info("API ready; Redis connection OK")
    yield
    await redis.aclose()


app = FastAPI(title="Fair-Share Event Router", version="2.0", lifespan=lifespan)


def get_queue(request: Request) -> QueueManager:
    return request.app.state.queue


def require_key(x_api_key: str | None = Header(default=None), settings: Settings = Depends(get_settings)) -> None:
    if settings.ingest_api_key and not secrets.compare_digest(x_api_key or "", settings.ingest_api_key):
        raise HTTPException(status_code=401, detail="invalid API key")


@app.exception_handler(RedisError)
async def redis_down(_: Request, exc: RedisError) -> JSONResponse:
    # Tell callers to retry instead of leaking a stack trace.
    log.error("Redis error: %s", exc)
    return JSONResponse(status_code=503, content={"detail": "queue unavailable"}, headers={"Retry-After": "2"})


def _respond(accepted: int, rejected: dict[str, int]) -> JSONResponse:
    body = IngestResponse(accepted=accepted, rejected=sum(rejected.values()), rejected_by_tenant=rejected)
    if accepted == 0 and rejected:  # everything refused => proper backpressure signal
        return JSONResponse(status_code=429, content=body.model_dump(), headers={"Retry-After": "5"})
    return JSONResponse(status_code=202, content=body.model_dump())


@app.post("/v1/ingest", dependencies=[Depends(require_key)])
async def ingest_one(post: Post, queue: QueueManager = Depends(get_queue)):
    accepted, rejected = await queue.enqueue_many([post.model_dump()])
    return _respond(accepted, rejected)


@app.post("/v1/ingest/batch", dependencies=[Depends(require_key)])
async def ingest_batch(req: BatchIngestRequest, queue: QueueManager = Depends(get_queue)):
    """Bulk endpoint (<=1000 posts). One pipelined Redis write for the whole batch."""
    accepted, rejected = await queue.enqueue_many([p.model_dump() for p in req.posts])
    return _respond(accepted, rejected)


@app.get("/v1/stats", dependencies=[Depends(require_key)])
async def stats(queue: QueueManager = Depends(get_queue)):
    """Per-tenant backlog, throughput and enqueue->done latency (the fairness dashboard)."""
    return await queue.stats()


@app.post("/v1/admin/reset", dependencies=[Depends(require_key)])
async def reset(queue: QueueManager = Depends(get_queue), settings: Settings = Depends(get_settings)):
    if not settings.allow_admin_reset:
        raise HTTPException(status_code=403, detail="set ALLOW_ADMIN_RESET=true to enable (demo only)")
    return {"deleted_keys": await queue.reset()}


@app.get("/health")
async def health(queue: QueueManager = Depends(get_queue)):
    await queue.redis.ping()
    return {"status": "ok"}
