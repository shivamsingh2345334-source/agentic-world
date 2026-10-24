"""
models.py - Typed contracts shared by the API, the queue and the agents.

WHY: Validating at the edge (FastAPI) means the queue only ever contains
well-formed, tenant-safe payloads, so workers never crash on garbage.
"""
from __future__ import annotations

import re
import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Tenant ids become part of Redis key names ("fsr:stream:<tenant>"). A strict
# slug regex prevents key-injection tricks like "vercel:*" or whitespace.
TENANT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")


class Post(BaseModel):
    """One social-media mention, as delivered by the (mock) crawler."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    tenant: str
    text: str = Field(min_length=1, max_length=5000)
    author: str = "unknown"
    platform: str = "x"
    url: str | None = None
    created_at: float = Field(default_factory=time.time)

    @field_validator("tenant")
    @classmethod
    def _normalise_tenant(cls, v: str) -> str:
        v = v.strip().lower()
        if not TENANT_RE.match(v):
            raise ValueError("tenant must match ^[a-z0-9][a-z0-9_-]{0,39}$")
        return v


class BatchIngestRequest(BaseModel):
    posts: list[Post] = Field(min_length=1, max_length=1000)


class IngestResponse(BaseModel):
    accepted: int
    rejected: int = 0
    rejected_by_tenant: dict[str, int] = Field(default_factory=dict)


Sentiment = Literal["Positive", "Negative", "Bug", "Neutral"]
Intent = Literal[
    "support_request",
    "bug_report",
    "churn_risk",
    "feature_request",
    "purchase_intent",
    "praise",
    "other",
]


class Analysis(BaseModel):
    """Tier-2 (Gemini) structured output. Docstrings/descriptions double as
    the schema instructions the LLM sees, so keep them precise."""

    sentiment: Sentiment = Field(description="Overall sentiment toward the brand. Use 'Bug' for outages, errors or defects.")
    intent: Intent = Field(description="What the author wants or is signalling.")
    urgency: int = Field(description="Integer 1 (ignorable) to 5 (needs a human right now).")
    summary: str = Field(description="One neutral sentence (max 25 words) describing the issue or feedback.")

    @field_validator("urgency")
    @classmethod
    def _clamp(cls, v: int) -> int:
        return max(1, min(5, int(v)))
