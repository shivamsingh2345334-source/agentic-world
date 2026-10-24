"""
agentic_router.py - The L1/L2 tiered routing state machine (LangGraph).

        +-----------+   SIGNAL   +-----------+
 post ->|  BOUNCER  |----------->|   BRAIN   |--> END (analysis attached)
        | Groq/Llama|            |  Gemini   |
        +-----+-----+            +-----------+
              | NOISE
              v
             END (dropped - ~80% of traffic never touches the expensive model)

WHY LANGGRAPH FOR A 2-NODE FLOW?
  The graph is deterministic: the LLM never decides control flow, it only
  fills in a value ("SIGNAL"/"NOISE") and plain Python routes on it. That
  makes behaviour testable and auditable, and adding a Tier 3 later (e.g.
  human-review node, CRM enrichment) is just another node + edge.

FAIL-OPEN vs FAIL-CLOSED
  If the cheap classifier is down, we treat the post as SIGNAL. Wasting a
  Gemini call on spam is cheap; silently dropping a real outage report from
  a paying customer is not.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.config import Settings
from app.llm_clients import build_gemini, build_groq, call_with_retry, limiter_for
from app.models import Analysis

log = logging.getLogger(__name__)

BOUNCER_SYSTEM = """You are a ruthless triage filter for a social-listening product monitoring the brand "{tenant}".
Reply with EXACTLY one word: SIGNAL or NOISE.

SIGNAL = a real person giving feedback, reporting a bug/outage, asking for help or a feature, comparing
products, showing purchase/churn intent, or praising/complaining about the brand.
NOISE = spam, giveaways, crypto/airdrops, follow-for-follow, bots, ads, job spam, or posts that merely
share a name with the brand but are unrelated."""

BRAIN_SYSTEM = """You analyse a social-media mention about the brand "{tenant}" for its customer-success team.
Extract sentiment, intent, urgency (1-5) and a one-sentence neutral summary. Use sentiment 'Bug' for outages,
errors or defects. Be conservative with urgency 5: reserve it for outages or churn threats."""


class RouterState(TypedDict, total=False):
    post: dict
    verdict: Literal["SIGNAL", "NOISE"]
    analysis: dict


@dataclass
class RouterResult:
    post: dict
    verdict: str
    analysis: Analysis | None

    @property
    def is_signal(self) -> bool:
        return self.analysis is not None


# ------------------------------------------------------------- mock heuristics
_NOISE_WORDS = ("giveaway", "airdrop", "crypto", "follow me", "dm me", "promo code", "free followers", "nft")
_BUG_WORDS = ("down", "outage", "502", "503", "500", "error", "failing", "failed", "broken", "crash", "bug", "timeout")
_NEG_WORDS = ("slow", "expensive", "hate", "cancel", "migrate", "switching", "disappointed", "worst")
_POS_WORDS = ("love", "great", "awesome", "amazing", "fast", "thanks", "best")


def _mock_verdict(text: str) -> str:
    t = text.lower()
    if any(w in t for w in _NOISE_WORDS):
        return "NOISE"
    return "SIGNAL"


def _mock_analysis(text: str) -> Analysis:
    t = text.lower()
    if any(w in t for w in _BUG_WORDS):
        return Analysis(sentiment="Bug", intent="bug_report", urgency=5, summary=text[:140])
    if any(w in t for w in _NEG_WORDS):
        intent = "churn_risk" if any(w in t for w in ("cancel", "migrate", "switching")) else "support_request"
        return Analysis(sentiment="Negative", intent=intent, urgency=3, summary=text[:140])
    if any(w in t for w in _POS_WORDS):
        return Analysis(sentiment="Positive", intent="praise", urgency=1, summary=text[:140])
    return Analysis(sentiment="Neutral", intent="other", urgency=2, summary=text[:140])


# --------------------------------------------------------------------- router
class AgenticRouter:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.mock = settings.llm_mode == "mock"
        if not self.mock:
            self._groq = build_groq(settings)
            # with_structured_output => Gemini is forced to return our Pydantic schema,
            # so there is no fragile "parse JSON out of prose" step.
            self._gemini = build_gemini(settings).with_structured_output(Analysis)
        self._groq_limiter = limiter_for("groq", settings.groq_rpm)
        self._gemini_limiter = limiter_for("gemini", settings.gemini_rpm)
        self._graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(RouterState)
        g.add_node("bouncer", self._bouncer_node)
        g.add_node("brain", self._brain_node)
        g.set_entry_point("bouncer")
        # Pure-Python routing on the verdict: deterministic by construction.
        g.add_conditional_edges("bouncer", lambda s: "brain" if s["verdict"] == "SIGNAL" else "drop",
                                {"brain": "brain", "drop": END})
        g.add_edge("brain", END)
        return g.compile()

    # ---- Tier 1 -----------------------------------------------------------
    async def _bouncer_node(self, state: RouterState) -> RouterState:
        post = state["post"]
        if self.mock:
            await asyncio.sleep(self.s.mock_latency_ms / 2000)
            return {"verdict": _mock_verdict(post["text"])}
        msgs = [
            SystemMessage(content=BOUNCER_SYSTEM.format(tenant=post["tenant"])),
            HumanMessage(content=post["text"][:1000]),  # cap tokens; triage never needs a wall of text
        ]
        try:
            resp = await call_with_retry(lambda: self._groq.ainvoke(msgs), limiter=self._groq_limiter, name="groq")
        except Exception:  # noqa: BLE001
            log.warning("Bouncer unavailable -> failing OPEN (SIGNAL) for post %s", post.get("id"))
            return {"verdict": "SIGNAL"}
        answer = str(resp.content).strip().upper()
        return {"verdict": "NOISE" if answer.startswith("NOISE") else "SIGNAL"}  # ambiguous => SIGNAL

    # ---- Tier 2 -----------------------------------------------------------
    async def _brain_node(self, state: RouterState) -> RouterState:
        post = state["post"]
        if self.mock:
            await asyncio.sleep(self.s.mock_latency_ms / 1000)
            return {"analysis": _mock_analysis(post["text"]).model_dump()}
        msgs = [
            SystemMessage(content=BRAIN_SYSTEM.format(tenant=post["tenant"])),
            HumanMessage(content=post["text"][:2000]),
        ]
        # No fail-open here: if Gemini is down we RAISE, the worker leaves the message
        # un-ACKed, and the queue redelivers it later. Nothing is lost.
        analysis = await call_with_retry(lambda: self._gemini.ainvoke(msgs), limiter=self._gemini_limiter, name="gemini")
        if not isinstance(analysis, Analysis):
            analysis = Analysis.model_validate(analysis)
        return {"analysis": analysis.model_dump()}

    # ---- Public API -------------------------------------------------------
    async def process(self, post: dict) -> RouterResult:
        final = await self._graph.ainvoke({"post": post})
        raw = final.get("analysis")
        return RouterResult(post=post, verdict=final["verdict"], analysis=Analysis(**raw) if raw else None)
