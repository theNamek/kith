"""Default relationship derivers (DESIGN.md §5).

Relationship state is COMPUTED from observations, never stored as authority
(P3). Each deriver is a pure function over an observation list, returns a
value plus the observation ids it used (provenance), and is replaceable via
the Deriver protocol — bring your own psychology.

Defaults are chosen for defensibility, not novelty:
- trust: asymmetric update (negative events hit ~2x harder — loss-aversion
  asymmetry is one of the most replicated findings on trust), exponential
  time decay toward a neutral prior
- reliability: Laplace-smoothed promise/delivery ratio over a recent window
- sentiment: exponentially weighted moving average over affect valence
- capabilities: assertion ledger with per-claim provenance and confirmations

No deriver here claims psychological validity. The claim is AUDITABILITY:
every number can explain itself down to the observations that produced it.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .model import Observation

NEUTRAL_TRUST = 0.5
TRUST_HALF_LIFE_DAYS = 30.0     # decay toward neutral with this half-life
NEGATIVITY_WEIGHT = 2.0         # negative outcomes count this much harder
TRUST_STEP = 0.15               # per-event max nudge
RELIABILITY_WINDOW = 20         # most recent promise-bearing interactions
SENTIMENT_ALPHA = 0.3           # EWMA weight of the newest affect reading


class Deriver(Protocol):
    """One derived dimension of a relationship."""

    name: str

    def derive(self, observations: List[Observation],
               now: Optional[float] = None) -> Tuple[Any, List[str]]:
        """Return (value, provenance_observation_ids)."""
        ...


def _outcome_sign(obs: Observation) -> Optional[float]:
    """interaction outcome → +1 (good), -1 (bad), 0 (partial), None (n/a)."""
    p = obs.payload
    outcome = p.get("outcome", p.get("delivered"))
    if outcome is None:
        return None
    if isinstance(outcome, bool):
        return 1.0 if outcome else -1.0
    return {"success": 1.0, "failure": -1.0, "partial": 0.0}.get(outcome)


class TrustDeriver:
    name = "trust"

    def derive(self, observations, now=None):
        now = now or time.time()
        trust = NEUTRAL_TRUST
        used: List[str] = []
        events = [(o, _outcome_sign(o)) for o in observations
                  if o.kind == "interaction"]
        events = [(o, s) for o, s in events if s is not None]
        if not events:
            return NEUTRAL_TRUST, []
        prev_ts = events[0][0].ts
        for obs, sign in events:
            # decay accumulated trust toward neutral over the gap
            gap_days = max(0.0, (obs.ts - prev_ts) / 86400.0)
            decay = 0.5 ** (gap_days / TRUST_HALF_LIFE_DAYS)
            trust = NEUTRAL_TRUST + (trust - NEUTRAL_TRUST) * decay
            # asymmetric update
            step = TRUST_STEP * (NEGATIVITY_WEIGHT if sign < 0 else 1.0)
            trust = min(1.0, max(0.0, trust + sign * step))
            prev_ts = obs.ts
            used.append(obs.id)
        # final decay from last event to now
        gap_days = max(0.0, (now - prev_ts) / 86400.0)
        decay = 0.5 ** (gap_days / TRUST_HALF_LIFE_DAYS)
        trust = NEUTRAL_TRUST + (trust - NEUTRAL_TRUST) * decay
        return round(trust, 4), used


class ReliabilityDeriver:
    name = "reliability"

    def derive(self, observations, now=None):
        used: List[str] = []
        signs: List[float] = []
        for o in observations:
            if o.kind != "interaction":
                continue
            s = _outcome_sign(o)
            if s is None:
                continue
            signs.append(s)
            used.append(o.id)
        signs = signs[-RELIABILITY_WINDOW:]
        used = used[-RELIABILITY_WINDOW:]
        if not signs:
            return None, []   # no evidence — explicitly unknown, not 0.5
        delivered = sum(1 for s in signs if s > 0)
        partial = sum(0.5 for s in signs if s == 0)
        # Laplace smoothing: one virtual success + one virtual failure
        score = (delivered + partial + 1) / (len(signs) + 2)
        return round(score, 4), used


class SentimentDeriver:
    name = "sentiment"

    def derive(self, observations, now=None):
        ewma: Optional[float] = None
        used: List[str] = []
        recent: List[float] = []
        for o in observations:
            if o.kind != "affect":
                continue
            v = o.payload.get("valence")
            if v is None:
                continue
            ewma = v if ewma is None else \
                SENTIMENT_ALPHA * v + (1 - SENTIMENT_ALPHA) * ewma
            recent.append(v)
            used.append(o.id)
        if ewma is None:
            return None, []
        trend = 0.0
        if len(recent) >= 4:
            half = len(recent) // 2
            trend = (sum(recent[half:]) / len(recent[half:])
                     - sum(recent[:half]) / len(recent[:half]))
        return {"valence": round(ewma, 4), "trend": round(trend, 4)}, used


class CapabilitiesDeriver:
    name = "capabilities"

    def derive(self, observations, now=None):
        claims: Dict[str, Dict[str, Any]] = {}
        used: List[str] = []
        for o in observations:
            if o.kind != "assertion":
                continue
            claim = o.payload.get("claim")
            if not claim:
                continue
            entry = claims.setdefault(claim, {
                "claim": claim, "sources": [], "confirmations": 0,
            })
            src = o.payload.get("source", "self")
            entry["sources"].append(src)
            if src in ("observed", "third_party"):
                entry["confirmations"] += 1
            used.append(o.id)
        return list(claims.values()), used


DEFAULT_DERIVERS: List[Deriver] = [
    TrustDeriver(), ReliabilityDeriver(), SentimentDeriver(),
    CapabilitiesDeriver(),
]
