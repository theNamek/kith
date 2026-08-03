"""RelationshipView: derived, explainable relationship state (DESIGN.md §4.3).

Views are materialized on read from scoped observations — the reader only
ever derives from what they are allowed to see (P1 composes with P3: you
cannot leak a hidden observation through a trust score computed over it).

A reader with no visible observations about a subject gets a NEUTRAL view,
indistinguishable from "never met" — never an error revealing that hidden
observations exist.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .derivers import DEFAULT_DERIVERS, Deriver
from .model import Observation


class RelationshipView:
    """One observer's derived model of one subject."""

    def __init__(self, observer: str, subject: str,
                 observations: List[Observation],
                 derivers: Optional[List[Deriver]] = None,
                 now: Optional[float] = None):
        self.observer = observer
        self.subject = subject
        self._observations = observations
        self._now = now or time.time()
        self._values: Dict[str, Any] = {}
        self._provenance: Dict[str, List[str]] = {}
        for d in (derivers or DEFAULT_DERIVERS):
            value, used = d.derive(observations, now=self._now)
            self._values[d.name] = value
            self._provenance[d.name] = used

    @classmethod
    def build(cls, store, observer: str, subject: str,
              context: Optional[str] = None,
              derivers: Optional[List[Deriver]] = None) -> "RelationshipView":
        # The ONLY data source is the scoped read path (P1).
        obs = store.observations(observer, subject=subject,
                                 reader_context=context)
        return cls(observer, subject, obs, derivers=derivers)

    # -- derived dimensions --------------------------------------------------

    @property
    def trust(self) -> float:
        return self._values.get("trust")

    @property
    def reliability(self) -> Optional[float]:
        return self._values.get("reliability")

    @property
    def sentiment(self) -> Optional[Dict[str, float]]:
        return self._values.get("sentiment")

    @property
    def capabilities(self) -> List[Dict[str, Any]]:
        return self._values.get("capabilities", [])

    # -- introspection ---------------------------------------------------------

    def history(self, k: int = 5) -> List[Observation]:
        """Last k observations this view was derived from."""
        return self._observations[-k:]

    def explain(self) -> Dict[str, Any]:
        """Full provenance: which observations produced which value."""
        return {
            "observer": self.observer,
            "subject": self.subject,
            "derived_at": self._now,
            "observation_count": len(self._observations),
            "dimensions": {
                name: {
                    "value": self._values[name],
                    "derived_from": self._provenance[name],
                }
                for name in self._values
            },
        }

    def __repr__(self) -> str:
        return (f"RelationshipView({self.observer!r} -> {self.subject!r}, "
                f"trust={self.trust}, reliability={self.reliability}, "
                f"n={len(self._observations)})")
