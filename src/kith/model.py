"""Core data model: Principal, Scope, Observation.

Design doc: docs/DESIGN.md §4. Three record types, one visibility rule.

Observations are the append-only source of truth (P3); everything else is
derived. Scope is the visibility contract attached to every observation (P1).
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Principal ids look like "kind:name" — "agent:planner-7", "human:scarlett",
# "group:deploy-team". Conservative charset so ids are safe in SQL, URLs,
# and log lines without escaping.
_PRINCIPAL_ID_RE = re.compile(r"^[a-z]+:[A-Za-z0-9_.\-]+$")
_CONTEXT_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+:[A-Za-z0-9_.\-]+$|^[A-Za-z0-9_.\-]+$")

PRINCIPAL_KINDS = ("agent", "human", "group")
OBSERVATION_KINDS = ("interaction", "assertion", "affect")


class KithError(ValueError):
    """Base error for kith validation failures."""


def _validate_principal_id(pid: str) -> str:
    if not isinstance(pid, str) or not _PRINCIPAL_ID_RE.match(pid):
        raise KithError(
            f"Invalid principal id {pid!r}. Expected 'kind:name' with kind in "
            f"lowercase letters and name in [A-Za-z0-9_.-], e.g. 'agent:planner-7'."
        )
    kind = pid.split(":", 1)[0]
    if kind not in PRINCIPAL_KINDS:
        raise KithError(
            f"Unknown principal kind {kind!r} in {pid!r}. "
            f"Expected one of {PRINCIPAL_KINDS}."
        )
    return pid


def _validate_context(ctx: str) -> str:
    if not isinstance(ctx, str) or not _CONTEXT_ID_RE.match(ctx):
        raise KithError(
            f"Invalid context {ctx!r}. Expected 'namespace:name' or a bare "
            f"name in [A-Za-z0-9_.-], e.g. 'task:deploy-42'."
        )
    return ctx


@dataclass(frozen=True)
class Scope:
    """Visibility contract for one observation (DESIGN.md §4.4).

    - ``holders``: principals allowed to read. Empty = observer-private
      (the store always treats the observer as an implicit holder).
    - ``contexts``: if non-empty, the observation is additionally restricted
      to readers operating within one of these contexts (AND semantics).

    Grants (explicit shares) are modeled as observations of kind
    'assertion' with payload {"grant": ...} — they widen ``holders`` at
    read time and leave an audit trail. See Store.grant().
    """

    holders: tuple = ()
    contexts: tuple = ()

    def __post_init__(self):
        object.__setattr__(
            self, "holders",
            tuple(_validate_principal_id(h) for h in self.holders),
        )
        object.__setattr__(
            self, "contexts",
            tuple(_validate_context(c) for c in self.contexts),
        )

    def to_json(self) -> str:
        return json.dumps(
            {"holders": list(self.holders), "contexts": list(self.contexts)},
            ensure_ascii=False, sort_keys=True,
        )

    @classmethod
    def from_json(cls, raw: str) -> "Scope":
        d = json.loads(raw) if raw else {}
        return cls(holders=tuple(d.get("holders", ())),
                   contexts=tuple(d.get("contexts", ())))


@dataclass(frozen=True)
class Observation:
    """One append-only record: observer's memory about subject (DESIGN.md §4.2)."""

    observer: str
    subject: str
    kind: str                      # interaction | assertion | affect
    payload: Dict[str, Any]
    context: Optional[str] = None
    scope: Scope = field(default_factory=Scope)
    ts: float = 0.0                # store stamps at insert when 0
    id: str = ""                   # store assigns when empty

    def __post_init__(self):
        _validate_principal_id(self.observer)
        _validate_principal_id(self.subject)
        if self.kind not in OBSERVATION_KINDS:
            raise KithError(
                f"Unknown observation kind {self.kind!r}. "
                f"Expected one of {OBSERVATION_KINDS}."
            )
        if self.context is not None:
            _validate_context(self.context)
        if not isinstance(self.payload, dict):
            raise KithError("payload must be a dict.")
        # payload must survive a JSON round-trip (sqlite storage + audit log)
        try:
            json.dumps(self.payload)
        except (TypeError, ValueError) as e:
            raise KithError(f"payload is not JSON-serializable: {e}") from e
        if not self.id:
            object.__setattr__(self, "id", uuid.uuid4().hex)
        if not self.ts:
            object.__setattr__(self, "ts", time.time())


# ---------------------------------------------------------------------------
# Payload conventions (validated leniently — unknown keys always allowed,
# known keys type-checked so derivers can rely on them)
# ---------------------------------------------------------------------------

def validate_payload(kind: str, payload: Dict[str, Any]) -> None:
    """Check the *conventional* fields for each observation kind.

    interaction: outcome (bool | 'success'|'failure'|'partial'), promised (str)
    assertion:   claim (str), source ('self'|'observed'|'third_party')
    affect:      valence [-1,1], arousal [0,1], label (str) — at least one
    """
    if kind == "interaction":
        outcome = payload.get("outcome", payload.get("delivered"))
        if outcome is not None and not isinstance(outcome, bool) and \
                outcome not in ("success", "failure", "partial"):
            raise KithError(
                "interaction payload: 'outcome'/'delivered' must be bool or "
                "'success'|'failure'|'partial'."
            )
    elif kind == "assertion":
        if "claim" in payload and not isinstance(payload["claim"], str):
            raise KithError("assertion payload: 'claim' must be a string.")
        src = payload.get("source")
        if src is not None and src not in ("self", "observed", "third_party"):
            raise KithError(
                "assertion payload: 'source' must be 'self'|'observed'|'third_party'."
            )
    elif kind == "affect":
        has_signal = False
        if "valence" in payload:
            v = payload["valence"]
            if not isinstance(v, (int, float)) or not -1 <= v <= 1:
                raise KithError("affect payload: 'valence' must be in [-1, 1].")
            has_signal = True
        if "arousal" in payload:
            a = payload["arousal"]
            if not isinstance(a, (int, float)) or not 0 <= a <= 1:
                raise KithError("affect payload: 'arousal' must be in [0, 1].")
            has_signal = True
        if isinstance(payload.get("label"), str):
            has_signal = True
        if not has_signal:
            raise KithError(
                "affect payload needs at least one of: valence, arousal, label."
            )
