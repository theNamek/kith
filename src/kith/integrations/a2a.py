"""A2A (Agent2Agent) profile: derive kith observations from protocol traffic.

A2A gives multi-agent systems a standard wire format for delegation
(Task lifecycle) and identity (AgentCard). This module is the passive
observation bridge: feed it the A2A objects your client already handles,
and relationship memory accrues without bespoke instrumentation.

Mappings (against the v1 spec, a2aproject/A2A `specification/a2a.proto`):

Task terminal states -> interaction outcome
    completed            -> delivered True
    failed / rejected    -> delivered False
    canceled             -> no observation (withdrawn work is not a
                            reliability signal for the remote agent)
    non-terminal states  -> no observation (nothing happened yet)

AgentCard skills -> capability assertions with source="self"
    An AgentCard is the remote agent's OWN claim about itself. kith's
    capabilities deriver keeps self-claims separate from confirmations,
    so a card never inflates a track record — completed tasks do.

Identity: the A2A side of a dyad gets a principal id derived from the
card/task by the CALLER's runtime (P2) — pass it in; never let a model
supply it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..model import Observation
from ..store import BoundPrincipal

# JSON wire form uses kebab/lower strings; proto uses enum names.
_STATE_ALIASES = {
    "completed": "completed", "task_state_completed": "completed",
    "failed": "failed", "task_state_failed": "failed",
    "rejected": "rejected", "task_state_rejected": "rejected",
    "canceled": "canceled", "cancelled": "canceled",
    "task_state_canceled": "canceled",
}

_OUTCOME = {"completed": True, "failed": False, "rejected": False}


def _task_state(task: Dict[str, Any]) -> Optional[str]:
    status = task.get("status") or {}
    raw = status.get("state") or task.get("state") or ""
    return _STATE_ALIASES.get(str(raw).strip().lower())


def _task_context(task: Dict[str, Any]) -> Optional[str]:
    ctx = task.get("contextId") or task.get("context_id")
    if not ctx:
        return None
    import re
    return "a2a:" + re.sub(r"[^A-Za-z0-9_.-]", "-", str(ctx))


def _task_label(task: Dict[str, Any]) -> str:
    """Short label for what was asked — first user message text if present."""
    for msg in task.get("history") or []:
        for part in msg.get("parts") or []:
            text = part.get("text")
            if text:
                return str(text)[:200]
    return f"a2a task {task.get('id', '?')}"


def observe_task(me: BoundPrincipal, counterpart: str,
                 task: Dict[str, Any]) -> Optional[Observation]:
    """Record the outcome of a terminal A2A task against ``counterpart``.

    Call this when your A2A client sees a task reach a terminal state
    (e.g. in the tasks/get poll loop or the streaming status handler).
    Returns the Observation, or None when the state carries no signal
    (non-terminal, canceled, or unrecognized).
    """
    state = _task_state(task)
    if state is None or state not in _OUTCOME and state != "canceled":
        return None
    if state == "canceled":
        return None
    return me.observe(
        subject=counterpart, kind="interaction",
        payload={
            "promised": _task_label(task),
            "delivered": _OUTCOME[state],
            "a2a_task_id": str(task.get("id", "")),
            "a2a_state": state,
        },
        context=_task_context(task),
    )


def assert_card(me: BoundPrincipal, counterpart: str,
                card: Dict[str, Any]) -> List[Observation]:
    """Record an AgentCard's skills as self-declared capability assertions.

    Idempotence is inherited from the store's duplicate handling at the
    view level: repeated identical claims add observations, but the
    capabilities deriver aggregates by claim, so re-reading a card
    inflates nothing but the source list.
    """
    out: List[Observation] = []
    for skill in card.get("skills") or []:
        claim = skill.get("name") or skill.get("id")
        if not claim:
            continue
        out.append(me.observe(
            subject=counterpart, kind="assertion",
            payload={
                "claim": str(claim)[:120],
                "source": "self",           # a card is the agent's own claim
                "a2a_skill_id": str(skill.get("id", "")),
            },
        ))
    return out


def confirm_capability(me: BoundPrincipal, counterpart: str,
                       claim: str) -> Observation:
    """Record an OBSERVED confirmation of a claimed capability.

    Call after a completed task actually exercised the skill — this is
    what moves a capability from "they say so" to "we've seen it".
    """
    return me.observe(
        subject=counterpart, kind="assertion",
        payload={"claim": str(claim)[:120], "source": "observed"},
    )
