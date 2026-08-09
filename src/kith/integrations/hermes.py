"""hermes-agent MemoryProvider adapter.

Implements the hermes-agent ``MemoryProvider`` ABC so kith can run as a
memory backend plugin (the pluggable-provider path of hermes issue #47349),
side by side with fact-memory providers.

What it does inside hermes:

- **on_delegation** (automatic): every subagent completion the parent
  witnesses becomes an interaction observation — task, outcome — so
  delegation reliability accrues with zero agent effort. Outcome is judged
  by a cheap heuristic (error markers in the result), overridable.
- **system_prompt_block**: a compact, scope-safe relationship brief of the
  principals this session has history with.
- **tools**: ``kith_observe`` (record an interaction/affect/assertion about
  a peer) and ``kith_view`` (query a derived relationship view, with
  provenance). Subject ids are validated; the observer is ALWAYS the
  session principal resolved at initialize() — the model cannot observe
  as someone else (P2).

Install (once hermes plugin dir layout):
    ~/.hermes/plugins/memory/kith/{plugin.yaml,__init__.py}
    with __init__.py doing: from kith.integrations.hermes import KithProvider
    and register(collector) -> collector.register_memory_provider(KithProvider())

This module deliberately imports nothing from hermes: the ABC is duck-typed
(hermes loads providers by shape), so kith stays installable without a
hermes checkout. Contract tests in kith's suite verify the shape against
the real ABC when a hermes tree is available.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..model import KithError
from ..store import Store

logger = logging.getLogger(__name__)

_ERROR_MARKERS = re.compile(
    r"\b(error|failed|failure|traceback|exception|could not|unable to)\b",
    re.IGNORECASE,
)

_SUBAGENT_SEQ = 0


def _default_judge(result: str) -> bool:
    """Cheap delegation-outcome heuristic; replace via KithProvider(judge=...)."""
    if not (result or "").strip():
        return False
    return not _ERROR_MARKERS.search(result[:2000])


class KithProvider:
    """hermes-agent MemoryProvider backed by a kith store."""

    def __init__(self, db_path: Optional[str] = None, judge=None):
        self._db_path = db_path            # default resolved at initialize()
        self._judge = judge or _default_judge
        self._store: Optional[Store] = None
        self._me = None                    # BoundPrincipal for this session
        self._enabled = True

    # -- identity of the provider itself --

    @property
    def name(self) -> str:
        return "kith"

    def is_available(self) -> bool:
        return True                        # stdlib + sqlite only

    # -- lifecycle --

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home") or "."
        platform = (kwargs.get("platform") or "cli").strip().lower()
        agent_context = kwargs.get("agent_context") or "primary"
        # Non-primary contexts (cron/flush) must not write relationship
        # state — same reasoning as hermes' own guidance.
        self._enabled = agent_context in ("primary", "subagent")
        path = self._db_path or f"{hermes_home}/kith/kith.db"
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._store = Store(path)
        # The session principal: trusted runtime identity, never the model.
        user_id = str(kwargs.get("user_id") or "").strip()
        pid = f"agent:hermes-{platform}"
        if user_id:
            # per-user principal so multi-user gateways don't blend views
            safe = re.sub(r"[^A-Za-z0-9_.-]", "-", user_id)
            pid = f"agent:hermes-{platform}-{safe}"
        self._me = self._store.principal(pid)

    def shutdown(self) -> None:
        self._store = None
        self._me = None

    # -- prompt surface --

    def system_prompt_block(self) -> str:
        if not self._store or not self._me:
            return ""
        # Brief: subjects this principal has history with (scoped reads only)
        obs = self._me.observations(limit=500)
        subjects = []
        for o in obs:
            if o.subject != self._me.id and o.subject not in subjects:
                subjects.append(o.subject)
        if not subjects:
            return ""
        lines = ["RELATIONSHIPS (kith — derived from your own observations):"]
        for s in subjects[-8:]:
            v = self._me.view(s)
            rel = "unknown" if v.reliability is None else f"{v.reliability:.2f}"
            senti = "" if v.sentiment is None else \
                f", sentiment {v.sentiment['valence']:+.2f}"
            lines.append(f"- {s}: reliability {rel}{senti}")
        return "\n".join(lines)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return ""                          # no semantic retrieval in v0

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        pass

    def sync_turn(self, *args, **kwargs) -> None:
        pass

    # -- the automatic write path --

    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        """Every witnessed subagent completion becomes relationship memory."""
        if not self._enabled or not self._me:
            return
        global _SUBAGENT_SEQ
        subject = kwargs.get("subagent_id")
        if not subject:
            # stable-ish principal per child session; falls back to a pool id
            suffix = re.sub(r"[^A-Za-z0-9_.-]", "-", child_session_id or "")
            if not suffix:
                _SUBAGENT_SEQ += 1
                suffix = f"pool-{_SUBAGENT_SEQ}"
            subject = f"agent:subagent-{suffix}"
        try:
            self._me.observe(
                subject=subject, kind="interaction",
                payload={"promised": (task or "")[:200],
                         "delivered": bool(self._judge(result))},
                context="task:delegation",
            )
        except KithError as e:
            logger.warning("kith on_delegation skipped: %s", e)

    # -- tool surface --

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "kith_observe",
                "description": (
                    "Record a relationship observation about another agent or "
                    "person you interacted with. kinds: 'interaction' (they "
                    "delivered / failed something — include delivered: true/"
                    "false), 'affect' (how an exchange felt — include valence "
                    "in [-1,1]), 'assertion' (a capability claim — include "
                    "claim). You always observe as yourself."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string",
                                    "description": "Who it is about, e.g. 'agent:coder' or 'human:alex'."},
                        "kind": {"type": "string",
                                 "enum": ["interaction", "affect", "assertion"]},
                        "payload": {"type": "object",
                                    "description": "kind-specific fields (delivered / valence / claim...)."},
                    },
                    "required": ["subject", "kind", "payload"],
                },
            },
            {
                "name": "kith_view",
                "description": (
                    "Get your derived relationship view of another principal: "
                    "trust, reliability, sentiment, capabilities — with "
                    "provenance. Neutral view if you have no history."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "explain": {"type": "boolean",
                                    "description": "Include full provenance."},
                    },
                    "required": ["subject"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any],
                         **kwargs) -> str:
        if not self._store or not self._me:
            return json.dumps({"success": False,
                               "error": "kith provider not initialized."})
        try:
            if tool_name == "kith_observe":
                if not self._enabled:
                    return json.dumps({"success": False,
                                       "error": "writes disabled in this agent context."})
                obs = self._me.observe(
                    subject=args["subject"], kind=args["kind"],
                    payload=dict(args.get("payload") or {}),
                )
                return json.dumps({"success": True, "id": obs.id})
            if tool_name == "kith_view":
                v = self._me.view(args["subject"])
                out: Dict[str, Any] = {
                    "success": True, "subject": args["subject"],
                    "trust": v.trust, "reliability": v.reliability,
                    "sentiment": v.sentiment, "capabilities": v.capabilities,
                }
                if args.get("explain"):
                    out["explain"] = v.explain()
                return json.dumps(out, ensure_ascii=False)
            return json.dumps({"success": False,
                               "error": f"unknown tool {tool_name!r}"})
        except (KithError, KeyError) as e:
            return json.dumps({"success": False, "error": str(e)})

    # -- optional hooks hermes may call; harmless defaults --

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        pass

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        pass

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        return ""

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return []

    def backup_paths(self) -> List[str]:
        return []
