"""LangGraph integration: relationship memory for supervisor-style graphs.

Two primitives, deliberately framework-light (they work with any LangGraph
version because they touch nodes only as callables — no internal APIs):

- ``observe_node(fn, ...)`` — wrap a graph node so every invocation is
  recorded into kith as an interaction observation (success judged by a
  caller-supplied function over the node's output).

- ``KithSupervisor`` — the delegation policy from the demo, packaged:
  pick a worker by derived reliability (explore unknowns, avoid
  known-bad), record outcomes, and render a compact relationship brief
  for injection into a supervisor prompt.

Identity note (P2): principal ids are bound HERE, by the code that builds
the graph — the runtime. Model output never chooses who it is.

Naming note: LangGraph node names may not contain ``:`` — keep a mapping
between node names ("coder") and kith principal ids ("agent:coder");
``KithSupervisor.pick()`` returns principal ids.

Usage sketch::

    from langgraph.graph import StateGraph
    from kith import Store
    from kith.integrations.langgraph import KithSupervisor, observe_node

    store = Store("sqlite:///team.db")
    sup = KithSupervisor(store, supervisor="agent:supervisor",
                         workers=["agent:coder", "agent:researcher"])

    def route(state):                    # supervisor node
        return sup.pick()                # -> node name of chosen worker

    graph.add_node("coder", observe_node(
        coder_fn, store=store, observer="agent:supervisor",
        subject="agent:coder", judge=lambda out: not out.get("error")))
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from ..model import KithError
from ..store import Store

Judge = Callable[[Any], bool]


def observe_node(
    fn: Callable[..., Any],
    *,
    store: Store,
    observer: str,
    subject: str,
    judge: Judge,
    context: Optional[str] = None,
    task_label: str = "task",
) -> Callable[..., Any]:
    """Wrap a LangGraph node so each run records an interaction observation.

    ``judge(output) -> bool`` decides delivered/failed from the node's
    return value. Exceptions count as failures and re-raise unchanged —
    the wrapper must never swallow graph control flow.
    """
    me = store.principal(observer)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            out = fn(*args, **kwargs)
        except Exception:
            me.observe(subject=subject, kind="interaction",
                       payload={"promised": task_label, "delivered": False,
                                "error": "exception"},
                       context=context)
            raise
        ok = bool(judge(out))
        me.observe(subject=subject, kind="interaction",
                   payload={"promised": task_label, "delivered": ok},
                   context=context)
        return out

    wrapped.__name__ = getattr(fn, "__name__", "observed_node")
    return wrapped


class KithSupervisor:
    """View-driven worker selection for supervisor graphs.

    Policy (same shape the delegation demo benchmarks at ~80% fewer
    repeat failures): explore workers with no track record, otherwise
    pick the highest derived reliability, excluding those below
    ``avoid_below`` unless nothing else remains.
    """

    def __init__(self, store: Store, supervisor: str,
                 workers: Sequence[str], *,
                 context: Optional[str] = None,
                 avoid_below: float = 0.35,
                 rng: Optional[random.Random] = None):
        if not workers:
            raise KithError("KithSupervisor needs at least one worker id.")
        self._store = store
        self.me = store.principal(supervisor)
        self.workers = list(workers)
        self.context = context
        self.avoid_below = avoid_below
        self._rng = rng or random.Random()

    def pick(self, among: Optional[Iterable[str]] = None) -> str:
        """Return the worker id to delegate to next."""
        pool = list(among) if among is not None else self.workers
        unknown, known = [], []
        for wid in pool:
            rel = self.me.view(wid, context=self.context).reliability
            (unknown if rel is None else known).append((wid, rel))
        if unknown:
            return self._rng.choice(unknown)[0]
        viable = [(w, r) for w, r in known if r >= self.avoid_below]
        candidates = viable or known
        best = max(r for _, r in candidates)
        return self._rng.choice([w for w, r in candidates if r == best])

    def record(self, worker: str, ok: bool, *,
               task_label: str = "task") -> None:
        """Record a delegation outcome the supervisor witnessed."""
        self.me.observe(subject=worker, kind="interaction",
                        payload={"promised": task_label, "delivered": ok},
                        context=self.context)

    def brief(self, among: Optional[Iterable[str]] = None) -> str:
        """Compact relationship brief for a supervisor system prompt.

        One line per worker, derived values only — safe to inject (it is
        built from the supervisor's own scoped views, so it can never
        surface anything the supervisor may not see).
        """
        pool = list(among) if among is not None else self.workers
        lines = []
        for wid in pool:
            v = self.me.view(wid, context=self.context)
            rel = "unknown" if v.reliability is None else f"{v.reliability:.2f}"
            senti = "" if v.sentiment is None else \
                f", sentiment {v.sentiment['valence']:+.2f}"
            lines.append(f"- {wid}: reliability {rel}"
                         f" (n={len(v.history(k=1000))}){senti}")
        return "\n".join(lines)
