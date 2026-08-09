"""LangGraph adapter tests — run against a REAL StateGraph when langgraph
is installed (integration), plus framework-free unit tests that always run."""

import random

import pytest

from kith import Store
from kith.integrations.langgraph import KithSupervisor, observe_node

langgraph = pytest.importorskip("langgraph", reason="integration tests need langgraph")


# ---------------------------------------------------------------------------
# Unit level (no graph)
# ---------------------------------------------------------------------------

class TestObserveNode:
    def test_records_success_and_failure(self):
        store = Store(":memory:")
        calls = {"n": 0}

        def node(state):
            calls["n"] += 1
            return {"error": calls["n"] == 1}   # first call fails

        wrapped = observe_node(node, store=store, observer="agent:sup",
                               subject="agent:w1",
                               judge=lambda out: not out["error"])
        wrapped({})
        wrapped({})
        v = store.principal("agent:sup").view("agent:w1")
        assert v.reliability == pytest.approx((1 + 1) / (2 + 2))  # 1 ok of 2

    def test_exception_recorded_and_reraised(self):
        store = Store(":memory:")

        def bad(state):
            raise RuntimeError("boom")

        wrapped = observe_node(bad, store=store, observer="agent:sup",
                               subject="agent:w1", judge=lambda out: True)
        with pytest.raises(RuntimeError):
            wrapped({})
        obs = store.principal("agent:sup").observations(subject="agent:w1")
        assert obs[-1].payload["delivered"] is False
        assert obs[-1].payload["error"] == "exception"


class TestSupervisorPolicy:
    def test_explores_unknowns_first(self):
        store = Store(":memory:")
        sup = KithSupervisor(store, "agent:sup",
                             ["agent:w1", "agent:w2"], rng=random.Random(0))
        sup.record("agent:w1", True)
        assert sup.pick() == "agent:w2"      # w2 has no track record yet

    def test_avoids_known_bad(self):
        store = Store(":memory:")
        sup = KithSupervisor(store, "agent:sup",
                             ["agent:good", "agent:bad"], rng=random.Random(0))
        for _ in range(5):
            sup.record("agent:good", True)
            sup.record("agent:bad", False)
        assert all(sup.pick() == "agent:good" for _ in range(10))

    def test_brief_reports_derived_state_only(self):
        store = Store(":memory:")
        sup = KithSupervisor(store, "agent:sup", ["agent:w1", "agent:w2"])
        sup.record("agent:w1", True)
        text = sup.brief()
        assert "agent:w1: reliability 0.67" in text
        assert "agent:w2: reliability unknown" in text

    def test_brief_is_scope_safe(self):
        """Another principal's scoped observation must not surface in the
        supervisor's brief (P1 composes into the adapter layer)."""
        store = Store(":memory:")
        secret = "w1 leaked credentials SENTINEL-XYZZY"
        store.observe("agent:someone-else", "agent:w1", "interaction",
                      {"delivered": False, "note": secret})
        sup = KithSupervisor(store, "agent:sup", ["agent:w1"])
        assert secret not in sup.brief()
        assert "unknown" in sup.brief()      # sup has no view of w1 at all


# ---------------------------------------------------------------------------
# Integration: a real LangGraph StateGraph with a flaky and a solid worker
# ---------------------------------------------------------------------------

class TestRealStateGraph:
    def test_supervisor_learns_to_avoid_flaky_worker(self):
        from typing_extensions import TypedDict
        from langgraph.graph import StateGraph, START, END

        class S(TypedDict):
            worker: str
            ok: bool
            done: int

        store = Store(":memory:")
        rng = random.Random(42)
        sup = KithSupervisor(store, "agent:supervisor",
                             ["agent:solid", "agent:flaky"],
                             rng=random.Random(1))

        def make_worker(p_success):
            def worker(state: S) -> dict:
                return {"ok": rng.random() < p_success,
                        "done": state["done"] + 1}
            return worker

        solid = observe_node(make_worker(0.95), store=store,
                             observer="agent:supervisor",
                             subject="agent:solid",
                             judge=lambda out: out["ok"])
        flaky = observe_node(make_worker(0.10), store=store,
                             observer="agent:supervisor",
                             subject="agent:flaky",
                             judge=lambda out: out["ok"])

        # LangGraph node names may not contain ':' — map node name <-> principal
        node_of = {"agent:solid": "solid", "agent:flaky": "flaky"}

        def route(state: S) -> dict:
            return {"worker": sup.pick()}

        g = StateGraph(S)
        g.add_node("route", route)
        g.add_node("solid", solid)
        g.add_node("flaky", flaky)
        g.add_edge(START, "route")
        g.add_conditional_edges("route", lambda s: node_of[s["worker"]],
                                {"solid": "solid", "flaky": "flaky"})
        g.add_edge("solid", END)
        g.add_edge("flaky", END)
        app = g.compile()

        picks = []
        for _ in range(20):
            out = app.invoke({"worker": "", "ok": False, "done": 0})
            picks.append(out["worker"])

        # after both are explored, the flaky worker stops being chosen
        late_picks = picks[5:]
        flaky_share = late_picks.count("agent:flaky") / len(late_picks)
        assert flaky_share <= 0.2
        # and the memory is queryable/auditable after the run
        sup_view = store.principal("agent:supervisor")
        v_flaky = sup_view.view("agent:flaky")
        v_solid = sup_view.view("agent:solid")
        assert v_flaky.reliability is not None
        assert v_solid.reliability > v_flaky.reliability
        assert v_flaky.explain()["dimensions"]["reliability"]["derived_from"]
