"""Core behavior: model validation, store round-trips, derivers, views."""

import json
import time

import pytest

from kith import KithError, Observation, RelationshipView, Scope, Store


@pytest.fixture
def store():
    return Store(":memory:")


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------

class TestModel:
    def test_valid_principal_ids(self):
        for pid in ["agent:planner-7", "human:scarlett", "group:team_a",
                    "agent:x.y-z_2"]:
            Observation(observer=pid, subject="agent:b",
                        kind="interaction", payload={})

    @pytest.mark.parametrize("bad", [
        "planner-7",            # no kind
        "robot:x",              # unknown kind
        "agent:has space",      # bad charset
        "agent:",               # empty name
        "AGENT:x",              # kind must be lowercase
    ])
    def test_invalid_principal_ids(self, bad):
        with pytest.raises(KithError):
            Observation(observer=bad, subject="agent:b",
                        kind="interaction", payload={})

    def test_unknown_observation_kind(self):
        with pytest.raises(KithError):
            Observation(observer="agent:a", subject="agent:b",
                        kind="gossip", payload={})

    def test_payload_must_be_json_serializable(self):
        with pytest.raises(KithError):
            Observation(observer="agent:a", subject="agent:b",
                        kind="interaction", payload={"x": object()})

    def test_affect_payload_needs_signal(self, store):
        with pytest.raises(KithError):
            store.observe("agent:a", "agent:b", "affect", {})

    def test_affect_valence_range(self, store):
        with pytest.raises(KithError):
            store.observe("agent:a", "agent:b", "affect", {"valence": 2.0})

    def test_scope_json_roundtrip(self):
        s = Scope(holders=("agent:a", "human:u"), contexts=("task:1",))
        assert Scope.from_json(s.to_json()) == s


# ---------------------------------------------------------------------------
# Store round-trips
# ---------------------------------------------------------------------------

class TestStore:
    def test_observe_and_read_back(self, store):
        obs = store.observe("agent:a", "agent:b", "interaction",
                            {"outcome": True}, context="task:1")
        got = store.observations("agent:a", subject="agent:b")
        assert len(got) == 1
        assert got[0].id == obs.id
        assert got[0].payload == {"outcome": True}
        assert got[0].context == "task:1"

    def test_sqlite_url_form(self, tmp_path):
        s = Store(f"sqlite:///{tmp_path}/t.db")
        s.observe("agent:a", "agent:b", "interaction", {"outcome": True})
        assert len(s.observations("agent:a", subject="agent:b")) == 1

    def test_observations_ordered_by_time(self, store):
        o1 = store.observe("agent:a", "agent:b", "interaction", {"outcome": True})
        o2 = store.observe("agent:a", "agent:b", "interaction", {"outcome": False})
        got = store.observations("agent:a", subject="agent:b")
        assert [o.id for o in got] == [o1.id, o2.id]

    def test_limit_returns_most_recent(self, store):
        for i in range(5):
            store.observe("agent:a", "agent:b", "interaction",
                          {"outcome": True, "i": i})
        got = store.observations("agent:a", subject="agent:b", limit=2)
        assert [o.payload["i"] for o in got] == [3, 4]

    def test_bound_principal_facade(self, store):
        me = store.principal("agent:a")
        me.observe("agent:b", "interaction", {"outcome": True})
        assert len(me.observations(subject="agent:b")) == 1


# ---------------------------------------------------------------------------
# Derivers via RelationshipView
# ---------------------------------------------------------------------------

class TestDerivedView:
    def test_never_met_is_neutral(self, store):
        v = store.principal("agent:a").view("agent:stranger")
        assert v.trust == 0.5
        assert v.reliability is None
        assert v.sentiment is None
        assert v.capabilities == []

    def test_successes_raise_trust_failures_lower_it_harder(self, store):
        me = store.principal("agent:a")
        for _ in range(3):
            me.observe("agent:good", "interaction", {"outcome": True})
            me.observe("agent:bad", "interaction", {"outcome": False})
        good, bad = me.view("agent:good"), me.view("agent:bad")
        assert good.trust > 0.5 > bad.trust
        # asymmetry: |drop| > |gain| for the same event count
        assert (0.5 - bad.trust) > (good.trust - 0.5)

    def test_trust_decays_toward_neutral(self, store):
        me = store.principal("agent:a")
        me.observe("agent:b", "interaction", {"outcome": True})
        fresh = me.view("agent:b").trust
        # rebuild the view as if 90 days later
        obs = me.observations(subject="agent:b")
        later = RelationshipView("agent:a", "agent:b", obs,
                                 now=time.time() + 90 * 86400)
        assert 0.5 < later.trust < fresh

    def test_reliability_counts_promises(self, store):
        me = store.principal("agent:a")
        me.observe("agent:b", "interaction", {"promised": "x", "delivered": True})
        me.observe("agent:b", "interaction", {"promised": "y", "delivered": False})
        r = me.view("agent:b").reliability
        # 1 delivered of 2, Laplace smoothed: (1+1)/(2+2) = 0.5
        assert r == pytest.approx(0.5)

    def test_sentiment_ewma_and_trend(self, store):
        me = store.principal("agent:a")
        for v in [-0.8, -0.5, 0.3, 0.7]:
            me.observe("agent:b", "affect", {"valence": v})
        s = me.view("agent:b").sentiment
        # EWMA is pulled up by recent positives but carries the negative
        # history (that's the point of a memory); trend captures direction.
        assert -0.8 < s["valence"] < 0.7
        assert s["valence"] > -0.65      # strictly above the early average
        assert s["trend"] > 0            # improving

    def test_capabilities_track_confirmation(self, store):
        me = store.principal("agent:a")
        me.observe("agent:b", "assertion",
                   {"claim": "kubernetes", "source": "self"})
        me.observe("agent:b", "assertion",
                   {"claim": "kubernetes", "source": "observed"})
        caps = me.view("agent:b").capabilities
        assert len(caps) == 1
        assert caps[0]["claim"] == "kubernetes"
        assert caps[0]["confirmations"] == 1

    def test_explain_provenance_covers_every_dimension(self, store):
        me = store.principal("agent:a")
        o1 = me.observe("agent:b", "interaction", {"outcome": True})
        o2 = me.observe("agent:b", "affect", {"valence": 0.5})
        exp = me.view("agent:b").explain()
        assert exp["dimensions"]["trust"]["derived_from"] == [o1.id]
        assert exp["dimensions"]["sentiment"]["derived_from"] == [o2.id]
        assert exp["observation_count"] == 2

    def test_history(self, store):
        me = store.principal("agent:a")
        for i in range(7):
            me.observe("agent:b", "interaction", {"outcome": True, "i": i})
        h = me.view("agent:b").history(k=3)
        assert [o.payload["i"] for o in h] == [4, 5, 6]
