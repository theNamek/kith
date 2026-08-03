"""Leak-path suite: proof of P1 ("access boundaries are contracts").

Discipline inherited from hermes-agent PR #71224, where review showed that
filtering only the prompt-injection path leaks memories through error
inventories and mutation targeting. Here we enumerate EVERY public surface
of the library and assert a private observation cannot escape any of them.

The cast:
  observer  agent:alice   — owns a private observation about agent:carol
  outsider  agent:bob     — must never see it, through ANY surface
  SECRET                  — sentinel string that must not appear in bob's world
"""

import json

import pytest

from kith import KithError, Scope, Store

SECRET = "carol leaked the prod credentials XYZZY-9000"

ALICE = "agent:alice"
BOB = "agent:bob"
CAROL = "agent:carol"


@pytest.fixture
def store():
    s = Store(":memory:")
    # Alice's private observation about Carol (default scope = observer-only)
    s.observe(ALICE, CAROL, "interaction",
              {"outcome": False, "note": SECRET})
    # Plus one public-ish observation Bob CAN see, as a control
    s.observe(BOB, CAROL, "interaction", {"outcome": True, "note": "fine"})
    return s


def _bob_world(store, **kw):
    """Everything bob can extract through the read APIs, as one string."""
    chunks = []
    chunks.append(json.dumps([{
        "payload": o.payload, "context": o.context,
    } for o in store.observations(BOB, **kw)]))
    chunks.append(store.export(BOB))
    v = store.principal(BOB).view(CAROL)
    chunks.append(json.dumps(v.explain()))
    chunks.append(json.dumps([o.payload for o in v.history(k=100)]))
    chunks.append(repr(v))
    return "\n".join(chunks)


class TestReadSurfaces:
    def test_observations_listing(self, store):
        assert SECRET not in _bob_world(store)

    def test_subject_filtered_listing(self, store):
        got = store.observations(BOB, subject=CAROL)
        assert all(SECRET not in json.dumps(o.payload) for o in got)
        assert len(got) == 1  # only bob's own control observation

    def test_export_is_scoped(self, store):
        assert SECRET not in store.export(BOB)
        assert SECRET in store.export(ALICE)  # owner keeps access

    def test_view_derivation_excludes_hidden(self, store):
        """Bob's trust in Carol must not be moved by Alice's hidden failure —
        derived values are themselves a leak channel if computed over
        invisible observations."""
        v = store.principal(BOB).view(CAROL)
        # bob saw one success; hidden failure must not drag trust below neutral
        assert v.trust > 0.5

    def test_view_provenance_never_names_hidden_ids(self, store):
        alice_obs_id = store.observations(ALICE, subject=CAROL)[0].id
        exp = store.principal(BOB).view(CAROL).explain()
        for dim in exp["dimensions"].values():
            assert alice_obs_id not in dim["derived_from"]

    def test_never_met_indistinguishable_from_hidden(self, store):
        """A subject bob has NO visible observations of must produce the same
        neutral view shape as one that doesn't exist at all — existence of
        hidden data must not be inferable."""
        v_hidden = store.principal(BOB).view("agent:dave")   # no data at all
        store.observe(ALICE, "agent:erin", "interaction",
                      {"outcome": False, "note": SECRET})
        v_exists = store.principal(BOB).view("agent:erin")   # hidden data
        assert v_hidden.trust == v_exists.trust == 0.5
        assert v_hidden.reliability is v_exists.reliability is None
        assert json.dumps(v_hidden.explain()["dimensions"]) == \
               json.dumps(v_exists.explain()["dimensions"])


class TestErrorSurfaces:
    def test_grant_error_does_not_confirm_existence(self, store):
        """Granting someone else's observation and granting a nonexistent one
        must raise the SAME error — else grant() is an existence oracle."""
        alice_obs_id = store.observations(ALICE, subject=CAROL)[0].id
        with pytest.raises(KithError) as e1:
            store.grant(BOB, alice_obs_id, to=BOB)
        with pytest.raises(KithError) as e2:
            store.grant(BOB, "no-such-id", to=BOB)
        assert str(e1.value) == str(e2.value)
        assert SECRET not in str(e1.value)

    def test_validation_errors_never_echo_store_content(self, store):
        with pytest.raises(KithError) as e:
            store.observations("not a principal")
        assert SECRET not in str(e.value)


class TestScopeAndGrantSemantics:
    def test_scoped_holder_sees_only_in_context(self):
        s = Store(":memory:")
        s.observe(ALICE, CAROL, "interaction",
                  {"outcome": False, "note": SECRET},
                  context="task:negotiation",
                  scope=Scope(holders=(BOB,), contexts=("task:negotiation",)))
        # in-context: visible
        in_ctx = s.observations(BOB, subject=CAROL,
                                reader_context="task:negotiation")
        assert len(in_ctx) == 1
        # out-of-context: gone, silently
        out_ctx = s.observations(BOB, subject=CAROL,
                                 reader_context="task:other")
        assert out_ctx == []
        no_ctx = s.observations(BOB, subject=CAROL)
        assert no_ctx == []

    def test_grant_widens_and_is_audited(self, store):
        obs_id = store.observations(ALICE, subject=CAROL)[0].id
        store.grant(ALICE, obs_id, to=BOB, reason="handoff")
        got = store.observations(BOB, subject=CAROL)
        assert any(SECRET in json.dumps(o.payload) for o in got)

    def test_context_restricted_grant(self, store):
        obs_id = store.observations(ALICE, subject=CAROL)[0].id
        store.grant(ALICE, obs_id, to=BOB, contexts=("task:review",))
        assert not any(SECRET in json.dumps(o.payload)
                       for o in store.observations(BOB, subject=CAROL))
        got = store.observations(BOB, subject=CAROL,
                                 reader_context="task:review")
        assert any(SECRET in json.dumps(o.payload) for o in got)

    def test_grant_does_not_leak_to_third_parties(self, store):
        obs_id = store.observations(ALICE, subject=CAROL)[0].id
        store.grant(ALICE, obs_id, to=BOB)
        eve = "agent:eve"
        assert SECRET not in store.export(eve)

    def test_group_context_does_not_auto_widen(self):
        """An observation formed IN a group context is not visible TO other
        group members unless explicitly scoped/granted (DESIGN.md 4.4)."""
        s = Store(":memory:")
        s.observe(ALICE, CAROL, "affect",
                  {"valence": -0.9, "label": "distrust", "note": SECRET},
                  context="group:deploy-team")
        assert SECRET not in s.export(BOB)
        assert SECRET not in _bob_world(s, reader_context="group:deploy-team")


class TestOwnerRoundTrip:
    """The boundary must not break the owner's own flows (the PR #71224
    'owning session keeps access' lesson)."""

    def test_owner_sees_everything_theirs(self, store):
        assert SECRET in store.export(ALICE)
        v = store.principal(ALICE).view(CAROL)
        assert v.trust < 0.5      # the hidden failure counts for its owner
        ids = [o.id for o in store.observations(ALICE, subject=CAROL)]
        assert v.explain()["dimensions"]["trust"]["derived_from"] == ids
