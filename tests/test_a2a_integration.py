"""A2A profile tests — realistic v1 wire-shaped Task/AgentCard dicts."""

import pytest

from kith import Store
from kith.integrations.a2a import assert_card, confirm_capability, observe_task


@pytest.fixture
def me():
    return Store(":memory:").principal("agent:orchestrator")

def _task(state, task_id="t-1", context_id="ctx-9", text="summarize the report"):
    return {
        "id": task_id,
        "contextId": context_id,
        "status": {"state": state},
        "history": [{"role": "user", "parts": [{"text": text}]}],
    }


class TestObserveTask:
    def test_completed_records_success(self, me):
        obs = observe_task(me, "agent:remote-writer", _task("completed"))
        assert obs.payload["delivered"] is True
        assert obs.payload["promised"] == "summarize the report"
        assert obs.context == "a2a:ctx-9"
        assert me.view("agent:remote-writer").reliability > 0.5

    @pytest.mark.parametrize("state", ["failed", "rejected",
                                       "TASK_STATE_FAILED"])
    def test_failure_states(self, me, state):
        obs = observe_task(me, "agent:remote-writer", _task(state))
        assert obs.payload["delivered"] is False

    @pytest.mark.parametrize("state", ["canceled", "cancelled", "working",
                                       "submitted", "input-required",
                                       "auth-required", "bogus", ""])
    def test_no_signal_states_record_nothing(self, me, state):
        assert observe_task(me, "agent:remote-writer", _task(state)) is None
        assert me.observations(subject="agent:remote-writer") == []

    def test_proto_enum_names_accepted(self, me):
        obs = observe_task(me, "agent:r", _task("TASK_STATE_COMPLETED"))
        assert obs.payload["delivered"] is True

    def test_task_without_history_gets_fallback_label(self, me):
        t = {"id": "t-9", "status": {"state": "completed"}}
        obs = observe_task(me, "agent:r", t)
        assert "t-9" in obs.payload["promised"]
        assert obs.context is None


class TestAgentCard:
    CARD = {
        "name": "Remote Writer",
        "skills": [
            {"id": "sum", "name": "summarization", "tags": ["text"]},
            {"id": "tr", "name": "translation", "tags": ["text"]},
            {"id": "", "name": ""},                    # malformed — skipped
        ],
    }

    def test_card_becomes_self_assertions(self, me):
        out = assert_card(me, "agent:remote-writer", self.CARD)
        assert len(out) == 2
        caps = me.view("agent:remote-writer").capabilities
        assert {c["claim"] for c in caps} == {"summarization", "translation"}
        # self-declared only: zero confirmations
        assert all(c["confirmations"] == 0 for c in caps)

    def test_confirmation_lifecycle(self, me):
        assert_card(me, "agent:remote-writer", self.CARD)
        confirm_capability(me, "agent:remote-writer", "summarization")
        caps = {c["claim"]: c for c in
                me.view("agent:remote-writer").capabilities}
        assert caps["summarization"]["confirmations"] == 1
        assert caps["translation"]["confirmations"] == 0

    def test_full_a2a_loop(self, me):
        """Card -> delegation -> outcome -> confirmed capability."""
        assert_card(me, "agent:remote-writer", self.CARD)
        observe_task(me, "agent:remote-writer",
                     _task("completed", text="summarize Q3 earnings"))
        confirm_capability(me, "agent:remote-writer", "summarization")
        v = me.view("agent:remote-writer")
        assert v.reliability > 0.5
        assert v.trust > 0.5
        caps = {c["claim"]: c for c in v.capabilities}
        assert caps["summarization"]["confirmations"] == 1
        # provenance chains all the way down
        exp = v.explain()
        assert exp["observation_count"] == 4
