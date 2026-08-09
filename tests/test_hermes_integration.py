"""hermes MemoryProvider adapter tests.

Unit tests always run (the adapter is duck-typed, no hermes import).
When a hermes-agent checkout is available (HERMES_AGENT_SRC or the
default sibling path), a contract test verifies KithProvider satisfies
the real MemoryProvider ABC.
"""

import json
import os
import sys

import pytest

from kith.integrations.hermes import KithProvider, _default_judge

HERMES_SRC = os.environ.get(
    "HERMES_AGENT_SRC",
    os.path.expanduser("~/2026/explore/Projects/hermes-agent"),
)


@pytest.fixture
def provider(tmp_path):
    p = KithProvider(db_path=str(tmp_path / "kith.db"))
    p.initialize("sess-1", hermes_home=str(tmp_path), platform="telegram",
                 agent_context="primary", user_id="42")
    return p


class TestLifecycle:
    def test_initialize_builds_per_user_principal(self, provider):
        assert provider._me.id == "agent:hermes-telegram-42"

    def test_cli_without_user(self, tmp_path):
        p = KithProvider(db_path=str(tmp_path / "k.db"))
        p.initialize("s", hermes_home=str(tmp_path), platform="cli")
        assert p._me.id == "agent:hermes-cli"

    def test_cron_context_disables_writes(self, tmp_path):
        p = KithProvider(db_path=str(tmp_path / "k.db"))
        p.initialize("s", hermes_home=str(tmp_path), platform="cli",
                     agent_context="cron")
        p.on_delegation("task", "done", child_session_id="c1")
        assert p._me.observations() == []
        out = json.loads(p.handle_tool_call(
            "kith_observe",
            {"subject": "agent:x", "kind": "affect",
             "payload": {"valence": 0.5}}))
        assert out["success"] is False


class TestDelegationHook:
    def test_success_and_failure_judged(self, provider):
        provider.on_delegation("build the report", "Report complete, all good.",
                               child_session_id="c-1")
        provider.on_delegation("fix the bug",
                               "Traceback (most recent call last): ...",
                               child_session_id="c-2")
        obs = provider._me.observations()
        assert obs[0].payload["delivered"] is True
        assert obs[1].payload["delivered"] is False
        assert obs[0].subject == "agent:subagent-c-1"

    def test_judge_heuristic(self):
        assert _default_judge("All tests pass.") is True
        assert _default_judge("I was unable to complete this") is False
        assert _default_judge("") is False

    def test_custom_judge(self, tmp_path):
        p = KithProvider(db_path=str(tmp_path / "k.db"),
                         judge=lambda r: r == "OK")
        p.initialize("s", hermes_home=str(tmp_path), platform="cli")
        p.on_delegation("t", "OK", child_session_id="a")
        p.on_delegation("t", "great success", child_session_id="b")
        obs = p._me.observations()
        assert obs[0].payload["delivered"] is True
        assert obs[1].payload["delivered"] is False


class TestToolSurface:
    def test_observe_and_view_roundtrip(self, provider):
        out = json.loads(provider.handle_tool_call(
            "kith_observe",
            {"subject": "agent:coder", "kind": "interaction",
             "payload": {"promised": "fix", "delivered": False}}))
        assert out["success"] is True
        view = json.loads(provider.handle_tool_call(
            "kith_view", {"subject": "agent:coder", "explain": True}))
        assert view["reliability"] == pytest.approx(1 / 3, abs=1e-4)
        assert view["explain"]["dimensions"]["reliability"]["derived_from"]

    def test_observer_is_always_session_principal(self, provider):
        """P2: the model cannot pick who it observes as — args carry no
        observer field, and the stored observer is the session principal."""
        schemas = provider.get_tool_schemas()
        observe = next(s for s in schemas if s["name"] == "kith_observe")
        assert "observer" not in observe["parameters"]["properties"]
        provider.handle_tool_call(
            "kith_observe",
            {"subject": "agent:coder", "kind": "affect",
             "payload": {"valence": -0.5}})
        obs = provider._me.observations(subject="agent:coder")
        assert all(o.observer == "agent:hermes-telegram-42" for o in obs)

    def test_invalid_args_return_clean_error(self, provider):
        out = json.loads(provider.handle_tool_call(
            "kith_observe",
            {"subject": "not-a-principal", "kind": "affect",
             "payload": {"valence": 2}}))
        assert out["success"] is False

    def test_prompt_block_lists_relationships(self, provider):
        provider.on_delegation("t", "done ok", child_session_id="w1")
        block = provider.system_prompt_block()
        assert "agent:subagent-w1" in block
        assert "reliability" in block

    def test_prompt_block_empty_without_history(self, tmp_path):
        p = KithProvider(db_path=str(tmp_path / "k.db"))
        p.initialize("s", hermes_home=str(tmp_path), platform="cli")
        assert p.system_prompt_block() == ""


# ---------------------------------------------------------------------------
# Contract test against the real hermes ABC (when a checkout is present)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.isdir(os.path.join(HERMES_SRC, "agent")),
                    reason="hermes-agent checkout not found")
class TestRealABCContract:
    def test_satisfies_memory_provider_abc(self, tmp_path):
        sys.path.insert(0, HERMES_SRC)
        try:
            from agent.memory_provider import MemoryProvider
            abstract = getattr(MemoryProvider, "__abstractmethods__", set())
            for method in abstract:
                assert hasattr(KithProvider, method), \
                    f"KithProvider missing abstract member {method!r}"
            # duck-typed registration path: hermes checks isinstance in some
            # code paths — build a subclass shim the way the plugin does
            class _Shim(KithProvider, MemoryProvider):
                pass
            shim = _Shim(db_path=str(tmp_path / "k.db"))
            assert isinstance(shim, MemoryProvider)
            shim.initialize("s", hermes_home=str(tmp_path), platform="cli")
            assert shim.system_prompt_block() == ""
        finally:
            sys.path.remove(HERMES_SRC)
