# kith

**Social memory for multi-agent systems — who remembers what, about whom, visible to whom.**

> Working draft README. Not yet published.

## Why

Agent memory today answers one question: *what facts should I remember?* (Mem0, Zep, LangMem — all excellent at it.)

But when agents work **with other agents** (or with you, across contexts), the failures that actually hurt are not fact-retrieval failures:

- Your planner delegates to the same flaky coder agent for the fifth time, because nothing remembers the first four failures.
- A negotiation agent's private read on its counterpart leaks into a shared channel, because memory has no notion of *where a memory may go*.
- Every session, your agents meet each other as strangers.

Research on multi-agent failures (the MAST taxonomy) puts inter-agent misalignment and weak verification at the top of the list. These are **relationship-state failures**: the system has no memory of *who anyone is to anyone*.

kith is that memory.

## Install

```bash
pip install kith-ai        # imports as `kith`
```

## What it is

A small Python library (sqlite by default, zero heavy deps) that gives your agents:

- **Observations** — append-only records of what happened between principals: interactions, claims, emotional reads
- **Relationship views** — trust, reliability, sentiment, capabilities: *derived* from observations with decay and full provenance (`view.explain()` shows its work)
- **Scope contracts** — every memory has an explicit visibility boundary, enforced at a single gate below every read path (retrieval, errors, exports — no leaks through side doors)

```python
import kith

store = kith.Store("sqlite:///team.db")
me = store.principal("agent:planner-7")

me.observe(subject="agent:coder-2", kind="interaction",
           payload={"promised": "fix by 5pm", "delivered": False},
           context="task:deploy-42")

v = me.view("agent:coder-2")
if v.reliability < 0.4:
    plan.add_verification_step()   # remembered, not repeated
```

It is **not** a Mem0 replacement — run it beside your fact store. Facts are what happened; kith is what it did to the relationship.

## Design commitments

1. **Access boundaries are contracts, not filters.** One visibility gate under every read surface, with a leak-path test suite to prove it. (Battle-tested design: grew out of [hermes-agent #71224](https://github.com/NousResearch/hermes-agent/pull/71224), where snapshot-only filtering was shown to leak through error inventories.)
2. **Identity comes from the runtime, never from the model.** Models say `current`/`peer`; the runtime resolves who that is. LLMs don't get to invent IDs.
3. **Psychology is pluggable.** Default trust/reliability/sentiment derivers are documented and swappable — bring your own model of a mind.
4. **Useful at n=2.** One user, one assistant, cross-context privacy: already worth it. Scales to agent teams from there.

## Does it help? (reproducible)

A planner delegating to 20 workers of hidden, mixed reliability — identical
setup, the only difference is memory:

```
policy                        failures  repeat-fail  retries
------------------------------------------------------------
baseline (no memory)             100.6    83.1±32.9    100.6
kith (relationship memory)        25.2    15.3±9.5      25.2

repeat-delegation failures cut by 82%
```

`python examples/delegation_sim/simulator.py` — seeded, no LLM calls,
runs in seconds. [Details](examples/delegation_sim/README.md).

And the observation log doubles as a research instrument: in the
[contagion demo](examples/contagion_viz/README.md), kith's derived
sentiment views reconstruct a team's emotional dynamics — and locate the
toxic source — from per-dyad memories alone, no access to anyone's
internal state.

## Status

v0.1 on PyPI (`pip install kith-ai`). Core library + leak-path test suite.
See [docs/DESIGN.md](docs/DESIGN.md). Adapters planned: LangGraph,
hermes-agent (MemoryProvider), A2A.

## Author

Yan Liu ([@theNamek](https://github.com/theNamek)) — PhD researcher on group emotion dynamics in multi-agent LLM systems.
