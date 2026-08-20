# r/LocalLLaMA post

**Title** (pick one, A recommended):

- A: `I built a relationship-memory layer for multi-agent systems — repeat-delegation failures down ~82% (MIT, sqlite, no deps)`
- B: `Your agents keep re-delegating to workers that already failed them. I built the memory layer that fixes that.`

**Flair**: Resources (or Discussion if Resources unavailable)

---

**Body**:

Every agent memory tool I've tried (Mem0, Zep, LangMem — all solid) answers
"what facts should I remember?" None of them remember *relationships*: who
delivered, who failed you twice, whose claims got verified, and — critically —
who is allowed to see each memory.

So I built **kith** (MIT, ~1k lines, stdlib+sqlite, zero heavy deps):

- **Observations** (append-only): interactions, capability claims, affect readings
- **Derived views**: trust / reliability / sentiment / capabilities, computed
  with decay — and `view.explain()` traces every number back to the exact
  observations that produced it. Don't like my trust curve? The derivers are
  a swappable interface.
- **Scope contracts**: every memory has an explicit visibility boundary,
  enforced at one gate under *every* read path. The test suite hunts a
  sentinel secret through error messages, exports, and even derived scores
  (a hidden failure must not move an outsider's trust number).

Reproducible result (seeded, no LLM calls, runs in seconds): a planner
delegating to 20 workers of hidden mixed reliability cuts repeat-delegation
failures ~82% just by checking the derived view before delegating. Second
demo: per-dyad sentiment views locate the toxic member of a 16-agent team
from the observation log alone, 5/5 seeds.

Adapters: LangGraph (wrap any node, supervisor picks by track record),
hermes-agent (memory-provider plugin — delegation outcomes accrue
automatically), A2A (terminal Task states → outcomes; AgentCard skills stay
"self-claimed" until a real task confirms them).

Repo: https://github.com/theNamek/kith
Install: `pip install kith-ai`

Background: I'm a final-year PhD student working on group emotion dynamics
in multi-agent LLM systems; the scope-contract design came out of a
scoped-memory PR to hermes-agent where the maintainer's review taught me
that filtering the prompt path isn't enough — memories leak through error
inventories too.

Honest limitations: no semantic retrieval (compose it with Mem0/Zep — kith
is the relationship layer, not the fact layer); trust/sentiment defaults are
documented heuristics, not validated psychology (hence swappable +
explainable); single-store trust domain for now.

Would love feedback on the scope grammar and what an Ollama/local-first
multi-agent setup would need from this.

---

**Posting notes**:
- r/LocalLLaMA cares about: local-first (sqlite ✓), no API calls (✓),
  MIT (✓), reproducibility (✓) — the post leads with all four.
- The last line invites the community's identity ("local-first setups") —
  genuine question, not flattery; be ready to actually discuss Ollama
  integration if asked.
- Same 3-hour engagement rule as HN. Same FAQ ammo applies (show-hn.md).
- Do NOT crosspost to r/MachineLearning (hostile to tool posts).
