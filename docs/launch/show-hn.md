# Show HN post

**Title options** (HN limit 80 chars; pick one, A recommended):

- A: `Show HN: Kith – social memory for multi-agent systems (who trusts whom, and why)`
- B: `Show HN: Kith – my agents kept re-delegating to workers that already failed them`
- C: `Show HN: Kith – relationship memory for AI agents, with scope contracts`

**URL**: https://github.com/theNamek/kith

---

**Text** (posted as the first comment, per Show HN convention):

Hi HN — I'm a final-year PhD student working on group emotion dynamics in
multi-agent LLM systems, and kith is the systems half of that research
turned into a small library.

The itch: every agent memory system I could find (Mem0, Zep, LangMem — all
good at what they do) answers one question: *what facts should I
remember?* But when agents work with other agents, the failures that hurt
are relationship-shaped, not fact-shaped:

- my planner delegated to the same flaky coder agent for the fifth time,
  because nothing remembered the first four failures
- a "private" read on a counterpart leaked into a shared channel, because
  memory had no notion of where a memory may go
- every session, my agents met each other as strangers

So kith stores **observations** (append-only: interactions, claims,
affect readings) and derives **relationship views** — trust, reliability,
sentiment, capabilities — with full provenance (`view.explain()` traces
every number to the observations that produced it). Every observation
carries a **scope contract**, enforced at a single gate under every read
path.

Two things I think are genuinely different from existing tools:

1. **Access boundaries are contracts, not filters.** I learned this the
   hard way contributing scoped memory to hermes-agent: my first design
   filtered only the prompt-injection path, and the maintainer's review
   showed memories leaking through error inventories and mutation
   targeting. Kith's test suite enumerates every public surface against a
   sentinel secret — including derived values (a hidden failure must not
   move an outsider's trust score) and existence oracles (a never-met
   subject and a hidden one produce byte-identical views).

2. **Relationship state is computed, never stored as gospel.** Trust
   decays, negative events hit ~2x harder, and every deriver is a
   swappable interface — if you think my trust curve is naive psychology,
   bring your own; the store is just auditable observations.

Reproducible result (seeded, no LLM calls, runs in seconds): a planner
delegating to 20 workers of hidden mixed reliability cuts
repeat-delegation failures by ~82% just by consulting the derived view
before delegating. There's also a demo where per-dyad sentiment views
locate the toxic member of a 16-agent team from the observation log
alone — no access to anyone's internal state (5/5 seeds).

Adapters: LangGraph (wrap any node; supervisor picks by track record),
hermes-agent (memory-provider plugin; delegations auto-accrue), and A2A
(terminal Task states -> outcomes; AgentCard skills stay "self-claimed"
until a real task confirms them).

It's ~1,000 lines of stdlib+sqlite Python, MIT. `pip install kith-ai`
(the bare name was squatted in 2016). Design doc with the failure modes
and non-goals in the repo.

Things I'm unsure about and would love pushback on: whether trust/
sentiment scalars invite over-trust in under-specified psychology (the
`explain()` provenance is my hedge); whether scope-as-text-prefix is too
cute vs. a real storage column; and what the A2A bridge should do about
partially-failed multi-artifact tasks.

---

**First-comment FAQ prep** (don't post; have ready):

- *"Isn't this just a bandit?"* — The delegation demo's policy is
  deliberately a naive bandit; kith is the memory substrate underneath.
  Bandits forget nothing gracefully, share nothing across processes, and
  explain nothing. The contribution is durable, scoped, auditable
  relationship state that ANY policy (incl. your bandit) can read.
- *"Trust scores are pseudoscience"* — The defaults are documented
  heuristics with citations, explicitly swappable (Deriver protocol), and
  every value explains itself down to raw observations. We claim
  auditability, not psychological validity.
- *"Why not just put this in the prompt/Mem0?"* — Facts about people are
  not relationships with people: you need per-observer asymmetry (A
  trusts B ≠ B trusts A), decay, provenance, and visibility contracts.
  Fact stores have none of these as primitives; kith composes with them.
- *"MAST says multi-agent doesn't work"* — MAST's top failure classes
  (inter-agent misalignment, weak verification) are exactly
  relationship-state failures. Also kith is useful at n=2: one user, one
  assistant, cross-context privacy.
- *"Scope in entry text can be spoofed by the model"* — Identity is
  resolved by the runtime (P2): model-facing tools only ever say
  `current`/`peer`; principal binding happens at graph-build/initialize
  time. The model cannot observe as someone else (tested).
