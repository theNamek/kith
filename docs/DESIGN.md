# Kith — Design Document

**Social memory for multi-agent systems: who remembers what, about whom, visible to whom.**

Status: draft v0.1 (2026-08-03) · Author: Yan Liu (@theNamek)

> Working name **kith** ("kith and kin" — the people one knows). Alternatives considered: `rapport`, `entre-nous`, `dossier`. Final name TBD before publish.

---

## 1. Problem

Every agent memory system on the market answers one question: *"what facts should this agent remember?"* Mem0 (61k★), Zep, LangMem, and hermes-agent's built-in memory all model memory as a **single-subject knowledge base**: one agent, one pile of facts, with retrieval quality as the competitive axis.

None of them answer the questions that actually determine whether multi-agent systems work:

1. **Who is this memory *about*?** "User prefers short answers" and "Agent B exaggerates its confidence" are structurally different memories. The second is a *social* memory — a model of another mind — and no current system represents it.
2. **Who may *see* this memory?** A memory formed in one relationship (a DM, a private negotiation, a team channel) leaking into another is not a retrieval bug, it's a trust violation. Current systems treat visibility as an afterthought; production issues prove it (hermes-agent #28279/#11430/#14162/#16833 — four open issues, three competing PRs, all about memory leaking across social contexts).
3. **How does the memory *evolve with the relationship*?** Trust is earned and lost; emotional tone accumulates; a collaborator who failed you twice should be remembered differently from one who failed you once, ten interactions ago. Flat fact stores have no temporal-relational structure (Mem0 scores ~64% on LoCoMo largely because of missing temporal reasoning).

The failure modes documented by multi-agent research (MAST taxonomy: inter-agent misalignment, specification violations, weak verification; Cognition's context-sharing critique) are **relationship-state failures**. Agents fail to coordinate because they have no memory of *each other* — every interaction starts from zero social context.

**Kith is a memory layer for the relationship, not the fact.**

## 2. Positioning

```
            what it stores            competitive axis        examples
──────────────────────────────────────────────────────────────────────────
Fact memory   subject knowledge       retrieval quality       Mem0, Zep, LangMem
Kith          relationship state      social fidelity +       (this project)
              (dyadic + group)        access boundaries
```

Kith is **not** a Mem0 competitor. It composes with fact memory: Mem0 remembers *that the deploy failed*; kith remembers *that it was Agent B's plan, that B under-reported the risk, and that A's trust in B dropped*. Design goal: run beside any fact store, never replace one.

Three product surfaces, in priority order:

1. **Python library** (`pip install kith`) — core data model + store, zero deps beyond stdlib+sqlite by default
2. **Framework adapters** — LangGraph checkpoint hook, hermes-agent MemoryProvider plugin (issue #47349's pluggable-backend path), CrewAI/AutoGen memory shims, A2A metadata profile
3. **Observation bridge** — optional: derive relationship state passively from message traffic (A2A `Task` streams, LangGraph event logs) instead of requiring explicit writes

## 3. Design principles

These four principles are load-bearing; everything in §4–§6 derives from them.

**P1 — Access boundaries are contracts, not filters.** (Learned the hard way in hermes-agent PR #71224: Teknium's review showed that filtering only the prompt-injection path leaks memories through error inventories, match previews, and mutation targeting.) Every read surface — retrieval, error messages, introspection APIs, exports — goes through one visibility gate. If a caller's identity doesn't grant access, the memory does not exist for them, in any code path. Test suites must enumerate *all* result paths, not just the happy one.

**P2 — Identity is resolved by the runtime, never asserted by the model.** An LLM cannot be trusted to know "who am I talking to" — it will hallucinate IDs (PR #71224's second review finding: the model was told to construct `platform:chat_id` it had no access to). Kith APIs take principal identity from trusted runtime context (session objects, A2A AgentCards, framework auth), and expose only resolved tokens (`current`, `peer`) to model-facing tools.

**P3 — Relationship state is computed, not stored as gospel.** Raw observations (interaction records) are the source of truth; trust scores, sentiment summaries, and reliability estimates are *derived views* with explicit decay and provenance. This keeps the store auditable ("why does A distrust B?" → replay the observations) and makes the psychology swappable (see §5).

**P4 — Useful at n=2, designed for n=200.** The minimal unit is one dyad (user↔assistant counts). Group constructs (shared context pools, reputation aggregation, emotional-contagion signals) build on dyadic primitives rather than being a separate system.

## 4. Data model

Three record types, one visibility rule.

### 4.1 Principal

Any party that can be a subject or holder of memory: human user, agent instance, agent role, or group.

```python
Principal(
    id="agent:planner-7",         # stable, runtime-resolved
    kind="agent",                  # agent | human | group
    aliases=["a2a:did:...", "telegram:555000111"],  # cross-framework identity links
)
```

Alias linking is how kith survives heterogeneous deployments (the same human is `telegram:123` in one channel and `slack:U99` in another).

### 4.2 Observation (append-only source of truth)

```python
Observation(
    id=...,
    observer="agent:planner-7",    # who formed this memory
    subject="agent:coder-2",       # who it is about
    context="task:deploy-42",      # relationship context where it formed
    kind="interaction",            # interaction | assertion | affect
    payload={...},                 # kind-specific: outcome, claim, valence/arousal
    visibility=Scope(...),         # see 4.4
    ts=...,
)
```

- `interaction`: something happened (B delivered / failed / went silent) — feeds reliability
- `assertion`: a claim about the subject ("B says it has k8s expertise") — feeds capability model, marked by source
- `affect`: emotional reading of an exchange (valence/arousal or discrete label) — feeds sentiment trajectory; schema deliberately compatible with activation-level emotion probes (see §8)

### 4.3 RelationshipView (derived, materialized on read)

```python
view = kith.view(observer="agent:planner-7", subject="agent:coder-2")
view.trust          # [0,1], decayed aggregate over interaction outcomes
view.reliability    # promise-vs-delivery ratio, windowed
view.sentiment      # recent affect trajectory (EWMA + trend)
view.capabilities   # assertion-derived, with per-claim provenance + confirmation count
view.history(k=5)   # last k observations this view is derived from
view.explain()      # provenance: which observations produced these numbers
```

Derivation functions are pluggable (§5). Views are cheap to recompute and never persisted as authority — only cached.

### 4.4 Scope (the visibility contract)

Directly generalizes the PR #71224 model from strings to principals/contexts:

```python
Scope(
    holders=["agent:planner-7"],           # who may read (default: observer only)
    contexts=["task:deploy-42"],           # AND: only within these contexts
    grants=[],                             # explicit shares, each with provenance
)
```

Rules (each maps to a P1 test class):
- Default scope = observer-private. Sharing is always explicit.
- Scope checks run inside the store, below every API — retrieval, `view()`, errors, exports. There is no unscoped read path, including for "admin" tooling (admin = a principal with granted scopes).
- A grant is itself an observation (auditable: who shared what with whom, when).
- Group contexts don't auto-widen scope: an observation formed *in* a group channel is not automatically visible *to* the group — the observer decides.

## 5. Relationship psychology (pluggable, defaults included)

The derivation layer ships with defaults chosen for defensibility, not novelty:

- **Trust**: asymmetric update (drops fast on betrayal, recovers slowly), exponential time decay toward a neutral prior. Parameters exposed; default curve documented with citations.
- **Reliability**: windowed promise/delivery bookkeeping, Laplace-smoothed.
- **Sentiment**: EWMA over affect observations + short-window trend; discrete emotion labels optional.
- **Reputation (group)**: aggregation of dyadic trust with observer-diversity weighting (resists one agent's grudge dominating), explicitly *not* a global score by default — reputation is always *reputation-with-someone*.

Everything here is a `Deriver` interface: researchers can drop in their own (e.g. Bayesian trust, EmotionBench-calibrated affect). This is the research/product bridge — my emotion-dynamics work (activation-level probes) can feed `affect` observations directly, making kith the systems substrate for the ICLR line without coupling the library to it.

## 6. API sketch

```python
import kith

store = kith.Store("sqlite:///team.db")     # sqlite default; pg optional

me = store.principal("agent:planner-7")

# write path — explicit
me.observe(subject="agent:coder-2", kind="interaction",
           payload={"promised": "fix by 5pm", "delivered": False},
           context="task:deploy-42")

# read path — always scoped to caller
v = me.view("agent:coder-2")
if v.reliability < 0.4:
    plan.add_verification_step()             # MAST-style failure, prevented by memory

# sharing — explicit, audited
me.grant(subject="agent:coder-2", to="agent:reviewer-1",
         contexts=["task:deploy-42"], reason="handoff")

# model-facing tool surface (framework adapters generate this)
# — identity comes from runtime, model only ever says "peer"/"current" (P2)
```

Failure-path behavior (P1): a caller asking about a subject they have no scoped observations of gets an *empty view with a neutral prior* — indistinguishable from "never met" — never an error that reveals existence of hidden observations.

## 7. Architecture

```
┌────────────────────────────────────────────────────┐
│ adapters: LangGraph · hermes MemoryProvider · A2A  │   (thin, per-framework identity resolution)
├────────────────────────────────────────────────────┤
│ tool surface: observe / view / grant  (P2 tokens)  │
├────────────────────────────────────────────────────┤
│ derivers: trust · reliability · sentiment · rep    │   (pluggable, P3)
├────────────────────────────────────────────────────┤
│ scope gate  ← single choke point, every read (P1)  │
├────────────────────────────────────────────────────┤
│ store: append-only observations (sqlite / pg)      │
└────────────────────────────────────────────────────┘
```

Non-goals for v0: vector retrieval (compose with Mem0/Zep for that), cross-process ACL enforcement (single-store trust domain first), automatic affect inference from text (bridge ships later; explicit writes first).

## 8. Research complementarity

Deliberate two-way coupling with the ICLR 2027 emotion-dynamics work, loose enough that neither blocks the other:

- **Research → kith**: activation-level emotion probes produce `affect` observations with better-than-self-report fidelity; the paper's contagion metrics become a kith `Deriver`.
- **Kith → research**: kith's observation log is exactly the longitudinal relationship dataset the group-dynamics experiments need; experiments run on kith get persistence and provenance for free.
- **Shared narrative**: "emotion/relationship state as first-class infrastructure for multi-agent systems" — paper cites library, library README cites paper.

## 9. Milestones

- **M0 (week 1–2)**: core library — data model, sqlite store, scope gate with full leak-path test suite (the PR #71224 test discipline, generalized), default derivers. Ship to PyPI.
- **M1 (week 3–4)**: demo that sells the thesis — 20–50 agent team simulation where enabling kith measurably reduces MAST-style coordination failures (repeat-delegation-to-unreliable-agent as the headline metric); one-command replay + provenance visualization.
- **M2 (week 5–8)**: adapters — LangGraph first (largest orchestration user base), hermes-agent MemoryProvider second (existing reputation there), A2A metadata profile third. HN/掘金 dual launch with the demo.
- **M3 (ongoing)**: observation bridge (passive derivation from A2A/LangGraph traffic), EmotionBench-calibrated affect deriver, paper cross-pollination.

## 10. Risks

| Risk | Mitigation |
|---|---|
| "Trust scores" read as pseudo-science | P3: derived views with provenance + citations, pluggable derivers, `explain()` everywhere; never claim psychological validity, claim *auditability* |
| Scope creep into a full agent framework | Non-goals list in §7; library-first, adapters thin |
| Mem0/Zep add a "social" field and squash the niche | Their architecture is single-subject fact retrieval; relationship-state derivation + scope contracts are a different core loop. Speed + research moat (§8) |
| Multi-agent skepticism ("don't build multi-agents") | Kith's pitch works at n=2 (user↔assistant personalization with privacy) — P4 makes single-agent deployments a real market |
| A2A adoption stalls | A2A is one adapter among three; core is framework-neutral |
