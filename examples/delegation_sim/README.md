# Delegation simulation — the "why kith" demo

Does relationship memory reduce multi-agent coordination failures?

A planner delegates tasks to a pool of workers with hidden, heterogeneous
reliability. Two policies, identical in everything except memory:

- **baseline** — no memory; every task meets every worker as a stranger
- **kith** — record each interaction as a kith observation; before
  delegating, consult `view(worker).reliability` (explore unknowns,
  avoid known-bad)

```bash
python examples/delegation_sim/simulator.py --workers 20 --tasks 200 --seeds 10
```

Representative output:

```
policy                        failures  repeat-fail  retries
------------------------------------------------------------
baseline (no memory)             100.6    83.1±32.9    100.6
kith (relationship memory)        25.2    15.3±9.5      25.2

repeat-delegation failures cut by 82% (83.1 -> 15.3 per 200 tasks)
```

At 50 workers / 500 tasks / 20 seeds the cut holds at ~79%.

**Repeat-delegation failure** — handing a task to a worker that already
failed you — is the memory-shaped slice of the MAST failure taxonomy's
"inter-agent misalignment" class: the information existed, the system
just didn't remember it. That is the failure kith removes.

Notes:

- No LLM calls; worker behavior is simulated so runs are exact, seeded,
  and finish in seconds. The subject under test is the memory substrate,
  not a model.
- The kith policy is deliberately naive (greedy + explore-unknowns). A
  bandit algorithm would do better still — bring your own; kith just
  remembers.
- Every number the policy uses is auditable: `view.explain()` traces a
  reliability score back to the exact observations that produced it.
