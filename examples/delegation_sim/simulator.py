"""M1 demo: does relationship memory reduce coordination failures?

A planner delegates tasks to a pool of worker agents with heterogeneous
(and hidden) reliability. We compare delegation policies:

  - baseline   : no memory — pick a worker at random every time
               (every session meets every agent as a stranger)
  - kith       : consult kith's derived view before delegating — prefer
               workers with high reliability, explore unknowns

The headline metric is REPEAT-DELEGATION FAILURE RATE: how often the
planner hands a task to a worker that has already failed it before.
This is the memory-shaped slice of MAST's "inter-agent misalignment"
failure class: the information existed, the system just didn't remember it.

No LLM calls — worker behavior is simulated so results are exact,
reproducible (seeded), and runnable in seconds. The point is the memory
substrate, not the model.
"""

from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import kith


@dataclass
class Worker:
    """A worker agent with hidden ground-truth reliability."""
    id: str
    p_success: float          # hidden from the planner

    def attempt(self, rng: random.Random) -> bool:
        return rng.random() < self.p_success


@dataclass
class EpisodeResult:
    successes: int = 0
    failures: int = 0
    repeat_failures: int = 0      # delegated to a worker that failed us before
    retries: int = 0              # extra attempts caused by failures
    per_worker_failures: Dict[str, int] = field(default_factory=dict)

    @property
    def tasks(self) -> int:
        return self.successes


def make_team(rng: random.Random, n_workers: int) -> List[Worker]:
    """Heterogeneous pool: some solid, some mediocre, a few flaky."""
    team = []
    for i in range(n_workers):
        r = rng.random()
        if r < 0.4:
            p = rng.uniform(0.85, 0.98)     # solid
        elif r < 0.75:
            p = rng.uniform(0.55, 0.85)     # mediocre
        else:
            p = rng.uniform(0.10, 0.45)     # flaky
        team.append(Worker(id=f"agent:worker-{i}", p_success=p))
    return team


# ---------------------------------------------------------------------------
# Delegation policies
# ---------------------------------------------------------------------------

class BaselinePolicy:
    """No memory: uniform random choice, every task."""

    name = "baseline (no memory)"

    def __init__(self, workers: List[Worker], rng: random.Random):
        self.workers = workers
        self.rng = rng

    def pick(self) -> Worker:
        return self.rng.choice(self.workers)

    def record(self, worker: Worker, ok: bool) -> None:
        pass


class KithPolicy:
    """Consult the relationship view; explore unknowns, avoid known-bad.

    Selection rule (deliberately simple — the demo sells the memory, not
    the bandit algorithm):
      1. if any worker is UNKNOWN (reliability is None), try one (explore)
      2. else pick the highest reliability; break ties randomly
      3. workers with reliability < threshold are excluded unless all are
    """

    name = "kith (relationship memory)"
    AVOID_BELOW = 0.35

    def __init__(self, workers: List[Worker], rng: random.Random,
                 store: Optional[kith.Store] = None):
        self.workers = {w.id: w for w in workers}
        self.rng = rng
        self.store = store or kith.Store(":memory:")
        self.me = self.store.principal("agent:planner")

    def pick(self) -> Worker:
        unknown, known = [], []
        for wid, w in self.workers.items():
            rel = self.me.view(wid).reliability
            (unknown if rel is None else known).append((wid, rel))
        if unknown:
            return self.workers[self.rng.choice(unknown)[0]]
        viable = [(wid, rel) for wid, rel in known if rel >= self.AVOID_BELOW]
        pool = viable or known
        best = max(rel for _, rel in pool)
        top = [wid for wid, rel in pool if rel == best]
        return self.workers[self.rng.choice(top)]

    def record(self, worker: Worker, ok: bool) -> None:
        self.me.observe(subject=worker.id, kind="interaction",
                        payload={"promised": "task", "delivered": ok},
                        context="task:sim")


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(policy, workers: List[Worker], n_tasks: int,
                rng: random.Random, max_retries: int = 5) -> EpisodeResult:
    res = EpisodeResult()
    failed_before: Dict[str, bool] = {}
    for _ in range(n_tasks):
        attempts = 0
        while True:
            w = policy.pick()
            ok = w.attempt(rng)
            policy.record(w, ok)
            attempts += 1
            if ok:
                res.successes += 1
                break
            res.failures += 1
            if failed_before.get(w.id):
                res.repeat_failures += 1
            failed_before[w.id] = True
            res.per_worker_failures[w.id] = res.per_worker_failures.get(w.id, 0) + 1
            if attempts > max_retries:
                break
        res.retries += attempts - 1
    return res


def run_experiment(n_workers: int, n_tasks: int, seeds: List[int]):
    rows = []
    for seed in seeds:
        rng = random.Random(seed)
        team = make_team(rng, n_workers)
        # Same team, same task stream (fresh rng per policy, same seed) —
        # the ONLY difference between arms is memory.
        for policy_cls in (BaselinePolicy, KithPolicy):
            prng = random.Random(seed + 10_000)
            policy = policy_cls(team, prng)
            res = run_episode(policy, team, n_tasks, random.Random(seed + 20_000))
            rows.append({
                "seed": seed, "policy": policy.name,
                "failures": res.failures,
                "repeat_failures": res.repeat_failures,
                "retries": res.retries,
            })
    return rows


def summarize(rows, n_tasks: int) -> str:
    out = []
    policies = sorted({r["policy"] for r in rows})
    stats = {}
    for p in policies:
        sub = [r for r in rows if r["policy"] == p]
        stats[p] = {
            "failures": statistics.mean(r["failures"] for r in sub),
            "repeat": statistics.mean(r["repeat_failures"] for r in sub),
            "retries": statistics.mean(r["retries"] for r in sub),
            "repeat_sd": statistics.stdev([r["repeat_failures"] for r in sub])
                         if len(sub) > 1 else 0.0,
        }
    out.append(f"{'policy':<28} {'failures':>9} {'repeat-fail':>12} {'retries':>8}")
    out.append("-" * 60)
    for p in policies:
        s = stats[p]
        out.append(f"{p:<28} {s['failures']:>9.1f} "
                   f"{s['repeat']:>7.1f}±{s['repeat_sd']:<4.1f} {s['retries']:>8.1f}")
    base = next(p for p in policies if "baseline" in p)
    kith_p = next(p for p in policies if "kith" in p)
    if stats[base]["repeat"] > 0:
        cut = 100 * (1 - stats[kith_p]["repeat"] / stats[base]["repeat"])
        out.append("")
        out.append(f"repeat-delegation failures cut by {cut:.0f}% "
                   f"({stats[base]['repeat']:.1f} -> {stats[kith_p]['repeat']:.1f} "
                   f"per {n_tasks} tasks, mean over {len(rows)//2} seeds)")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--tasks", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    rows = run_experiment(args.workers, args.tasks,
                          seeds=list(range(args.seeds)))
    print(f"team={args.workers} workers · {args.tasks} tasks/episode · "
          f"{args.seeds} seeds\n")
    print(summarize(rows, args.tasks))
