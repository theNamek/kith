"""M1 demo #2: emotional contagion, reconstructed from kith's observation log.

A team of agents exchanges messages over a small-world network. Each agent
has a latent mood; every interaction carries an affect valence drawn from
the sender's mood; receivers are nudged toward what they receive (classic
contagion). One agent is a persistent "toxic" source.

BOTH sides record each interaction into kith as an `affect` observation
(observer = receiver, subject = sender: "how did this exchange feel").

The payoff: WITHOUT access to anyone's latent mood, kith's per-dyad
sentiment views reconstruct the group's emotional dynamics —
  - the contagion wave spreading from the toxic source
  - who is downstream of whom (provenance chains)
  - early-warning: dyad sentiment toward the source collapses before
    the group average does

This is the systems half of the group-emotion-dynamics research agenda:
the observation log IS the longitudinal dataset.

LLM-free, seeded, runs in seconds. Optional PNG chart via matplotlib
(pip install matplotlib); terminal output works without it.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import kith

CONTAGION_RATE = 0.25      # how strongly a received valence moves your mood
MOOD_RECOVERY = 0.02       # per-step drift back toward personal baseline
NOISE = 0.08               # valence expression noise


@dataclass
class SimAgent:
    id: str
    baseline: float                 # personal set-point in [-1, 1]
    mood: float = 0.0
    toxic: bool = False

    def express(self, rng: random.Random) -> float:
        """Valence of an outgoing interaction, from current mood."""
        if self.toxic:
            return max(-1.0, min(1.0, rng.gauss(-0.8, NOISE)))
        return max(-1.0, min(1.0, rng.gauss(self.mood, NOISE)))

    def receive(self, valence: float) -> None:
        if not self.toxic:
            self.mood += CONTAGION_RATE * (valence - self.mood)

    def step(self) -> None:
        if not self.toxic:
            self.mood += MOOD_RECOVERY * (self.baseline - self.mood)


def small_world_edges(n: int, k: int, rewire: float,
                      rng: random.Random) -> List[Tuple[int, int]]:
    """Watts-Strogatz-ish ring lattice with rewiring."""
    edges = set()
    for i in range(n):
        for j in range(1, k // 2 + 1):
            a, b = i, (i + j) % n
            if rng.random() < rewire:
                b = rng.randrange(n)
                if b == a:
                    continue
            edges.add((min(a, b), max(a, b)))
    return sorted(edges)


def run_sim(n_agents: int, steps: int, seed: int,
            store: Optional[kith.Store] = None):
    rng = random.Random(seed)
    agents = [
        SimAgent(id=f"agent:member-{i}",
                 baseline=rng.uniform(0.1, 0.5),
                 mood=rng.uniform(0.0, 0.4))
        for i in range(n_agents)
    ]
    agents[0].toxic = True          # patient zero, persistent negative source
    edges = small_world_edges(n_agents, k=4, rewire=0.2, rng=rng)
    neighbors: Dict[int, List[int]] = {}
    for a, b in edges:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)

    store = store or kith.Store(":memory:")
    principals = {a.id: store.principal(a.id) for a in agents}

    mood_history: List[List[float]] = []
    for t in range(steps):
        # each agent talks to one random neighbor per step
        for i, agent in enumerate(agents):
            if not neighbors.get(i):
                continue
            j = rng.choice(neighbors[i])
            peer = agents[j]
            v = agent.express(rng)
            peer.receive(v)
            # receiver records how the exchange felt (observer=receiver)
            principals[peer.id].observe(
                subject=agent.id, kind="affect",
                payload={"valence": round(v, 4)},
                context="group:team",
            )
        for a in agents:
            a.step()
        mood_history.append([a.mood for a in agents])

    return agents, store, mood_history


def reconstruct_from_kith(store: kith.Store, agents) -> Dict[str, Dict[str, float]]:
    """What each agent's kith views say about each peer — no latent mood used."""
    out: Dict[str, Dict[str, float]] = {}
    for a in agents:
        me = store.principal(a.id)
        row = {}
        for b in agents:
            if a.id == b.id:
                continue
            s = me.view(b.id).sentiment
            if s is not None:
                row[b.id] = s["valence"]
        out[a.id] = row
    return out


def spark(values: List[float], lo: float = -1, hi: float = 1) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    out = []
    for v in values:
        idx = int((v - lo) / (hi - lo) * (len(blocks) - 1) + 0.5)
        out.append(blocks[max(0, min(len(blocks) - 1, idx))])
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--agents", type=int, default=16)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--png", type=str, default="",
                    help="write a chart to this path (needs matplotlib)")
    args = ap.parse_args()

    agents, store, mood_history = run_sim(args.agents, args.steps, args.seed)
    views = reconstruct_from_kith(store, agents)

    toxic_id = agents[0].id
    print(f"{args.agents} agents · {args.steps} steps · seed {args.seed} · "
          f"toxic source: {toxic_id}\n")

    # 1. group mood collapses (ground truth, sim-internal)
    group_avg = [sum(step) / len(step) for step in mood_history]
    print("group mean mood over time (ground truth, latent):")
    print(f"  {spark(group_avg)}   {group_avg[0]:+.2f} -> {group_avg[-1]:+.2f}\n")

    # 2. kith reconstruction: everyone's derived sentiment toward the source
    toward_toxic = [v[toxic_id] for v in views.values() if toxic_id in v]
    toward_others = [val for a, row in views.items()
                     for b, val in row.items() if b != toxic_id]
    avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print("kith-derived sentiment (from observation log only):")
    print(f"  toward toxic source : {avg(toward_toxic):+.3f}  "
          f"(n={len(toward_toxic)} dyads)")
    print(f"  toward everyone else: {avg(toward_others):+.3f}  "
          f"(n={len(toward_others)} dyads)\n")

    # 3. source localization: rank subjects by how the group feels about them
    felt: Dict[str, List[float]] = {}
    for row in views.values():
        for subj, val in row.items():
            felt.setdefault(subj, []).append(val)
    ranked = sorted(felt.items(), key=lambda kv: avg(kv[1]))
    print("most negatively perceived (kith view ranking):")
    for subj, vals in ranked[:3]:
        marker = "  <-- toxic source" if subj == toxic_id else ""
        print(f"  {subj:<18} {avg(vals):+.3f}{marker}")
    hit = ranked[0][0] == toxic_id
    print(f"\nsource localization from memory alone: "
          f"{'CORRECT' if hit else 'missed'}")

    # 4. audit: one dyad's provenance
    victim = agents[1]
    v = store.principal(victim.id).view(toxic_id)
    exp = v.explain()
    n_obs = len(exp["dimensions"]["sentiment"]["derived_from"])
    print(f"\naudit sample — {victim.id}'s view of the source: "
          f"valence {v.sentiment['valence']:+.3f}, trend {v.sentiment['trend']:+.3f}, "
          f"derived from {n_obs} observations (view.explain() lists each)")

    if args.png:
        from plot import render_png   # local module, optional dep
        render_png(args.png, mood_history, views, toxic_id, agents)
        print(f"\nchart written to {args.png}")


if __name__ == "__main__":
    main()
