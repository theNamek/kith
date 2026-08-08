"""PNG chart for the contagion demo (optional; needs matplotlib).

Two panels:
  A — emphasis line chart: each agent's latent mood as recessive gray,
      group mean highlighted; the story is the collapse, not 16 series.
  B — diverging bar: kith-derived sentiment toward each member (how the
      group feels about them, reconstructed from observations alone);
      the toxic source stands out at the negative pole.

Palette: validated blue/red pair on light surface (dataviz reference
palette; CVD ΔE 74.6, contrast ≥3:1).
"""

from __future__ import annotations

from typing import Dict, List

# reference palette (light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2ND = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"       # emphasis series / negative-pole-free bars
RED = "#e34948"        # warm pole: negative sentiment / toxic source
GRAY_DEEMPH = "#c9c8c1"


def render_png(path: str, mood_history: List[List[float]],
               views: Dict[str, Dict[str, float]], toxic_id: str,
               agents) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(12, 4.6), dpi=150,
        gridspec_kw={"width_ratios": [1.25, 1]},
    )
    fig.patch.set_facecolor(SURFACE)

    # ── Panel A: emphasis line — individual moods recede, mean leads ──
    ax_a.set_facecolor(SURFACE)
    steps = range(len(mood_history))
    n = len(mood_history[0])
    for i in range(n):
        series = [row[i] for row in mood_history]
        ax_a.plot(steps, series, color=GRAY_DEEMPH, lw=0.9, zorder=1)
    mean = [sum(row) / len(row) for row in mood_history]
    ax_a.plot(steps, mean, color=BLUE, lw=2.2, zorder=3)
    # selective direct labels (not a legend box: one highlighted series)
    ax_a.annotate("group mean", (len(mean) - 1, mean[-1]),
                  xytext=(6, 0), textcoords="offset points",
                  color=BLUE, fontsize=9, fontweight="bold", va="center")
    ax_a.annotate("individual members", (len(mean) // 5,
                  max(r[len(r) // 3] for r in [mood_history[len(mood_history) // 5]])),
                  xytext=(0, 10), textcoords="offset points",
                  color=MUTED, fontsize=8)
    ax_a.axhline(0, color=BASELINE, lw=1, zorder=0)
    ax_a.set_title("Latent mood over time (ground truth)",
                   color=INK, fontsize=11, loc="left")
    ax_a.set_xlabel("step", color=INK_2ND, fontsize=9)
    ax_a.set_ylabel("mood (valence)", color=INK_2ND, fontsize=9)
    ax_a.set_ylim(-1, 1)

    # ── Panel B: diverging bar — kith-derived perception of each member ──
    ax_b.set_facecolor(SURFACE)
    felt: Dict[str, List[float]] = {}
    for row in views.values():
        for subj, val in row.items():
            felt.setdefault(subj, []).append(val)
    ranked = sorted(((sum(v) / len(v), s) for s, v in felt.items()))
    values = [v for v, _ in ranked]
    labels = [s.split("-")[-1] for _, s in ranked]
    colors = [RED if s == toxic_id else BLUE for _, s in ranked]
    y = range(len(ranked))
    ax_b.barh(y, values, color=colors, height=0.62, zorder=2)
    ax_b.axvline(0, color=BASELINE, lw=1, zorder=1)
    ax_b.set_yticks(list(y))
    ax_b.set_yticklabels(labels, fontsize=7.5, color=INK_2ND)
    # direct label on the one that matters — inside the bar, surface-toned
    for yi, (v, s) in enumerate(ranked):
        if s == toxic_id:
            ax_b.annotate("toxic source", (v, yi), xytext=(6, 0),
                          textcoords="offset points", ha="left", va="center",
                          color=SURFACE, fontsize=9, fontweight="bold")
    ax_b.set_title("How the group feels about each member\n"
                   "(kith views — observation log only)",
                   color=INK, fontsize=11, loc="left")
    ax_b.set_xlabel("mean derived sentiment (valence)", color=INK_2ND, fontsize=9)
    ax_b.set_xlim(-1, 1)

    for ax in (ax_a, ax_b):
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout(pad=1.6)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
