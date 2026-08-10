# Launch playbook

## Pre-flight (must be done BEFORE posting)

- [ ] v0.2.0 uploaded to PyPI (`twine upload dist/*` — needs token in ~/.pypirc)
- [ ] `pip install kith-ai` in a clean venv pulls 0.2.0 and README example runs
- [ ] CI badge green on GitHub main
- [ ] contagion.png renders in the GitHub README view (relative path check)
- [ ] Set repo topics: `agents`, `multi-agent`, `memory`, `llm`, `trust`, `langgraph`
- [ ] Pin the two demo READMEs' numbers — rerun both demos once to confirm

## Timing

- **HN**: Tuesday–Thursday, 8–10am US Eastern (北京时间 20:00–22:00) is the
  highest-throughput window for Show HN. Avoid US holidays. One shot — a
  Show HN cannot be reposted quickly, so pre-flight matters.
- **掘金**: 工作日早 9–10 点或晚 8–9 点。可以和 HN 同日(先掘金后 HN,
  时区正好顺)。
- Cross-post to X/Twitter after HN post is live (link the HN thread, not
  just the repo — HN upvotes come from the thread).

## Engagement rules (first 3 hours are decisive on HN)

- Reply to EVERY substantive comment, fast, technically, without
  defensiveness. Concede real gaps ("good catch — filed #N").
- The FAQ prep section in show-hn.md covers the 5 predictable attacks:
  bandit-reduction, pseudoscience, why-not-Mem0, multi-agent-skepticism,
  scope-spoofing.
- Do NOT argue benchmarks beyond what the seeded demos claim. The claim
  is narrow and reproducible; keep it narrow.
- If someone from Mem0/Zep/LangChain comments — be generous, they are
  potential integration partners, not rivals ("kith composes with X" is
  the party line and it is true).

## What NOT to do

- Don't mention job-seeking anywhere near the launch.
- Don't link the hermes PR as self-promotion; mention the review lesson
  (it's in the post) and let people find the PR themselves.
- Don't submit to r/MachineLearning same day (their crosspost norms are
  hostile); r/LocalLLaMA a few days later is fine.

## Success metrics (for our own calibration, 2 weeks)

- Floor: 100+ GitHub stars, 3+ substantive issues from strangers
- Good: 300+ stars, an integration request from a framework we didn't
  build an adapter for, 1+ external PR
- Great: front page HN (top 10 for 2h+), 1000+ stars, inbound from a lab
  or agent-infra company

## After the wave

- Convert the best HN/掘金 questions into FAQ.md + design-doc updates
- File every real gap raised as a GitHub issue same day (public roadmap
  momentum)
- Week after: write the "what I learned launching" 复盘 in Chinese —
  second content wave for basically free
