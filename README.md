# CrossMill — MegaForge

**Blast Furnace Steel Production · POMDP RL Environment · OpenEnv-Compatible**

CrossMill MegaForge is a reinforcement-learning environment that simulates a
blast furnace producing molten iron via the Fe₂O₃ + 3CO → 2Fe + 3CO₂ reduction
reaction. The agent controls oxygen flow, blast temperature, coke / ore feed
rates, limestone addition, tapping schedule, and temperature ramp rate to hit
target steel grades while minimising energy consumption, CO₂ emissions, and
thermal stress on the refractory lining.

Part of the **CrossMill** family of industrial RL environments, built for the
OpenEnv AI Hackathon 2026 (Meta × Scaler School of Technology).

## Why This Environment

Steel production accounts for roughly 7 % of global CO₂ emissions. Current
PID-based or human control wastes 15–25 % of energy through conservative fixed
setpoints that cannot respond to ore-quality variation, refractory degradation,
or real-time chemistry shifts. A memory-aware RL agent managing all 18 state
variables simultaneously can discover non-intuitive control policies — e.g.,
pre-emptively adjusting oxygen flow in anticipation of silicon pickup — that
reduce energy per tonne by 15–25 % while hitting quality targets.

## Features

- **18-dim state / 10-dim action** physics-grounded POMDP
- **3-zone furnace ODE** (top / middle / bottom) with reduction chemistry via
  `scipy.solve_ivp` (stiff BDF)
- **Tapping-gated hearth state** — slag basicity and sulfur content are hidden
  between tapping events, forcing genuine inference
- **Delayed chemistry assays, refractory wear, and supplier regime switches**
  make the task a true POMDP — LSTM policies required
- **Hybrid 60/40 reward** (dense progress + 5-component terminal) with per-step
  clamping to prevent reward hacking
- **Hard safety constraints** (pressure, thermal stress, wall temperature,
  reducing-atmosphere breach) with episode termination on breach
- **Easy / Medium / Hard tasks** with programmatic graders returning a single
  `grader_score ∈ [0, 1]`
- **OpenEnv-compatible**: subclasses `openenv.Env`, exposes full manifest

## Quick Start

```bash
git clone https://huggingface.co/<your-username>/crossmill-megaforge
cd crossmill-megaforge
pip install -r requirements.txt

# Run the sanity tests
python -m tests.test_env

# Score the hand-coded PID baseline
python -m app.baseline_agent

# Train an LSTM policy (100k steps ≈ 30 min on Colab T4)
python scripts/train_trl.py --task easy --timesteps 100000
```

## Task Tiers

| Task   | Episode Length     | Grader Target | Key Challenge                                              |
|--------|--------------------|---------------|------------------------------------------------------------|
| easy   | 200 steps  (~3 h)  | ≥ 0.88        | Deterministic, steady-state production                     |
| medium | 600 steps  (~10 h) | ≥ 0.80        | Stochastic inputs + 1 supplier regime shift                |
| hard   | 2000 steps (~33 h) | ≥ 0.74        | Full campaign + 60 % partial obs + forced maintenance window |

## Citation

If you use CrossMill MegaForge in your research, please cite the OpenEnv
AI Hackathon 2026 submission.

## License

MIT
