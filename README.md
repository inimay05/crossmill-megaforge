# CrossMill — MegaForge

> Blast Furnace Steel Production · POMDP RL Environment · OpenEnv AI Hackathon 2026

---

## What It Is

CrossMill MegaForge simulates control of a blast furnace in a steel production facility. The agent controls 10 process variables — oxygen flow, blast temperature, coke/ore feed rates, limestone addition, tapping schedule, and temperature ramp rate — to produce molten iron at a target carbon grade while minimising coke consumption, reducing CO₂ emissions, and protecting the refractory lining from thermal stress.

Part of the **CrossMill** platform, built for the OpenEnv AI Hackathon 2026 (Meta × Scaler School of Technology). See [crossmill-integration](https://github.com/inimay05/crossmill-integration) for the full platform and cross-industry memory transfer experiments.

---

## Why This Environment

Steel production accounts for roughly 7% of global CO₂ emissions. Current PID-based or human control wastes 15–25% of energy through conservative fixed setpoints that cannot respond to ore-quality variation, refractory degradation, or real-time chemistry shifts. A memory-aware RL agent managing all 18 state variables simultaneously can discover non-intuitive control policies — for example, pre-emptively adjusting oxygen flow in anticipation of silicon pickup — that reduce energy per tonne while hitting quality targets.

The environment was deliberately designed to be structurally isomorphic to SafeNutri (orange juice pasteurization): both require managing thermal inertia, protecting safety margins, optimising competing objectives, and recovering from regime shifts under partial observability. That shared geometry makes cross-industry memory transfer between them non-trivial and operationally meaningful.

---

## State Space — 18-Dimensional

| Index | Variable | Description |
|-------|----------|-------------|
| 0 | `hot_metal_temp` | Hot metal temperature (normalised) |
| 1 | `hearth_temp` | Hearth temperature (normalised) |
| 2 | `blast_temp` | Blast temperature (normalised) |
| 3 | `oxygen_flow` | Oxygen enrichment flow (normalised) |
| 4 | `carbon` | Carbon content in hot metal (normalised) |
| 5 | `silicon` | Silicon content (normalised) |
| 6 | `sulfur` | Sulfur content (normalised) |
| 7 | `top_pressure` | Furnace top pressure (normalised) |
| 8 | `co_co2_ratio` | CO/CO₂ gas utilisation ratio (normalised) |
| 9 | `coke_rate` | Coke consumption rate kg/tonne (normalised) |
| 10 | `ore_coke_ratio` | Ore-to-coke burden ratio (normalised) |
| 11 | `energy` | Energy consumption (normalised) |
| 12 | `production_rate` | Tonnes-per-hour output (normalised) |
| 13 | `wall_temp` | Furnace wall temperature (normalised) |
| 14 | `thermal_stress` | Refractory thermal stress (normalised) |
| 15 | `slag_basicity` | Slag basicity ratio (normalised) |
| 16 | `emissions_co2` | CO₂ emissions rate (normalised) |
| 17 | `equip_health` | Equipment / refractory health (normalised) |

---

## Action Space — 10-Dimensional

**8 continuous actions** (each normalised to [0, 1]):

| Action | Maps to |
|--------|---------|
| `oxygen_flow_delta` | ±20% oxygen flow adjustment |
| `blast_temp` | 900–1300°C |
| `coke_feed_delta` | −20% to +20% coke feed |
| `ore_feed_delta` | −15% to +15% ore feed |
| `limestone_addition` | 0–50 kg/tonne |
| `tapping_interval` | 2–6 hours |
| `temp_ramp_rate` | 5–100°C/h |
| `pressure_target` | 1.5–3.5 bar |

**2 discrete actions:**

| Action | Range | Description |
|--------|-------|-------------|
| `alloy_timing` | 0–3 | Alloy addition timing schedule |
| `emergency_cooling` | 0–1 | Triggers emergency cooling |

---

## Novel POMDP Features

This is a genuine POMDP — feedforward policies are insufficient. LSTM policies are required.

- **3-zone furnace ODE** (top/middle/bottom) with Fe₂O₃ + 3CO → 2Fe + 3CO₂ reduction chemistry solved via `scipy.solve_ivp` (stiff BDF solver). Extreme thermal inertia means a correction applied now may not manifest for many timesteps.
- **Tapping-gated hearth state**: slag basicity and sulfur content are hidden between tapping events. The agent cannot observe hearth chemistry directly — it must infer from surface signals and schedule its own tapping intervals to obtain readings.
- **Delayed chemistry assays**: Easy: 0-step delay. Medium: 15-step delay. Hard: 25-step delay. Carbon and silicon readings arrive late, forcing the agent to act on stale chemistry data.
- **Refractory wear**: `equip_health` degrades over time in proportion to cumulative thermal stress history — aggressive early control has long-term consequences.
- **Supplier regime switches**: coke quality and ore grade can shift mid-episode (Medium+). The agent cannot observe the regime directly.
- **Forced maintenance window** (Hard only): steps 1200–1299 are forced maintenance. Agent actions are overridden to safe-halt mode. The agent must plan around this window.
- **Partial observability** on Hard: only 60% of state fields are visible at any given step.

---

## Reward Design

CrossMill uses a **60/40 hybrid reward**:

- **Dense (60%)**: per-step shaping signal, clamped to ±0.05 to prevent reward farming.
- **Terminal (40%)**: 5-component multi-objective completion score.

Terminal weights:

| Objective | Weight |
|-----------|--------|
| Quality (carbon target accuracy) | 30% |
| Energy efficiency | 25% |
| Production rate | 20% |
| Safety (equipment health) | 15% |
| Environmental (CO₂ emissions) | 10% |

Carbon target is centred at 4.2% with tight tolerance. If carbon error exceeds threshold or any step triggers a catastrophic safety event, the episode score is immediately **zeroed** — not reduced, zeroed. There is no version of "high score but broken furnace."

`gamma = 0.995` — deliberate, to keep delayed terminal-quality objectives learnable on long horizons.

---

## Anti-Reward-Hacking

The CrossMill grader validates results beyond raw score. If carbon error exceeds threshold, or if any step triggers a catastrophic safety event (thermal stress breach, pressure breach, wall temperature breach, reducing-atmosphere breach), the episode score is zeroed regardless of other metrics. An agent cannot claim a high score while damaging the furnace.

---

## Task Tiers

| Task | Episode Length | Sim Time | Grader Target | Key Challenge |
|------|---------------|----------|---------------|---------------|
| `easy` | 200 steps | ~3h | ≥ 0.88 | Deterministic, steady-state production |
| `medium` | 600 steps | ~10h | ≥ 0.80 | Stochastic inputs + 1 supplier regime shift |
| `hard` | 2000 steps | ~33h | ≥ 0.74 | Full campaign + 60% partial obs + forced maintenance window |

---

## OpenEnv Compatibility

Subclasses `openenv.core.Environment`. Exposes the standard OpenEnv HTTP API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe |
| `/metadata` | GET | Environment metadata |
| `/state` | GET | Current environment state |
| `/schema` | GET | JSON schema for action and observation |
| `/reset` | POST | Reset environment, return first observation |
| `/step` | POST | Apply action, return next obs + reward + done |
| `/docs` | GET | Swagger UI |

**Live API**: https://huggingface.co/spaces/kolaai/crossmill-megaforge

---

## Quick Start

```bash
git clone https://github.com/inimay05/crossmill-megaforge
cd crossmill-megaforge
pip install -r requirements.txt

# Run sanity tests
python -m tests.test_env

# Score the hand-coded PID baseline
python -m app.baseline_agent
```

---

## Part of CrossMill

MegaForge is one of two environments in the CrossMill cross-industry RL platform.

* Integration layer and memory transfer: https://github.com/inimay05/crossmill-integration
* Companion environment (SafeNutri): https://github.com/inimay05/crossmill-safenutri
* Live Gradio demo: https://huggingface.co/spaces/kolaai/crossmill-integration
