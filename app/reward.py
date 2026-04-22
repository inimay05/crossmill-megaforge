import numpy as np

from app.config import (
    # Targets and tolerances
    CARBON_TARGET_PCT, CARBON_TOLERANCE_PCT, CARBON_MAX_ERR,
    SILICON_TARGET_PCT, SILICON_MAX_ERR,
    SULFUR_MAX_PCT,
    # Reward structure
    DENSE_CLAMP_LOW, DENSE_CLAMP_HIGH,
    SAFETY_PENALTY, PROGRESS_WEIGHT, TERMINAL_WEIGHT,
    W_QUALITY, W_ENERGY, W_PRODUCTION, W_SAFETY_EQUIP, W_ENVIRONMENTAL,
    # Scoring constants
    HOT_METAL_TEMP_TARGET, HOT_METAL_TEMP_TOLERANCE,
    PRODUCTION_TARGET_TPH,
    PRESSURE_KNEE_BAR, PRESSURE_SCALE,
    THERMAL_STRESS_SAFE_ZONE, THERMAL_STRESS_WARN,
    EMISSIONS_SPIKE_THRESHOLD,
    WALL_TEMP_WARN_C,
    CO_CO2_OPTIMAL_LOW, CO_CO2_OPTIMAL_HIGH,
    # Ranges and timing
    RANGES, TAPPING_EVERY_STEPS,
)
from app.models import Reward, RewardBreakdown


# --------------------------------------------------------------------------- #
# Dense progress reward                                                         #
# --------------------------------------------------------------------------- #

def compute_dense_progress(prev_raw: dict, next_raw: dict, action: dict) -> float:
    """
    Physics-informed partial credit per env step.
    Returns a scalar clamped to [DENSE_CLAMP_LOW, DENSE_CLAMP_HIGH].
    """
    r = 0.0

    # ── Positive signals ───────────────────────────────────────────────────── #

    # Carbon content moving toward target (4.2 ± 0.05)
    prev_cerr = abs(prev_raw['carbon_pct'] - CARBON_TARGET_PCT)
    next_cerr = abs(next_raw['carbon_pct'] - CARBON_TARGET_PCT)
    if next_cerr < prev_cerr or next_cerr <= CARBON_TOLERANCE_PCT:
        r += 0.02

    # Coke rate decreasing (efficiency gain)
    if next_raw['coke_rate_kgpt'] < prev_raw['coke_rate_kgpt']:
        r += 0.02

    # CO₂ emissions decreasing
    if next_raw['emissions_CO2_kgpt'] < prev_raw['emissions_CO2_kgpt']:
        r += 0.02

    # Thermal stress in safe zone (smooth operation)
    if next_raw['thermal_stress'] < THERMAL_STRESS_SAFE_ZONE:
        r += 0.01

    # Production rate improving
    if next_raw['production_rate_tph'] > prev_raw['production_rate_tph']:
        r += 0.01

    # CO/CO₂ ratio in optimal reduction band
    if CO_CO2_OPTIMAL_LOW <= next_raw['co_co2_ratio'] <= CO_CO2_OPTIMAL_HIGH:
        r += 0.01

    # ── Penalty signals ────────────────────────────────────────────────────── #

    # Thermal stress approaching hard safety limit
    if next_raw['thermal_stress'] > THERMAL_STRESS_WARN:
        r -= 0.02

    # Emissions spike — compliance breach imminent
    if next_raw['emissions_CO2_kgpt'] > EMISSIONS_SPIKE_THRESHOLD:
        r -= 0.03

    # Refractory wall temperature in damage zone
    if next_raw['wall_temp_C'] > WALL_TEMP_WARN_C:
        r -= 0.02

    return float(np.clip(r, DENSE_CLAMP_LOW, DENSE_CLAMP_HIGH))


# --------------------------------------------------------------------------- #
# Terminal score                                                                #
# --------------------------------------------------------------------------- #

def compute_terminal_score(trajectory: list) -> tuple:
    """
    5-component weighted terminal score in [0, 1].
    Returns (total_score, component_dict).
    """
    final = trajectory[-1]
    n     = len(trajectory)

    # ── Quality (W = 0.30) ─────────────────────────────────────────────────── #
    carbon_err   = abs(final['carbon_pct'] - CARBON_TARGET_PCT)
    carbon_score = float(np.clip(1.0 - carbon_err / CARBON_MAX_ERR, 0.0, 1.0))

    silicon_err   = abs(final['silicon_pct'] - SILICON_TARGET_PCT)
    silicon_score = float(np.clip(1.0 - silicon_err / SILICON_MAX_ERR, 0.0, 1.0))

    sulfur_score  = float(np.clip(1.0 - final['sulfur_pct'] / SULFUR_MAX_PCT, 0.0, 1.0))

    strength_proxy = float(np.clip(
        0.5 * carbon_score + 0.5 * (1.0 - silicon_err), 0.0, 1.0))

    quality = float(np.clip(
        0.35 * carbon_score
        + 0.25 * silicon_score
        + 0.25 * sulfur_score
        + 0.15 * strength_proxy,
        0.0, 1.0
    ))

    # ── Energy (W = 0.25) ──────────────────────────────────────────────────── #
    _coke_lo, _coke_hi = RANGES['coke_rate_kgpt']          # (300, 600)
    mean_coke   = float(np.mean([s['coke_rate_kgpt'] for s in trajectory]))
    coke_score  = float(np.clip((_coke_hi - mean_coke) / (_coke_hi - _coke_lo), 0.0, 1.0))

    _egy_lo, _egy_hi = RANGES['energy_GJpt']               # (18, 25)
    mean_energy  = float(np.mean([s['energy_GJpt'] for s in trajectory]))
    energy_score = float(np.clip((_egy_hi - mean_energy) / (_egy_hi - _egy_lo), 0.0, 1.0))

    heat_util = float(np.clip(
        1.0 - abs(final['hot_metal_temp_C'] - HOT_METAL_TEMP_TARGET) / HOT_METAL_TEMP_TOLERANCE,
        0.0, 1.0
    ))

    energy = float(np.clip(
        0.4 * coke_score + 0.4 * energy_score + 0.2 * heat_util,
        0.0, 1.0
    ))

    # ── Production (W = 0.20) ──────────────────────────────────────────────── #
    mean_prod  = float(np.mean([s['production_rate_tph'] for s in trajectory]))
    prod_score = float(np.clip(mean_prod / PRODUCTION_TARGET_TPH, 0.0, 1.0))

    uptime = 1.0   # placeholder — no downtime events tracked; future work

    tapping_count = sum(1 for s in trajectory if s.get('_tapping_this_step', False))
    expected_taps = n / TAPPING_EVERY_STEPS
    tapping_bonus = float(np.clip(
        1.0 - abs(tapping_count - expected_taps) / max(expected_taps, 1.0),
        0.0, 1.0
    ))

    production = float(np.clip(
        0.6 * prod_score + 0.2 * uptime + 0.2 * tapping_bonus,
        0.0, 1.0
    ))

    # ── Safety & Equipment (W = 0.15) ──────────────────────────────────────── #
    _stress_max  = RANGES['thermal_stress'][1]              # 100.0
    max_stress   = max(s['thermal_stress'] for s in trajectory)
    stress_score = float(np.clip(1.0 - max_stress / _stress_max, 0.0, 1.0))

    max_pressure    = max(s['top_pressure_bar'] for s in trajectory)
    pressure_score  = float(np.clip(
        1.0 - max(0.0, max_pressure - PRESSURE_KNEE_BAR) / PRESSURE_SCALE,
        0.0, 1.0
    ))

    equip_score = final['equip_health_pct'] / 100.0

    safety_equip = float(np.clip(
        0.4 * stress_score + 0.3 * pressure_score + 0.3 * equip_score,
        0.0, 1.0
    ))

    # ── Environmental (W = 0.10) ───────────────────────────────────────────── #
    _co2_lo, _co2_hi = RANGES['emissions_CO2_kgpt']        # (1400, 2200)
    mean_co2  = float(np.mean([s['emissions_CO2_kgpt'] for s in trajectory]))
    co2_score = float(np.clip((_co2_hi - mean_co2) / (_co2_hi - _co2_lo), 0.0, 1.0))

    nox_proxy  = 1.0   # placeholder
    waste_heat = min(1.0, final['equip_health_pct'] / 100.0 + 0.1)

    environmental = float(np.clip(
        0.6 * co2_score + 0.2 * nox_proxy + 0.2 * waste_heat,
        0.0, 1.0
    ))

    # ── Weighted total ─────────────────────────────────────────────────────── #
    total = float(np.clip(
        W_QUALITY      * quality
        + W_ENERGY     * energy
        + W_PRODUCTION * production
        + W_SAFETY_EQUIP   * safety_equip
        + W_ENVIRONMENTAL  * environmental,
        0.0, 1.0
    ))

    components = {
        'quality':       quality,
        'energy':        energy,
        'production':    production,
        'safety_equip':  safety_equip,
        'environmental': environmental,
    }
    return total, components


# --------------------------------------------------------------------------- #
# Main reward entry point                                                       #
# --------------------------------------------------------------------------- #

def compute_reward(prev_raw: dict, next_raw: dict, action: dict,
                   done: bool, trajectory: list,
                   safety_violation: bool) -> Reward:
    """
    Combines dense progress, terminal score, and safety penalty.
    Formula:
        total = PROGRESS_WEIGHT * dense
              + TERMINAL_WEIGHT * terminal
              + safety_pen
    """
    dense = compute_dense_progress(prev_raw, next_raw, action)

    if done and not safety_violation:
        terminal, _ = compute_terminal_score(trajectory)
    else:
        terminal = 0.0

    safety_pen = SAFETY_PENALTY if safety_violation else 0.0

    dense_contribution = 0.0 if safety_violation else PROGRESS_WEIGHT * dense
    total = dense_contribution + TERMINAL_WEIGHT * terminal + safety_pen

    return Reward(
        value=total,
        breakdown=RewardBreakdown(
            dense_progress=dense,
            terminal_score=terminal,
            safety_penalty=safety_pen,
            total=total,
        )
    )


# --------------------------------------------------------------------------- #
# Sanity check                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':

    def _state(**kw):
        """Minimal steady-state raw_state with optional overrides."""
        base = {
            'hot_metal_temp_C':    1500.0,
            'hearth_temp_C':       1520.0,
            'blast_temp_C':        1100.0,
            'oxygen_flow_Nm3pm':   120.0,
            'carbon_pct':          4.2,
            'silicon_pct':         0.8,
            'sulfur_pct':          0.03,
            'top_pressure_bar':    2.5,
            'co_co2_ratio':        2.2,
            'coke_rate_kgpt':      450.0,
            'ore_coke_ratio':      3.0,
            'energy_GJpt':         22.0,
            'production_rate_tph': 250.0,
            'wall_temp_C':         300.0,
            'thermal_stress':      20.0,
            'slag_basicity':       1.0,
            'emissions_CO2_kgpt':  1800.0,
            'equip_health_pct':    92.0,
        }
        base.update(kw)
        return base

    dummy_action = {
        'oxygen_flow_delta_pct':  0.0,
        'blast_temp_C':           1200.0,
        'coke_feed_delta_pct':    0.0,
        'ore_feed_delta_pct':     0.0,
        'limestone_addition_pct': 10.0,
        'tapping_interval_h':     4.0,
        'temp_ramp_rate_Cph':     20.0,
        'pressure_target_bar':    2.5,
        'alloy_timing':           1,
        'emergency_cooling':      0,
    }

    SEP = '=' * 58

    # ── Check 1: dense progress on improving step ───────────────────────────── #
    print(SEP)
    print("CHECK 1 — compute_dense_progress (improving step)")
    prev = _state(carbon_pct=4.30, coke_rate_kgpt=460.0,
                  emissions_CO2_kgpt=1900.0, thermal_stress=50.0,
                  production_rate_tph=240.0, co_co2_ratio=2.2)
    nxt  = _state(carbon_pct=4.25, coke_rate_kgpt=455.0,
                  emissions_CO2_kgpt=1850.0, thermal_stress=45.0,
                  production_rate_tph=250.0, co_co2_ratio=2.2)
    dense = compute_dense_progress(prev, nxt, dummy_action)
    print(f"  dense_progress = {dense:.4f}  (all 6 positive => clamped to +0.05)")
    assert DENSE_CLAMP_LOW <= dense <= DENSE_CLAMP_HIGH, \
        f"dense {dense} outside clamp range!"
    print(f"  ASSERT in [{DENSE_CLAMP_LOW}, {DENSE_CLAMP_HIGH}]: PASS")

    # ── Check 2: terminal score on high-quality episode ─────────────────────── #
    print(f"\n{SEP}")
    print("CHECK 2 — compute_terminal_score (200-step near-perfect episode)")
    trajectory = [_state() for _ in range(199)]
    trajectory.append(_state(
        carbon_pct=4.20, silicon_pct=0.8, sulfur_pct=0.03,
        equip_health_pct=92.0,
    ))

    total_ts, comps = compute_terminal_score(trajectory)
    print(f"  quality      = {comps['quality']:.4f}   (target >= 0.80)")
    print(f"  energy       = {comps['energy']:.4f}   (target >= 0.65)")
    print(f"  production   = {comps['production']:.4f}")
    print(f"  safety_equip = {comps['safety_equip']:.4f}")
    print(f"  environmental= {comps['environmental']:.4f}")
    print(f"  TOTAL        = {total_ts:.4f}   (target ~0.75+)")

    # ── Check 3: full Reward on terminal step ────────────────────────────────── #
    print(f"\n{SEP}")
    print("CHECK 3 — compute_reward (done=True, safety_violation=False)")
    reward_ok = compute_reward(
        prev, nxt, dummy_action,
        done=True, trajectory=trajectory,
        safety_violation=False,
    )
    print(f"  dense_progress  = {reward_ok.breakdown.dense_progress:.4f}")
    print(f"  terminal_score  = {reward_ok.breakdown.terminal_score:.4f}")
    print(f"  safety_penalty  = {reward_ok.breakdown.safety_penalty:.4f}")
    print(f"  total           = {reward_ok.value:.4f}")
    assert reward_ok.breakdown.safety_penalty == 0.0
    assert reward_ok.breakdown.terminal_score > 0.0
    print(f"  ASSERT no safety penalty, terminal > 0: PASS")

    # ── Check 4: safety violation ────────────────────────────────────────────── #
    print(f"\n{SEP}")
    print("CHECK 4 — compute_reward (safety_violation=True)")
    # Use penalty-heavy states so dense is clamped to DENSE_CLAMP_LOW (-0.05)
    bad = _state(thermal_stress=80.0, emissions_CO2_kgpt=2100.0, wall_temp_C=390.0)
    reward_viol = compute_reward(
        bad, bad, dummy_action,
        done=True, trajectory=trajectory,
        safety_violation=True,
    )
    print(f"  dense_progress  = {reward_viol.breakdown.dense_progress:.4f}")
    print(f"  terminal_score  = {reward_viol.breakdown.terminal_score:.4f}  "
          f"(suppressed — safety violation)")
    print(f"  safety_penalty  = {reward_viol.breakdown.safety_penalty:.4f}")
    print(f"  total           = {reward_viol.value:.4f}")
    assert reward_viol.value <= -1.0, \
        f"Expected total <= -1.0, got {reward_viol.value:.4f}"
    assert reward_viol.breakdown.safety_penalty == SAFETY_PENALTY
    print(f"  ASSERT total <= -1.0: PASS")
    print(f"  ASSERT safety_penalty == {SAFETY_PENALTY}: PASS")
    print(SEP)
