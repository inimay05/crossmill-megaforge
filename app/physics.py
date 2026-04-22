import numpy as np
from scipy.integrate import solve_ivp

from app.config import (
    R_GAS,
    REDUCTION_TOP, REDUCTION_MIDDLE, REDUCTION_BOTTOM,
    U_TOP, U_MIDDLE, U_BOTTOM, CP_IRON,
    REFRACTORY_WEAR_RATE, WEAR_TEMP_COEF,
    ODE_METHOD, ODE_RTOL, ODE_ATOL,
    STEP_INTERVAL_S, RANGES, CARBON_TARGET_PCT,
)

_ZONE_PARAMS = (REDUCTION_TOP, REDUCTION_MIDDLE, REDUCTION_BOTTOM)
_ZONE_U      = (U_TOP, U_MIDDLE, U_BOTTOM)


def physics_step(raw_state: dict, raw_action: dict, dt: float, rng) -> dict:
    """
    Run one env-step of 3-zone furnace physics.

    ODE state vector (length 9):
      y[0:3]  = T_top, T_middle, T_bottom        (°C)
      y[3:6]  = red_top, red_middle, red_bottom  (fraction Fe2O3 reduced, 0..1)
      y[6]    = C_hot_metal   (carbon_pct)
      y[7]    = Si_hot_metal  (silicon_pct)
      y[8]    = S_slag        (sulfur_pct)
    """
    state = dict(raw_state)

    zone_temps = list(state.get('_zone_temps', [900.0, 1300.0, 1500.0]))
    zone_red   = list(state.get('_zone_reduction_pct', [0.0, 0.0, 0.0]))

    y0 = np.array([
        zone_temps[0], zone_temps[1], zone_temps[2],
        zone_red[0],   zone_red[1],   zone_red[2],
        state.get('carbon_pct',  4.2),
        state.get('silicon_pct', 0.8),
        state.get('sulfur_pct',  0.04),
    ], dtype=float)

    # Cache scalars used inside the closure
    co_co2     = float(state['co_co2_ratio'])
    slag_bas   = float(state['slag_basicity'])
    coke_delta = float(raw_action.get('coke_feed_delta_pct', 0.0))

    def rhs(t, y, action):
        dy = np.zeros(9)

        for z in range(3):
            T_z    = y[z]
            params = _ZONE_PARAMS[z]
            U_z    = _ZONE_U[z]

            T_kelvin = max(T_z, 300.0) + 273.15
            r_z = (params['k0']
                   * np.exp(-params['Ea'] / (R_GAS * T_kelvin))
                   * max(co_co2, 0.1) ** params['n_co']
                   * (1.0 - y[3 + z]))

            # Gas is hotter deeper into the furnace
            T_gas  = action['blast_temp_C'] + 200.0 * z
            dT_dt  = U_z * 5.0 * (T_gas - T_z) / (1200.0 * CP_IRON) - r_z * 50.0

            dy[z]     = dT_dt
            dy[3 + z] = r_z

        # Carbon pickup at bottom zone gas contact (gated on melt temperature)
        melt_factor = float(np.clip((y[2] - 1400.0) / 100.0, 0.0, 1.0))
        dy[6] = (melt_factor * (0.0002 * co_co2 * (coke_delta / 100.0 + 1.0))
                 - 0.001 * (y[6] - CARBON_TARGET_PCT))

        # Silicon pickup driven by bottom-zone superheat
        dy[7] = 0.00001 * max(0.0, y[2] - 1500.0) ** 1.2 - 0.002 * y[7]

        # Sulfur absorption into slag
        dy[8] = -0.005 * slag_bas * (1500.0 / max(y[2], 1400.0)) * y[8]

        return dy

    emergency = (raw_action.get('emergency_cooling', 0) == 1)

    if emergency:
        def rhs_cooling(t, y, action):
            dy = rhs(t, y, action)
            dy[2] = -50.0   # forced rapid bottom-zone cooling
            return dy
        ode_func = rhs_cooling
    else:
        ode_func = rhs

    sol = solve_ivp(
        ode_func, [0.0, dt], y0,
        method=ODE_METHOD, rtol=ODE_RTOL, atol=ODE_ATOL,
        args=(raw_action,),
    )

    y_final = sol.y[:, -1].copy()

    # Apply turbulence noise once per env step (not inside the ODE, where the
    # solver calls rhs multiple times per step and would accumulate drift).
    y_final[0:3] += rng.normal(0.0, 0.5, size=3)

    # ------------------------------------------------------------------ #
    # Helper: clip a value to its physical range                           #
    # ------------------------------------------------------------------ #
    def _clip(key: str, val: float) -> float:
        lo, hi = RANGES[key]
        return float(np.clip(val, lo, hi))

    # ---- Zone internals ----
    state['_zone_temps']         = np.clip(y_final[0:3], 300.0, 2000.0).tolist()
    state['_zone_reduction_pct'] = np.clip(y_final[3:6], 0.0,   1.0  ).tolist()

    zone_temps_final = state['_zone_temps']
    zone_red_final   = state['_zone_reduction_pct']

    # ---- Temperatures ----
    state['hot_metal_temp_C'] = _clip('hot_metal_temp_C', y_final[2])
    state['hearth_temp_C']    = _clip('hearth_temp_C',    y_final[2] + 30.0)

    # ---- Chemistry ----
    state['carbon_pct']  = _clip('carbon_pct',  y_final[6])
    state['silicon_pct'] = _clip('silicon_pct', y_final[7])
    state['sulfur_pct']  = _clip('sulfur_pct',  y_final[8])

    # ---- Oxygen flow (clamped around old value ± 30 %) ----
    old_o2   = float(state['oxygen_flow_Nm3pm'])
    o2_frac  = raw_action.get('oxygen_flow_delta_pct', 0.0) / 100.0
    state['oxygen_flow_Nm3pm'] = _clip('oxygen_flow_Nm3pm', old_o2 * (1.0 + o2_frac))

    # ---- Blast temp: 60-s thermal-mass lag toward target ----
    old_blast = float(state['blast_temp_C'])
    tgt_blast = float(raw_action['blast_temp_C'])
    state['blast_temp_C'] = _clip('blast_temp_C', old_blast + (tgt_blast - old_blast) * 0.3)

    # ---- Coke rate ----
    old_coke = float(state['coke_rate_kgpt'])
    state['coke_rate_kgpt'] = _clip(
        'coke_rate_kgpt',
        old_coke * (1.0 + raw_action.get('coke_feed_delta_pct', 0.0) / 100.0)
    )

    # ---- Ore / coke ratio ----
    old_orc = float(state['ore_coke_ratio'])
    state['ore_coke_ratio'] = _clip(
        'ore_coke_ratio',
        old_orc * (1.0 + raw_action.get('ore_feed_delta_pct', 0.0) / 100.0)
    )

    # ---- Top pressure: 0.4-weight lag toward target ----
    old_p  = float(state['top_pressure_bar'])
    p_tgt  = float(raw_action.get('pressure_target_bar', old_p))
    state['top_pressure_bar'] = _clip('top_pressure_bar', old_p + (p_tgt - old_p) * 0.4)

    # ---- CO / CO2 ratio (lag toward target) ----
    o2_delta      = raw_action.get('oxygen_flow_delta_pct', 0.0)
    target_co_co2 = (2.2 + 0.01 * o2_delta
                     - 0.5 * float(np.mean(y_final[3:6]))
                     + rng.normal(0.0, 0.05))
    new_co_co2    = state['co_co2_ratio'] + (target_co_co2 - state['co_co2_ratio']) * 0.3
    state['co_co2_ratio'] = float(np.clip(new_co_co2, *RANGES['co_co2_ratio']))

    # ---- Energy (GJ / tonne) ----
    energy_est = (15.0
                  + 0.005 * state['blast_temp_C']
                  + 0.005 * state['coke_rate_kgpt'])
    state['energy_GJpt'] = _clip('energy_GJpt', energy_est)

    # ---- Production rate ----
    equip_frac = state['equip_health_pct'] / 100.0
    prod_est   = state['oxygen_flow_Nm3pm'] * 2.0 * equip_frac
    state['production_rate_tph'] = _clip('production_rate_tph', prod_est)

    # ---- Wall temperature (from mean zone temps) ----
    state['wall_temp_C'] = float(np.clip(
        np.mean(y_final[0:3]) * 0.25 - 20.0,
        *RANGES['wall_temp_C']
    ))

    # ---- Thermal stress ----
    dT_per_sec = np.abs(y_final[:3] - y0[:3]) / dt   # C / s
    max_dT     = float(np.max(dT_per_sec))
    old_stress = float(state.get('thermal_stress', 20.0))
    if max_dT > 0.5:
        new_stress = min(100.0, max_dT * 20.0)
    else:
        new_stress = old_stress * 0.95
    state['thermal_stress'] = float(np.clip(new_stress, 0.0, 100.0))

    # ---- Slag basicity (lag toward target) ----
    lime_pct   = float(raw_action.get('limestone_addition_pct', 10.0))
    target_bas = 0.8 + 0.03 * lime_pct / 10.0
    new_bas    = (state['slag_basicity']
                  + (target_bas - state['slag_basicity']) * 0.1
                  + rng.normal(0.0, 0.02))
    state['slag_basicity'] = float(np.clip(new_bas, *RANGES['slag_basicity']))

    # ---- CO2 emissions ----
    co2_est = 1500.0 + 3.0 * state['coke_rate_kgpt'] + 0.3 * state['energy_GJpt']
    state['emissions_CO2_kgpt'] = _clip('emissions_CO2_kgpt', co2_est)

    # ---- Equipment health ----
    wall_T    = float(state['wall_temp_C'])
    wear_mult = 1.0 + max(0.0, (wall_T - 300.0) / 100.0) * WEAR_TEMP_COEF
    if emergency:
        wear_mult *= 2.0
    state['equip_health_pct'] = _clip(
        'equip_health_pct',
        float(state['equip_health_pct']) - REFRACTORY_WEAR_RATE * wear_mult
    )

    return state


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def normalise(raw_state: dict) -> dict:
    """Map each variable v to (v - lo) / (hi - lo) clipped [0,1].
    Keys starting with '_' (internal hidden state) are skipped."""
    norm = {}
    for k, v in raw_state.items():
        if k.startswith('_'):
            continue
        lo, hi = RANGES[k]
        norm[k] = float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))
    return norm


def denormalise(norm_state: dict) -> dict:
    """Inverse of normalise."""
    raw = {}
    for k, v in norm_state.items():
        lo, hi = RANGES[k]
        raw[k] = float(lo + v * (hi - lo))
    return raw


def default_initial_state() -> dict:
    """Typical steady-state starting conditions for a well-run campaign."""
    return {
        'hot_metal_temp_C':    1500.0,
        'hearth_temp_C':       1520.0,
        'blast_temp_C':        1100.0,
        'oxygen_flow_Nm3pm':   120.0,
        'carbon_pct':          4.2,
        'silicon_pct':         0.8,
        'sulfur_pct':          0.04,
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
        'equip_health_pct':    95.0,
        # Hidden zone internals
        '_zone_temps':          [900.0, 1300.0, 1500.0],
        '_zone_reduction_pct':  [0.0,   0.0,   0.0],
    }


# ------------------------------------------------------------------ #
# Sanity check                                                         #
# ------------------------------------------------------------------ #

if __name__ == '__main__':
    import pprint

    rng   = np.random.default_rng(42)
    state = default_initial_state()
    action = {
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

    print("=== Initial state ===")
    pprint.pprint(state)

    for step in range(10):
        state = physics_step(state, action, STEP_INTERVAL_S, rng)

    print("\n=== State after 10 steps ===")
    pprint.pprint(state)

    print("\n=== Zone reduction progress ===")
    for zone, pct in zip(('top', 'middle', 'bottom'), state['_zone_reduction_pct']):
        print(f"  {zone:8s}: {pct:.6f} ({pct*100:.4f}% Fe2O3 reduced)")
