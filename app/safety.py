"""
Hard safety constraint checker for CrossMill-MegaForge.

Pure-function module: all checks are side-effect-free predicates on raw_state.
Call `check_intra_episode` once per env step; call `check_quality_at_termination`
only if done=True.
"""

from app.config import (
    PRESSURE_MAX_BAR,
    THERMAL_STRESS_MAX,
    WALL_TEMP_MAX_C,
    HOT_METAL_TEMP_MAX_C,
    CO_CO2_RATIO_MIN,
    CARBON_TARGET_PCT,
)

# --------------------------------------------------------------------------- #
# Public API                                                                    #
# --------------------------------------------------------------------------- #

def check_hard_limits(raw_state: dict) -> tuple:
    """
    Run ALL hard safety checks on the raw (un-normalised) state.
    Returns (violation_flag, reason_string).

    Returns
    -------
    (False, '')                   if all OK
    (True, 'PRESSURE_OVER_MAX')   if top_pressure_bar > PRESSURE_MAX_BAR (3.8)
    (True, 'THERMAL_STRESS_MAX')  if thermal_stress > THERMAL_STRESS_MAX (90)
    (True, 'WALL_TEMP_MAX')       if wall_temp_C > WALL_TEMP_MAX_C (420)
    (True, 'HOT_METAL_TEMP_MAX')  if hot_metal_temp_C > HOT_METAL_TEMP_MAX_C (1680)
    (True, 'REDUCING_ATMOS_LOST') if co_co2_ratio < CO_CO2_RATIO_MIN (1.0)

    First match wins (check in order above: most physically dangerous first).
    """
    s = raw_state

    # Check 1: Blast pressure relief risk
    if s['top_pressure_bar'] > PRESSURE_MAX_BAR:
        return True, 'PRESSURE_OVER_MAX'

    # Check 2: Refractory crack risk
    if s['thermal_stress'] > THERMAL_STRESS_MAX:
        return True, 'THERMAL_STRESS_MAX'

    # Check 3: Catastrophic refractory burn-through
    if s['wall_temp_C'] > WALL_TEMP_MAX_C:
        return True, 'WALL_TEMP_MAX'

    # Check 4: Hearth damage / slip risk
    if s['hot_metal_temp_C'] > HOT_METAL_TEMP_MAX_C:
        return True, 'HOT_METAL_TEMP_MAX'

    # Check 5: Reducing atmosphere lost; oxidation risk
    if s['co_co2_ratio'] < CO_CO2_RATIO_MIN:
        return True, 'REDUCING_ATMOS_LOST'

    return False, ''


def check_quality_at_termination(raw_state: dict) -> bool:
    """
    At the terminal step ONLY, also check catastrophic quality failure:
    carbon_pct off by more than 0.3 AND sulfur_pct > 0.08.

    Returns
    -------
    bool
        True if the episode ended with steel that is completely off-grade
        (not just low-quality).
    """
    carbon_err = abs(raw_state['carbon_pct'] - CARBON_TARGET_PCT)
    sulfur_pct = raw_state['sulfur_pct']
    return carbon_err > 0.3 and sulfur_pct > 0.08


def check_intra_episode(raw_state: dict) -> tuple:
    """
    What the env uses on every step. Same as check_hard_limits ---
    all 5 hard physical limits apply intra-episode.

    Returns
    -------
    tuple[bool, str]
        (violation_flag, reason_string) as per check_hard_limits.
    """
    return check_hard_limits(raw_state)


# --------------------------------------------------------------------------- #
# Sanity check                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':

    def _state(**kw):
        """Build a minimal raw_state with optional overrides."""
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

    SEP = '=' * 60

    # Build 6 test states: one safe, one per violation
    print(SEP)
    print("=== SAFETY CHECKS ===\n")

    # Test 1: Safe state
    print("TEST 1 -- Safe nominal state")
    safe = _state()
    viol, reason = check_intra_episode(safe)
    print(f"  violated={viol}, reason='{reason}'")
    assert not viol, f"Expected safe, got violation: {reason}"
    assert reason == '', f"Expected reason='', got '{reason}'"
    print("  ASSERT (False, ''): PASS\n")

    # Test 2: Pressure breach
    print("TEST 2 -- Pressure over max (3.9 > 3.8)")
    press = _state(top_pressure_bar=3.9)
    viol, reason = check_intra_episode(press)
    print(f"  violated={viol}, reason='{reason}'")
    assert viol, "Expected pressure violation"
    assert reason == 'PRESSURE_OVER_MAX', f"Expected 'PRESSURE_OVER_MAX', got '{reason}'"
    print("  ASSERT (True, 'PRESSURE_OVER_MAX'): PASS\n")

    # Test 3: Thermal stress breach
    print("TEST 3 -- Thermal stress over max (91 > 90)")
    stress = _state(thermal_stress=91.0)
    viol, reason = check_intra_episode(stress)
    print(f"  violated={viol}, reason='{reason}'")
    assert viol, "Expected thermal stress violation"
    assert reason == 'THERMAL_STRESS_MAX', f"Expected 'THERMAL_STRESS_MAX', got '{reason}'"
    print("  ASSERT (True, 'THERMAL_STRESS_MAX'): PASS\n")

    # Test 4: Wall temp breach
    print("TEST 4 -- Wall temp over max (425 > 420)")
    wall = _state(wall_temp_C=425.0)
    viol, reason = check_intra_episode(wall)
    print(f"  violated={viol}, reason='{reason}'")
    assert viol, "Expected wall temp violation"
    assert reason == 'WALL_TEMP_MAX', f"Expected 'WALL_TEMP_MAX', got '{reason}'"
    print("  ASSERT (True, 'WALL_TEMP_MAX'): PASS\n")

    # Test 5: Hot-metal temp breach
    print("TEST 5 -- Hot-metal temp over max (1700 > 1680)")
    hotm = _state(hot_metal_temp_C=1700.0)
    viol, reason = check_intra_episode(hotm)
    print(f"  violated={viol}, reason='{reason}'")
    assert viol, "Expected hot-metal temp violation"
    assert reason == 'HOT_METAL_TEMP_MAX', f"Expected 'HOT_METAL_TEMP_MAX', got '{reason}'"
    print("  ASSERT (True, 'HOT_METAL_TEMP_MAX'): PASS\n")

    # Test 6: CO/CO2 ratio breach
    print("TEST 6 -- CO/CO2 ratio under min (0.8 < 1.0)")
    ratio = _state(co_co2_ratio=0.8)
    viol, reason = check_intra_episode(ratio)
    print(f"  violated={viol}, reason='{reason}'")
    assert viol, "Expected CO/CO2 ratio violation"
    assert reason == 'REDUCING_ATMOS_LOST', f"Expected 'REDUCING_ATMOS_LOST', got '{reason}'"
    print("  ASSERT (True, 'REDUCING_ATMOS_LOST'): PASS\n")

    # Test 7: Quality at termination (good steel)
    print("TEST 7 -- Quality at termination (good steel)")
    good_steel = _state(carbon_pct=4.2, sulfur_pct=0.03)
    off_grade = check_quality_at_termination(good_steel)
    print(f"  off_grade={off_grade}")
    assert not off_grade, "Expected good steel (not off-grade)"
    print("  ASSERT not off_grade: PASS\n")

    # Test 8: Quality at termination (catastrophic failure)
    print("TEST 8 -- Quality at termination (catastrophic off-grade)")
    bad_steel = _state(carbon_pct=3.8, sulfur_pct=0.10)  # carbon_err=0.4 > 0.3, sulfur=0.10 > 0.08
    off_grade = check_quality_at_termination(bad_steel)
    print(f"  off_grade={off_grade}")
    assert off_grade, "Expected off-grade steel"
    print("  ASSERT off_grade: PASS\n")

    print(SEP)
    print("All safety checks PASSED.")
    print(SEP)
