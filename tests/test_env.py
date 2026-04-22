"""
Sanity tests for CrossMill-MegaForge environment.
Tests: shapes, contract, determinism, termination, tapping, safety.
"""

import numpy as np
from app.environment import MegaForgeEnv
from app.models import Observation, StepResponse
from app.config import STATE_DIM, ACTION_DIM, TASK_CONFIG


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #

def test_reset_shapes():
    """Reset returns valid Observation with correct shapes and bounds."""
    env = MegaForgeEnv(task_id='easy', seed=0)
    obs = env.reset(seed=0)
    assert isinstance(obs, Observation), "Reset must return Observation"
    # Check sample normalised fields are in [0,1]
    for f in ['hot_metal_temp', 'carbon', 'silicon', 'thermal_stress',
              'equip_health']:
        v = getattr(obs, f)
        assert 0.0 <= v <= 1.0, f'{f}={v} out of [0,1]'
    assert len(obs.observed_mask) == STATE_DIM, \
        f"observed_mask length {len(obs.observed_mask)} != {STATE_DIM}"
    assert obs.step_idx == 0, f"Reset step_idx should be 0, got {obs.step_idx}"
    print("  PASS: Reset returns valid Observation")


def test_step_contract():
    """Step returns StepResponse with all required fields."""
    env = MegaForgeEnv(task_id='easy', seed=1)
    env.reset(seed=1)
    resp = env.step([0.5] * ACTION_DIM)
    assert isinstance(resp, StepResponse), "Step must return StepResponse"
    assert isinstance(resp.done, bool), "done must be bool"
    assert isinstance(resp.truncated, bool), "truncated must be bool"
    assert -1.5 <= resp.reward.value <= 1.5, \
        f"reward {resp.reward.value} out of [-1.5, 1.5]"
    for key in ('safety_violation', 'regime_id', 'raw_state',
                'tapping_just_occurred'):
        assert key in resp.info, f"Missing key '{key}' in info dict"
    print("  PASS: Step returns valid StepResponse with all keys")


def test_determinism_under_seed():
    """Same seed + same actions produce identical trajectories."""
    def run_episode():
        env = MegaForgeEnv(task_id='easy', seed=42)
        env.reset(seed=42)
        rewards = []
        for _ in range(50):
            resp = env.step([0.50] * ACTION_DIM)
            rewards.append(resp.reward.value)
            if resp.done:
                break
        return rewards

    r1 = run_episode()
    r2 = run_episode()
    assert r1 == r2, \
        f"Same seed + same actions must produce same trajectory.\n" \
        f"  r1 = {r1}\n  r2 = {r2}"
    print("  PASS: Determinism under seed")


def test_episode_terminates():
    """Episode terminates within max_steps."""
    env = MegaForgeEnv(task_id='easy', seed=7)
    env.reset(seed=7)
    max_steps = TASK_CONFIG['easy']['episode_length']
    for s in range(max_steps + 10):
        resp = env.step([0.5] * ACTION_DIM)
        if resp.done:
            assert s < max_steps + 1, \
                f"Episode terminated at step {s}, expected <= {max_steps}"
            print(f"  PASS: Episode terminates at step {s} (max {max_steps})")
            return
    raise AssertionError(
        f'Episode did not terminate within {max_steps} + 1 steps')


def test_tapping_occurs():
    """At least 1 tapping event in a full Easy episode."""
    env = MegaForgeEnv(task_id='easy', seed=55)
    env.reset(seed=55)
    tap_events = 0
    steps_run = 0
    for _ in range(TASK_CONFIG['easy']['episode_length']):
        resp = env.step([0.5] * ACTION_DIM)
        steps_run += 1
        if resp.info.get('tapping_just_occurred'):
            tap_events += 1
        if resp.done:
            break
    assert tap_events >= 1, \
        f'Expected >=1 tap, got {tap_events} over {steps_run} steps'
    print(f"  PASS: {tap_events} tapping event(s) in {steps_run} steps")


def test_safety_trigger():
    """Verify safety violations can occur (random actions at specific seed)."""
    env = MegaForgeEnv(task_id='easy', seed=42)
    env.reset(seed=42)
    triggered = False
    triggered_at = None
    # Use seed 42 which we know triggers THERMAL_STRESS_MAX on step 1
    rng = np.random.default_rng(42)
    for step_num in range(20):
        a = rng.random(10).tolist()  # random action
        resp = env.step(a)
        if resp.info.get('safety_violation'):
            triggered = True
            triggered_at = step_num + 1
            reason = resp.info.get('safety_violation_reason', 'unknown')
            break
    assert triggered, \
        'Expected a safety violation to trigger with random actions'
    print(f"  PASS: Safety violation '{reason}' triggered at step {triggered_at}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    print("Running environment sanity tests...\n")

    test_reset_shapes()
    test_step_contract()
    test_determinism_under_seed()
    test_episode_terminates()
    test_tapping_occurs()
    test_safety_trigger()

    print("\nALL TESTS PASSED.")
