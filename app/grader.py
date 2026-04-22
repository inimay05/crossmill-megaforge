"""
Programmatic grader for CrossMill-MegaForge policies.

Evaluates agents over multiple seeded episodes and returns a 0.0–1.0 score
based on: reward normalisation, safety bonus, catastrophic penalty.
"""

from typing import Callable
import numpy as np

from app.config import (GRADER_EVAL_EPISODES, GRADER_SAFETY_BONUS,
                        GRADER_CATASTROPHIC_PENALTY, CARBON_TARGET_PCT)
from app.environment import MegaForgeEnv
from app.models import EpisodeResult
from app.tasks import grader_target


# --------------------------------------------------------------------------- #
# Core grading functions                                                       #
# --------------------------------------------------------------------------- #

def run_episode(env: MegaForgeEnv, policy: Callable,
                seed: int) -> EpisodeResult:
    """
    Run a single episode under a policy.

    Parameters
    ----------
    env : MegaForgeEnv
        Environment instance.
    policy : Callable
        Function taking an Observation (pydantic) and returning an action
        (dict, Action, or np.ndarray of length 10).
    seed : int
        Seed for env.reset().

    Returns
    -------
    EpisodeResult
        Pydantic model with episode statistics.
    """
    obs = env.reset(seed=seed)
    total_r = 0.0
    steps = 0
    catastrophic = False
    tap_events = 0
    prod_rates = []
    emissions_accum = 0.0

    CATASTROPHIC_REASONS = (
        'PRESSURE_OVER_MAX', 'THERMAL_STRESS_MAX', 'WALL_TEMP_MAX',
        'HOT_METAL_TEMP_MAX', 'REDUCING_ATMOS_LOST', 'QUALITY_FAIL_AT_END',
    )

    while True:
        a = policy(obs)
        resp = env.step(a)
        obs = resp.observation
        total_r += resp.reward.value
        steps += 1

        if resp.info.get('tapping_just_occurred'):
            tap_events += 1

        prod_rates.append(resp.info['raw_state']['production_rate_tph'])
        emissions_accum += resp.info['raw_state']['emissions_CO2_kgpt']

        if resp.info.get('safety_violation'):
            reason = resp.info.get('safety_violation_reason', '')
            if reason in CATASTROPHIC_REASONS:
                catastrophic = True

        if resp.done:
            break

    raw = resp.info['raw_state']
    return EpisodeResult(
        task_id=env.task_id,
        total_reward=float(total_r),
        steps_taken=steps,
        terminated_safely=not resp.info.get('safety_violation', False),
        catastrophic_failure=catastrophic,
        final_carbon_error_pct=float(abs(raw['carbon_pct'] - CARBON_TARGET_PCT)),
        final_coke_rate_kgpt=float(raw['coke_rate_kgpt']),
        total_co2_emissions_kgpt=float(emissions_accum / max(steps, 1)),
        equipment_health_end=float(raw['equip_health_pct']) / 100.0,
        tapping_events=int(tap_events),
        mean_production_rate_tph=(float(np.mean(prod_rates))
                                  if prod_rates else 0.0),
    )


def grader(policy: Callable, task_id: str = 'easy',
           num_eval_episodes: int = GRADER_EVAL_EPISODES,
           base_seed: int = 1000) -> dict:
    """
    Evaluate a policy over multiple episodes and return a grader score.

    Grading formula:
      normalised_reward = clip(mean_reward / (TARGET * 0.4), 0, 1)
        (Terminal-slot weight is 40%, so a fully-correct agent hits
         mean_reward ≈ TARGET * 0.4)
      base = normalised_reward
      bonus = GRADER_SAFETY_BONUS if safety_violation_rate == 0 else 0
      penalty = GRADER_CATASTROPHIC_PENALTY if catastrophic_rate > 0 else 0
      grader_score = clip(base + bonus - penalty, 0, 1)

    Parameters
    ----------
    policy : Callable
        Function (obs) -> action.
    task_id : str
        'easy', 'medium', or 'hard'.
    num_eval_episodes : int
        Number of seeded runs (default from config).
    base_seed : int
        Starting seed for episode runs.

    Returns
    -------
    dict
        Keys: grader_score, mean_reward, std_reward, safety_violation_rate,
              catastrophic_rate, mean_carbon_error_pct, mean_coke_rate_kgpt,
              mean_co2_emissions_kgpt, mean_equipment_health, episodes.
    """
    target = grader_target(task_id)
    env = MegaForgeEnv(task_id=task_id, seed=base_seed)

    results = []
    for i in range(num_eval_episodes):
        results.append(run_episode(env, policy, seed=base_seed + i))

    rewards = np.array([r.total_reward for r in results])
    viols = sum(1 for r in results if not r.terminated_safely)
    cats  = sum(1 for r in results if r.catastrophic_failure)

    mean_r = float(rewards.mean())
    normalised = float(np.clip(mean_r / (target * 0.4), 0.0, 1.0))
    viol_rate = float(viols) / num_eval_episodes
    cat_rate  = float(cats)  / num_eval_episodes

    bonus   = GRADER_SAFETY_BONUS if viols == 0 else 0.0
    penalty = GRADER_CATASTROPHIC_PENALTY if cats > 0 else 0.0
    score = float(np.clip(normalised + bonus - penalty, 0.0, 1.0))

    return {
        'grader_score':           score,
        'mean_reward':            mean_r,
        'std_reward':             float(rewards.std()),
        'safety_violation_rate':  viol_rate,
        'catastrophic_rate':      cat_rate,
        'mean_carbon_error_pct':  float(np.mean(
            [r.final_carbon_error_pct for r in results])),
        'mean_coke_rate_kgpt':    float(np.mean(
            [r.final_coke_rate_kgpt for r in results])),
        'mean_co2_emissions_kgpt': float(np.mean(
            [r.total_co2_emissions_kgpt for r in results])),
        'mean_equipment_health':  float(np.mean(
            [r.equipment_health_end for r in results])),
        'episodes':               results,
    }


# --------------------------------------------------------------------------- #
# Sanity check                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    rng = np.random.default_rng(7)

    def random_policy(obs):
        return rng.random(10).tolist()

    print('Grading random policy on easy task (5 episodes)...\n')
    out = grader(random_policy, task_id='easy', num_eval_episodes=5,
                 base_seed=2000)

    print('Results:')
    print()
    for k, v in out.items():
        if k != 'episodes':
            if isinstance(v, float):
                print(f'  {k:30s} = {v:.4f}')
            else:
                print(f'  {k:30s} = {v}')

    print('\nPer-episode breakdown:')
    print()
    for i, ep in enumerate(out['episodes'], start=1):
        print(f'  Episode {i}:')
        print(f'    reward         = {ep.total_reward:.4f}')
        print(f'    steps          = {ep.steps_taken}')
        print(f'    terminated_safely = {ep.terminated_safely}')
        print(f'    carbon_error   = {ep.final_carbon_error_pct:.4f} %')
        print(f'    equip_health   = {ep.equipment_health_end:.4f}')
        print(f'    taps           = {ep.tapping_events}')
