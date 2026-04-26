"""
CrossMill-MegaForge environment — blast furnace steel production POMDP.
OpenEnv-compatible: exposes observation_spec, action_spec, reset, step,
plus OpenEnv metadata.
"""

import copy

import numpy as np

try:
    from openenv.core import Environment as Env
except ImportError:
    Env = object  # type: ignore[assignment, misc]

from app.config import (TASK_CONFIG, STEP_INTERVAL_S, STATE_DIM, ACTION_DIM,
                        RANGES, ACTION_RANGES)
from app.models import (Observation, Action, StepResponse, Reward,
                        RewardBreakdown)
from app.physics import (physics_step, normalise, denormalise,
                         default_initial_state)
from app.pomdp import PomdpLayer
from app.reward import compute_reward
from app.safety import check_intra_episode, check_quality_at_termination


# --------------------------------------------------------------------------- #
# Environment                                                                   #
# --------------------------------------------------------------------------- #

class MegaForgeEnv(Env):
    """
    CrossMill MegaForge — blast furnace steel production POMDP.
    OpenEnv-compatible: exposes observation_spec, action_spec, reset, step,
    plus OpenEnv metadata.
    """

    name    = 'crossmill-megaforge'
    version = '1.0.0'

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def __init__(self, task_id: str = 'easy', seed=None):
        super().__init__()
        assert task_id in TASK_CONFIG, f"Unknown task '{task_id}'"
        self.task_id   = task_id
        self.task_cfg  = TASK_CONFIG[task_id]
        self.max_steps = self.task_cfg['episode_length']
        self._seed     = seed

        self.observation_spec = Observation.model_json_schema()
        self.action_spec      = Action.model_json_schema()

        # Maintenance window (Hard only): steps [1200, 1300) are forced-
        # maintenance; most actions are ignored.
        self.maintenance_window = (
            (1200, 1300) if self.task_cfg.get('require_maintenance_window')
            else None
        )

        self.reset(seed=seed)

    # ------------------------------------------------------------------ #

    def reset(self, seed=None, episode_id=None, **kwargs) -> Observation:
        self.rng = np.random.default_rng(
            seed if seed is not None else self._seed)

        self.raw_state = default_initial_state()

        # Stochastic initial jitter (Medium / Hard only)
        if self.task_cfg.get('stochastic', False):
            for key, std in [('carbon_pct', 0.05), ('silicon_pct', 0.05)]:
                lo, hi = RANGES[key]
                self.raw_state[key] = float(np.clip(
                    self.raw_state[key] + self.rng.normal(0.0, std), lo, hi))
            for key, frac in [('coke_rate_kgpt', 0.02),
                               ('oxygen_flow_Nm3pm', 0.02)]:
                lo, hi = RANGES[key]
                self.raw_state[key] = float(np.clip(
                    self.raw_state[key] * (1.0 + self.rng.normal(0.0, frac)),
                    lo, hi))

        self.pomdp      = PomdpLayer(self.task_cfg, self.rng)
        self.trajectory = []
        self.step_idx   = 0

        # Last denormalised raw-action dict (needed by POMDP for tapping calc)
        self._last_raw_action: dict = {}
        # Tapping flag cached by _build_observation → read by step()
        self._last_tapping: bool = False

        return self._build_observation()

    # ------------------------------------------------------------------ #
    # Core step                                                            #
    # ------------------------------------------------------------------ #

    def step(self, action) -> StepResponse:
        """
        action: dict | Action | np.ndarray | list  (length ACTION_DIM).
        """
        act = self._coerce_action(action)

        # 1. Maintenance window override (Hard only).
        #    Uses pre-increment step_idx so window [1200, 1300) maps to
        #    env steps 1200 … 1299.
        in_maint = (
            self.maintenance_window is not None
            and self.maintenance_window[0] <= self.step_idx
            < self.maintenance_window[1]
        )
        if in_maint:
            # Safe-maintenance action: halt feeds, minimum blast, no cooling
            act = Action(
                oxygen_flow_delta=0.5,   # centre → 0 % delta (no change)
                blast_temp=0.0,          # minimum → 900 °C
                coke_feed_delta=0.0,     # minimum → −20 % (halt coke)
                ore_feed_delta=0.0,      # minimum → −15 % (halt ore)
                limestone_addition=0.5,  # centre
                tapping_interval=0.5,    # centre → 4 h
                temp_ramp_rate=0.0,      # minimum → 5 °C/h
                pressure_target=0.0,     # minimum → 1.5 bar
                alloy_timing=0,
                emergency_cooling=0,
            )

        # 2. Regime update — MUST happen before POMDP forward()
        self.raw_state = self.pomdp.maybe_switch_regime(
            self.step_idx, self.raw_state)

        # 3. Denormalise to real engineering units
        raw_action = self._denormalise_action(act)
        self._last_raw_action = raw_action

        # 4. Snapshot previous state for dense-reward delta
        prev_raw = copy.deepcopy(self.raw_state)

        # 5. Physics
        self.raw_state = physics_step(
            self.raw_state, raw_action, STEP_INTERVAL_S, self.rng)

        # 6. Advance counter
        self.step_idx += 1

        # 7. POMDP observation pass (forward() runs check_tapping_event
        #    internally; result is cached in self._last_tapping).
        obs = self._build_observation()
        tapping_just_occurred = self._last_tapping

        # Stamp tapping flag on raw_state before appending to trajectory
        # (used by compute_terminal_score → tapping_bonus).
        self.raw_state['_tapping_this_step'] = tapping_just_occurred
        self.trajectory.append(dict(self.raw_state))

        # 8. Safety
        intra_violation, reason = check_intra_episode(self.raw_state)
        is_terminal_step = self.step_idx >= self.max_steps
        quality_fail = (check_quality_at_termination(self.raw_state)
                        if is_terminal_step else False)
        safety_violation = intra_violation
        done      = safety_violation or is_terminal_step
        truncated = is_terminal_step and not safety_violation

        # 9. Reward
        reward = compute_reward(
            prev_raw, self.raw_state, raw_action,
            done, self.trajectory, safety_violation,
        )

        # 10. Info
        info = {
            'safety_violation':        safety_violation,
            'safety_violation_reason': (reason if intra_violation
                                        else ('QUALITY_FAIL_AT_END'
                                              if quality_fail else '')),
            'quality_fail':            quality_fail,
            'regime_id':               self.pomdp.regime_id,
            'assay_queue_depth':       len(self.pomdp.assay_queue),
            'tapping_just_occurred':   tapping_just_occurred,
            'in_maintenance_window':   in_maint,
            'raw_state':               dict(self.raw_state),
        }

        info['reward_breakdown'] = reward.model_dump()
        return StepResponse(
            observation=obs,
            reward=reward.value,
            done=done,
            truncated=truncated,
            info=info,
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _build_observation(self,
                           tapping_just_occurred: bool = False) -> Observation:
        """
        Run the POMDP forward pass and pack the result into an Observation.

        forward() calls check_tapping_event internally — we cache its result
        in self._last_tapping so step() can stamp _tapping_this_step onto the
        trajectory entry without a second call.
        """
        obs_dict, tap = self.pomdp.forward(
            self.raw_state,
            self.step_idx,
            self._last_raw_action,
            normalise,
        )
        # Cache for step() to read
        self._last_tapping = tap

        assay_ready   = obs_dict.pop('assay_ready')
        observed_mask = obs_dict.pop('observed_mask')

        # Valid actions hint
        in_maint = (
            self.maintenance_window is not None
            and self.maintenance_window[0] <= self.step_idx
            < self.maintenance_window[1]
        )
        if in_maint:
            valid_actions = ['MAINTENANCE_ONLY']
        elif self.raw_state.get('thermal_stress', 0.0) > 75.0:
            valid_actions = ['NORMAL', 'EMERGENCY_COOLING']
        else:
            valid_actions = ['NORMAL']

        return Observation(
            **obs_dict,
            step_idx=self.step_idx,
            task_id=self.task_id,
            observed_mask=observed_mask,
            assay_ready=assay_ready,
            tapping_just_occurred=tap or tapping_just_occurred,
            valid_actions=valid_actions,
        )

    # ------------------------------------------------------------------ #

    def _coerce_action(self, action) -> Action:
        """
        Accept: Action | dict | np.ndarray (len 10) | list (len 10).

        Array field order:
          [0] oxygen_flow_delta  [1] blast_temp         [2] coke_feed_delta
          [3] ore_feed_delta     [4] limestone_addition  [5] tapping_interval
          [6] temp_ramp_rate     [7] pressure_target
          [8] alloy_timing       [9] emergency_cooling
        """
        if isinstance(action, Action):
            return action
        if isinstance(action, dict):
            return Action(**action)

        a = [float(v) for v in action]
        if len(a) != ACTION_DIM:
            raise ValueError(
                f"Expected action of length {ACTION_DIM}, got {len(a)}")

        return Action(
            oxygen_flow_delta=a[0],
            blast_temp=a[1],
            coke_feed_delta=a[2],
            ore_feed_delta=a[3],
            limestone_addition=a[4],
            tapping_interval=a[5],
            temp_ramp_rate=a[6],
            pressure_target=a[7],
            alloy_timing=int(round(a[8] * 3)),
            emergency_cooling=int(round(a[9])),
        )

    # ------------------------------------------------------------------ #

    def _denormalise_action(self, act: Action) -> dict:
        """Map normalised Action fields → raw engineering units via ACTION_RANGES."""
        def _dn(v, key):
            lo, hi = ACTION_RANGES[key]
            return float(v) * (hi - lo) + lo

        return {
            'oxygen_flow_delta_pct':  _dn(act.oxygen_flow_delta,  'oxygen_flow_delta_pct'),
            'blast_temp_C':           _dn(act.blast_temp,          'blast_temp_C'),
            'coke_feed_delta_pct':    _dn(act.coke_feed_delta,     'coke_feed_delta_pct'),
            'ore_feed_delta_pct':     _dn(act.ore_feed_delta,      'ore_feed_delta_pct'),
            'limestone_addition_pct': _dn(act.limestone_addition,  'limestone_addition_pct'),
            'tapping_interval_h':     _dn(act.tapping_interval,    'tapping_interval_h'),
            'temp_ramp_rate_Cph':     _dn(act.temp_ramp_rate,      'temp_ramp_rate_Cph'),
            'pressure_target_bar':    _dn(act.pressure_target,     'pressure_target_bar'),
            'alloy_timing':           act.alloy_timing,
            'emergency_cooling':      act.emergency_cooling,
        }

    # ------------------------------------------------------------------ #

    @property
    def state(self):
        """OpenEnv Environment.state abstract property — returns lightweight state."""
        try:
            from openenv.core.env_server.types import State
            return State(step_count=self.step_idx)
        except Exception:
            return {'step_count': self.step_idx}

    def get_metadata(self):
        try:
            from openenv.core.env_server.types import EnvironmentMetadata
            return EnvironmentMetadata(
                name=self.name,
                description='CrossMill MegaForge — blast furnace steel production POMDP.',
                version=self.version,
            )
        except Exception:
            return {'name': self.name, 'version': self.version}

    def close(self):
        pass


# --------------------------------------------------------------------------- #
# Sanity check                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import numpy as np
    from app.config import ACTION_DIM

    env = MegaForgeEnv(task_id='easy', seed=42)
    obs = env.reset(seed=42)
    print('Reset OK.')
    print(f'  obs.hot_metal_temp = {obs.hot_metal_temp:.4f}')
    print(f'  obs.step_idx       = {obs.step_idx}')
    print(f'  obs.task_id        = {obs.task_id}')
    print(f'  obs.valid_actions  = {obs.valid_actions}')
    print(f'  observed_mask sum  = {sum(obs.observed_mask)} / {STATE_DIM}')

    rng = np.random.default_rng(0)
    total_r  = 0.0
    steps    = 0
    tap_events = 0

    while True:
        a = rng.random(ACTION_DIM).tolist()
        resp = env.step(a)
        total_r += resp.reward
        steps   += 1
        if resp.info.get('tapping_just_occurred'):
            tap_events += 1
        if resp.done:
            break

    print(f'\nEpisode finished:')
    print(f'  steps              = {steps}')
    print(f'  total_reward       = {total_r:.4f}')
    print(f'  taps               = {tap_events}')
    print(f'  safety_violation   = {resp.info["safety_violation"]}')
    print(f'  violation_reason   = {resp.info["safety_violation_reason"]!r}')
    print(f'  truncated          = {resp.truncated}')
    print(f'  quality_fail       = {resp.info["quality_fail"]}')
    print(f'  final carbon       = {resp.info["raw_state"]["carbon_pct"]:.4f}')
    print(f'  final equip_hlth   = {resp.info["raw_state"]["equip_health_pct"]:.4f}')
    print(f'  final thermal_str  = {resp.info["raw_state"]["thermal_stress"]:.4f}')
    print(f'  final hot_metal_T  = {resp.info["raw_state"]["hot_metal_temp_C"]:.2f}')
    print(f'  final co_co2       = {resp.info["raw_state"]["co_co2_ratio"]:.4f}')
    _bd = resp.info.get('reward_breakdown', {})
    print(f'  reward breakdown:')
    print(f'    dense_progress   = {_bd.get("dense_progress", 0.0):.4f}')
    print(f'    terminal_score   = {_bd.get("terminal_score", 0.0):.4f}')
    print(f'    safety_penalty   = {_bd.get("safety_penalty", 0.0):.4f}')
    print(f'    total            = {resp.reward:.4f}')

    # ---- Run a second episode with a conservative no-op action ----------- #
    print('\n--- Conservative no-op agent (200 steps) ---')
    env2 = MegaForgeEnv(task_id='easy', seed=7)
    env2.reset(seed=7)
    noop = [0.5] * ACTION_DIM   # all-centre normalised action
    total_r2, steps2, taps2 = 0.0, 0, 0
    resp2 = None
    while True:
        resp2 = env2.step(noop)
        total_r2 += resp2.reward
        steps2   += 1
        if resp2.info.get('tapping_just_occurred'):
            taps2 += 1
        if resp2.done:
            break
    print(f'  steps              = {steps2}')
    print(f'  total_reward       = {total_r2:.4f}')
    print(f'  taps               = {taps2}')
    print(f'  safety_violation   = {resp2.info["safety_violation"]}')
    print(f'  violation_reason   = {resp2.info["safety_violation_reason"]!r}')
    print(f'  truncated          = {resp2.truncated}')
    print(f'  final carbon       = {resp2.info["raw_state"]["carbon_pct"]:.4f}')
    assert steps2 == 200, f'Expected 200 steps, got {steps2}'
    assert resp2.truncated, 'Expected truncated=True on clean termination'
    print('  ASSERT 200 steps + truncated: PASS')
