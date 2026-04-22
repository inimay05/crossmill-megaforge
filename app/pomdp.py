import numpy as np

from app.config import (
    STATE_DIM, MEAS_NOISE_STD, SLAG_NOISE_STD,
    TAPPING_EVERY_STEPS, STEP_INTERVAL_S,
    HEARTH_HIDDEN_VARS, RANGES, TASK_CONFIG,
)

# ---- Key mappings -------------------------------------------------------- #

# Ordered list of the 18 public state keys (insertion order == RANGES order)
_STATE_KEYS = list(RANGES.keys())
_KEY_IDX    = {k: i for i, k in enumerate(_STATE_KEYS)}

# RANGES key → Observation field name (models.py)
_TO_OBS = {
    'hot_metal_temp_C':    'hot_metal_temp',
    'hearth_temp_C':       'hearth_temp',
    'blast_temp_C':        'blast_temp',
    'oxygen_flow_Nm3pm':   'oxygen_flow',
    'carbon_pct':          'carbon',
    'silicon_pct':         'silicon',
    'sulfur_pct':          'sulfur',
    'top_pressure_bar':    'top_pressure',
    'co_co2_ratio':        'co_co2_ratio',
    'coke_rate_kgpt':      'coke_rate',
    'ore_coke_ratio':      'ore_coke_ratio',
    'energy_GJpt':         'energy',
    'production_rate_tph': 'production_rate',
    'wall_temp_C':         'wall_temp',
    'thermal_stress':      'thermal_stress',
    'slag_basicity':       'slag_basicity',
    'emissions_CO2_kgpt':  'emissions_co2',
    'equip_health_pct':    'equip_health',
}

_ASSAY_KEYS  = ('carbon_pct', 'silicon_pct', 'sulfur_pct')
_HEARTH_KEYS = tuple(HEARTH_HIDDEN_VARS)   # ('slag_basicity', 'sulfur_pct')


def _norm_single(key: str, val: float) -> float:
    lo, hi = RANGES[key]
    return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))


# ---- PomdpLayer ---------------------------------------------------------- #

class PomdpLayer:
    def __init__(self, task_cfg: dict, rng):
        self.task_cfg  = task_cfg
        self.rng       = rng
        self.lab_delay = task_cfg['lab_delay_steps']

        # Assay pipeline: list of (step_enqueued, {key: norm_val})
        self.assay_queue: list = []

        # Last assay values the agent actually received (normalised)
        self.last_assay_values: dict = {k: 0.0 for k in _ASSAY_KEYS}

        # Last hearth values revealed at most recent tapping event (normalised)
        self.last_hearth_values: dict = {k: 0.0 for k in _HEARTH_KEYS}

        self.steps_since_last_tap: int = 0

        # Regime state
        self.regime_id: int = 0
        switches = task_cfg['regime_switches']
        if switches == -1:
            self.regime_steps_remaining = int(rng.integers(100, 251))
        else:
            self.regime_steps_remaining = 999_999
        self._switched_once: bool = False

        # Fixed partial observability mask (True = visible this episode).
        # Hearth-gated and assay-delayed keys are EXCLUDED from the hidden pool —
        # they each have their own explicit observability mechanism, and must not
        # be additionally frozen by the partial mask.
        _excluded = set(_HEARTH_KEYS) | set(_ASSAY_KEYS)
        _maskable_idx = [i for i, k in enumerate(_STATE_KEYS)
                         if k not in _excluded]          # 14 eligible indices

        n_hidden = int(STATE_DIM * (1.0 - task_cfg.get('partial_obs', 1.0)))
        self.fixed_mask = np.ones(STATE_DIM, dtype=bool)
        if n_hidden > 0 and _maskable_idx:
            n_actual = min(n_hidden, len(_maskable_idx))
            hidden_idx = rng.choice(_maskable_idx, size=n_actual, replace=False)
            self.fixed_mask[hidden_idx] = False
        # Result: 4 excluded (always visible) + (14 − n_hidden) maskable visible
        # = STATE_DIM − n_hidden = 11 visible on Hard  ✓

        # Rolling last-known values for partial obs (populated on first call)
        self._last_known: dict = {}

    # ---------------------------------------------------------------------- #

    def maybe_switch_regime(self, step_idx: int, raw_state: dict) -> dict:
        switches = self.task_cfg['regime_switches']
        do_switch = False

        if switches == 0:
            return raw_state
        elif switches == 1:
            if not self._switched_once and step_idx >= 300:
                do_switch = True
                self._switched_once = True
        else:   # == -1: random interval
            self.regime_steps_remaining -= 1
            if self.regime_steps_remaining <= 0:
                do_switch = True
                self.regime_steps_remaining = int(self.rng.integers(100, 251))

        if not do_switch:
            return raw_state

        self.regime_id = int(self.rng.integers(0, 4))

        # Ore grade → carbon_pct multiplier 0.9–1.1
        ore_mult = float(self.rng.uniform(0.9, 1.1))
        lo, hi = RANGES['carbon_pct']
        raw_state['carbon_pct'] = float(np.clip(raw_state['carbon_pct'] * ore_mult, lo, hi))

        # Coke quality → coke_rate_kgpt ± 10 %
        coke_mult = float(self.rng.uniform(0.9, 1.1))
        lo, hi = RANGES['coke_rate_kgpt']
        raw_state['coke_rate_kgpt'] = float(np.clip(raw_state['coke_rate_kgpt'] * coke_mult, lo, hi))

        # Iron-ore moisture → oxygen_flow_Nm3pm ± 5 %
        moist_mult = float(self.rng.uniform(0.95, 1.05))
        lo, hi = RANGES['oxygen_flow_Nm3pm']
        raw_state['oxygen_flow_Nm3pm'] = float(
            np.clip(raw_state['oxygen_flow_Nm3pm'] * moist_mult, lo, hi)
        )

        return raw_state

    # ---------------------------------------------------------------------- #

    def check_tapping_event(self, step_idx: int, raw_action: dict,
                             raw_state: dict) -> bool:
        tap_h = float(raw_action.get('tapping_interval_h', 4.0))
        # Convert hours → steps (STEP_INTERVAL_S=60 → 1 step = 1 minute)
        action_interval_steps = int(tap_h * 60.0 / (STEP_INTERVAL_S / 60.0))
        tap_threshold = min(TAPPING_EVERY_STEPS, action_interval_steps)

        if self.steps_since_last_tap >= tap_threshold:
            for k in _HEARTH_KEYS:
                self.last_hearth_values[k] = _norm_single(k, raw_state[k])
            self.steps_since_last_tap = 0
            return True

        self.steps_since_last_tap += 1
        return False

    # ---------------------------------------------------------------------- #

    def apply_sensor_noise(self, norm_obs: dict) -> dict:
        obs = dict(norm_obs)
        for k in _STATE_KEYS:
            if k not in obs:
                continue
            std = SLAG_NOISE_STD if k == 'slag_basicity' else MEAS_NOISE_STD
            obs[k] = float(np.clip(obs[k] + self.rng.normal(0.0, std), 0.0, 1.0))
        return obs

    # ---------------------------------------------------------------------- #

    def apply_assay_delay(self, norm_obs: dict, step_idx: int,
                          fresh_raw: dict) -> tuple:
        obs = dict(norm_obs)
        assay_ready = False

        if self.lab_delay == 0:
            # Easy: chemistry always current
            for k in _ASSAY_KEYS:
                self.last_assay_values[k] = _norm_single(k, fresh_raw[k])
            assay_ready = True
        else:
            # Enqueue a fresh snapshot every lab_delay steps
            if step_idx % self.lab_delay == 0:
                snapshot = {k: _norm_single(k, fresh_raw[k]) for k in _ASSAY_KEYS}
                self.assay_queue.append((step_idx, snapshot))

            # Pop any matured entries
            while self.assay_queue and \
                    self.assay_queue[0][0] + self.lab_delay <= step_idx:
                _, values = self.assay_queue.pop(0)
                self.last_assay_values = values
                assay_ready = True

        # Always override obs with last known assay values
        for k in _ASSAY_KEYS:
            obs[k] = self.last_assay_values[k]

        return obs, assay_ready

    # ---------------------------------------------------------------------- #

    def apply_hearth_gating(self, norm_obs: dict, tapping_just_occurred: bool,
                             fresh_raw: dict, normalise_fn) -> dict:
        obs = dict(norm_obs)

        if tapping_just_occurred:
            for k in _HEARTH_KEYS:
                self.last_hearth_values[k] = _norm_single(k, fresh_raw[k])

        # Serve last-known hearth values (fresh on tap step, stale otherwise)
        for k in _HEARTH_KEYS:
            obs[k] = self.last_hearth_values[k]

        return obs

    # ---------------------------------------------------------------------- #

    def apply_partial_mask(self, norm_obs: dict) -> tuple:
        obs = dict(norm_obs)

        # Initialise last_known on very first call
        if not self._last_known:
            self._last_known = dict(obs)

        for i, k in enumerate(_STATE_KEYS):
            if k not in obs:
                continue
            if self.fixed_mask[i]:
                self._last_known[k] = obs[k]       # update rolling cache
            else:
                obs[k] = self._last_known.get(k, obs[k])   # serve stale value

        return obs, self.fixed_mask.tolist()

    # ---------------------------------------------------------------------- #

    def forward(self, raw_state: dict, step_idx: int, raw_action: dict,
                normalise_fn) -> tuple:
        # 1. Tapping event (updates last_hearth_values if tap fires)
        tapping = self.check_tapping_event(step_idx, raw_action, raw_state)

        # 2. Normalise ground-truth state
        norm_obs = normalise_fn(raw_state)

        # 3. Assay delay for C / Si / S
        norm_obs, assay_ready = self.apply_assay_delay(norm_obs, step_idx, raw_state)

        # 4. Hearth gating for slag_basicity / sulfur_pct
        norm_obs = self.apply_hearth_gating(norm_obs, tapping, raw_state, normalise_fn)

        # 5. Sensor noise (applied after gating so stale values also get noise)
        norm_obs = self.apply_sensor_noise(norm_obs)

        # 6. Partial observability mask
        norm_obs, observed_mask = self.apply_partial_mask(norm_obs)

        # Translate RANGES keys → Observation field names
        obs_dict = {_TO_OBS[k]: norm_obs[k] for k in _STATE_KEYS}
        obs_dict['observed_mask'] = observed_mask
        obs_dict['assay_ready']   = assay_ready

        return obs_dict, tapping


# ---- Sanity check -------------------------------------------------------- #

if __name__ == '__main__':
    from app.physics import default_initial_state, normalise

    rng   = np.random.default_rng(7)
    cfg   = TASK_CONFIG['hard']
    layer = PomdpLayer(cfg, rng)

    raw = default_initial_state()
    action = {
        'oxygen_flow_delta_pct':  0.0,
        'blast_temp_C':           1200.0,
        'coke_feed_delta_pct':    0.0,
        'ore_feed_delta_pct':     0.0,
        'limestone_addition_pct': 10.0,
        'tapping_interval_h':     4.0,   # 240 steps → only 120-step default tap fires
        'temp_ramp_rate_Cph':     20.0,
        'pressure_target_bar':    2.5,
        'alloy_timing':           1,
        'emergency_cooling':      0,
    }

    PRINT_STEPS = {0, 30, 60, 119, 120, 121, 149}

    slag_hist   = []
    carbon_hist = []
    mask_sums   = []

    for s in range(150):
        obs, tapping = layer.forward(raw, s, action, normalise)

        slag_hist.append(obs['slag_basicity'])
        carbon_hist.append(obs['carbon'])
        mask_sums.append(sum(obs['observed_mask']))

        if s in PRINT_STEPS:
            print(f"\n{'='*58}")
            print(f"Step {s:3d}  tapping={tapping}  assay_ready={obs['assay_ready']}")
            print(f"  observed_mask sum = {sum(obs['observed_mask'])} / {STATE_DIM}")
            print(f"  slag_basicity     = {obs['slag_basicity']:.4f}")
            print(f"  carbon            = {obs['carbon']:.4f}")
            print(f"  hot_metal_temp    = {obs['hot_metal_temp']:.4f}")
            print(f"  equip_health      = {obs['equip_health']:.4f}")
            print(f"  sulfur            = {obs['sulfur']:.4f}")

    # ---- Invariant checks ------------------------------------------------ #
    print(f"\n{'='*58}")
    print("=== INVARIANT CHECKS ===\n")

    # 1. Fixed mask is consistent every step
    unique_mask_sums = set(mask_sums)
    expected_visible = STATE_DIM - int(STATE_DIM * (1.0 - cfg['partial_obs']))
    print(f"[1] Mask sum unique values : {unique_mask_sums}")
    print(f"    Expected {expected_visible}/18 visible : "
          f"{'PASS' if unique_mask_sums == {expected_visible} else 'FAIL'}")

    # 2. Slag basicity constant pre-tap, jumps at step 120
    pre_tap_range  = max(slag_hist[:120]) - min(slag_hist[:120])
    tap_jump       = abs(slag_hist[120] - float(np.mean(slag_hist[115:120])))
    post_tap_range = max(slag_hist[121:]) - min(slag_hist[121:])
    print(f"\n[2] Slag pre-tap noise range  : {pre_tap_range:.4f}  (noise-only fluctuation)")
    print(f"    Slag jump at step 120     : {tap_jump:.4f}")
    print(f"    Slag post-tap noise range : {post_tap_range:.4f}")
    print(f"    Tap caused detectable jump: {'PASS' if tap_jump > pre_tap_range else 'FAIL'}")

    # 3. Carbon base-value updates only at 25-step boundaries.
    #    Use a high threshold (0.30) to ignore per-step noise fluctuations
    #    (~0.04 std, 7.5 σ required to trip) and catch only the genuine
    #    assay-delivery jump (~0.40).
    carbon_change_steps = [
        i for i in range(1, 150)
        if abs(carbon_hist[i] - carbon_hist[i - 1]) > 0.30
    ]
    print(f"\n[3] Carbon large-jump steps   : {carbon_change_steps}")
    first_change_ok = (len(carbon_change_steps) > 0 and carbon_change_steps[0] == 25)
    print(f"    First jump at step 25     : {'PASS' if first_change_ok else 'FAIL'}")
    on_boundary = all(s % 25 == 0 for s in carbon_change_steps)
    print(f"    All on 25-step boundary   : {'PASS' if on_boundary else 'FAIL'}")

    # 4. Noise differs each call.
    #    Pre-tap slag is N(0, 0.10) clipped at 0, so ~half the values are 0.0;
    #    checking > 2 distinct values confirms noise is being generated.
    distinct_slag = len(set(round(v, 4) for v in slag_hist[:10]))
    print(f"\n[4] Distinct slag values (first 10 steps): {distinct_slag}")
    print(f"    Noise varies each call    : {'PASS' if distinct_slag > 2 else 'FAIL'}")
