# ---- STATE DIMENSIONS ----
STATE_DIM = 18     # 18 physical variables (see spec)
ACTION_DIM = 10    # 10 continuous control actions

# ---- RAW VARIABLE RANGES (for normalisation; agent sees [0,1] normalised) ----
RANGES = {
    'hot_metal_temp_C':      (1450.0, 1600.0),
    'hearth_temp_C':         (1500.0, 1650.0),
    'blast_temp_C':          (900.0, 1300.0),
    'oxygen_flow_Nm3pm':     (50.0, 200.0),
    'carbon_pct':            (3.8, 4.8),
    'silicon_pct':           (0.3, 2.0),
    'sulfur_pct':            (0.01, 0.10),
    'top_pressure_bar':      (1.5, 3.5),
    'co_co2_ratio':          (1.5, 3.0),
    'coke_rate_kgpt':        (300.0, 600.0),
    'ore_coke_ratio':        (2.5, 4.0),
    'energy_GJpt':           (18.0, 25.0),
    'production_rate_tph':   (100.0, 400.0),
    'wall_temp_C':           (200.0, 400.0),
    'thermal_stress':        (0.0, 100.0),
    'slag_basicity':         (0.8, 1.3),           # CaO/SiO2 ratio proxy
    'emissions_CO2_kgpt':    (1400.0, 2200.0),
    'equip_health_pct':      (50.0, 100.0),
}

# ---- ACTION RANGES ----
# Agent outputs are normalised [0,1]; env denormalises to these real units.
ACTION_RANGES = {
    'oxygen_flow_delta_pct': (-30.0, 30.0),
    'blast_temp_C':          (900.0, 1300.0),
    'coke_feed_delta_pct':   (-20.0, 20.0),
    'ore_feed_delta_pct':    (-15.0, 15.0),
    'limestone_addition_pct':(5.0, 15.0),
    'tapping_interval_h':    (2.0, 6.0),
    'temp_ramp_rate_Cph':    (5.0, 50.0),
    'pressure_target_bar':   (1.5, 3.5),
    'alloy_timing':          (0, 3),                # discrete: 0=none,1=early,2=mid,3=late
    'emergency_cooling':     (0, 1),                # binary
}

# ---- REDUCTION CHEMISTRY (Fe2O3 + 3CO -> 2Fe + 3CO2) ----
# Per-zone empirical Arrhenius-ish rate:
#   r(T) = k0 * exp(-Ea / (R*T)) * (CO / CO2)^n
R_GAS = 8.314  # J/(mol·K)
REDUCTION_TOP    = {'k0': 5.0e-2, 'Ea': 8.0e4, 'n_co': 0.6}   # indirect, drying
REDUCTION_MIDDLE = {'k0': 2.0e0,  'Ea': 1.1e5, 'n_co': 0.8}   # main reduction
REDUCTION_BOTTOM = {'k0': 8.0e0,  'Ea': 1.6e5, 'n_co': 1.0}   # melting / final
ZONES = ('top', 'middle', 'bottom')

# ---- PHYSICS ENGINE ----
FURNACE_ZONES   = 3            # top -> middle -> bottom finite volumes
ODE_METHOD      = 'BDF'        # stiff solver
ODE_RTOL        = 1e-4
ODE_ATOL        = 1e-7
STEP_INTERVAL_S = 60.0         # 1 minute of real furnace time per env step
# Zone heat-transfer coefficients (W/(m²·K)):
U_TOP           = 400.0
U_MIDDLE        = 900.0
U_BOTTOM        = 1500.0
CP_GAS          = 1100.0       # J/(kg·K), hot flue gas
CP_IRON         = 820.0
# Refractory wear rate (hidden, slow):
REFRACTORY_WEAR_RATE = 0.0005  # pct-points per step at nominal conditions
WEAR_TEMP_COEF       = 2.0     # wear scales with (wall_temp - 300)/100

# ---- REWARD HYBRID ----
PROGRESS_WEIGHT  = 0.60
TERMINAL_WEIGHT  = 0.40
DENSE_CLAMP_LOW  = -0.05
DENSE_CLAMP_HIGH =  0.05
SAFETY_PENALTY   = -1.0

# Terminal component weights (must sum to 1.0):
W_QUALITY        = 0.30     # carbon accuracy, silicon, sulfur control, strength proxy
W_ENERGY         = 0.25     # coke rate, heat utilisation, kWh/tonne
W_PRODUCTION     = 0.20     # throughput + uptime + tapping schedule bonus
W_SAFETY_EQUIP   = 0.15     # temp margins, pressure stability, thermal-stress ratio
W_ENVIRONMENTAL  = 0.10     # CO2 / NOx / SOx reduction vs baseline

# Quality targets (for scoring):
CARBON_TARGET_PCT     = 4.2
CARBON_TOLERANCE_PCT  = 0.05
SILICON_TARGET_PCT    = 0.8
SULFUR_MAX_PCT        = 0.05

# ---- HARD SAFETY LIMITS ----
PRESSURE_MAX_BAR           = 3.8
THERMAL_STRESS_MAX         = 90.0
WALL_TEMP_MAX_C            = 420.0    # catastrophic refractory burn-through
HOT_METAL_TEMP_MAX_C       = 1680.0   # hearth damage / slip risk
CO_CO2_RATIO_MIN           = 1.0      # reducing atmosphere must be preserved

# ---- POMDP ----
MEAS_NOISE_STD       = 0.04       # ±4% Gaussian on most sensor readings
SLAG_NOISE_STD       = 0.10       # slag-chemistry sensors are notoriously noisy
LAB_DELAY_EASY       = 0
LAB_DELAY_MEDIUM     = 15         # 15 steps (~15 min) for chemistry assay
LAB_DELAY_HARD       = 25
PARTIAL_OBS_FRAC_HARD= 0.60       # only 60% of state visible in Hard

# Hearth/slag variables are NEVER directly visible - only observable at tapping:
HEARTH_HIDDEN_VARS = ('slag_basicity', 'sulfur_pct')
TAPPING_EVERY_STEPS = 120         # hearth variables refresh on tapping events

# ---- TASK CONFIGS (Easy / Medium / Hard) ----
TASK_CONFIG = {
    'easy': {
        'episode_length':  200,
        'stochastic':      False,
        'lab_delay_steps': LAB_DELAY_EASY,
        'regime_switches': 0,
        'grader_target':   0.88,
        'partial_obs':     1.0,
        'require_maintenance_window': False,
    },
    'medium': {
        'episode_length':  600,
        'stochastic':      True,
        'lab_delay_steps': LAB_DELAY_MEDIUM,
        'regime_switches': 1,      # 1 sudden low-grade ore delivery
        'grader_target':   0.80,
        'partial_obs':     1.0,
        'require_maintenance_window': False,
    },
    'hard': {
        'episode_length':  2000,   # ~33 hours of furnace operation
        'stochastic':      True,
        'lab_delay_steps': LAB_DELAY_HARD,
        'regime_switches': -1,     # random every 100-250 steps
        'grader_target':   0.74,
        'partial_obs':     PARTIAL_OBS_FRAC_HARD,
        'require_maintenance_window': True,
    },
}

# ---- REWARD SCORING CONSTANTS ----
CARBON_MAX_ERR           = 0.3
SILICON_MAX_ERR          = 0.8
HOT_METAL_TEMP_TARGET    = 1500.0
HOT_METAL_TEMP_TOLERANCE = 150.0
PRODUCTION_TARGET_TPH    = 300.0
PRESSURE_KNEE_BAR        = 3.0
PRESSURE_SCALE           = 0.8
THERMAL_STRESS_SAFE_ZONE = 60.0
THERMAL_STRESS_WARN      = 75.0
EMISSIONS_SPIKE_THRESHOLD = 2000.0
WALL_TEMP_WARN_C         = 380.0
CO_CO2_OPTIMAL_LOW       = 1.8
CO_CO2_OPTIMAL_HIGH      = 2.6

# ---- GRADER ----
GRADER_EVAL_EPISODES = 50
GRADER_SAFETY_BONUS  = 0.20
GRADER_CATASTROPHIC_PENALTY = 0.30
