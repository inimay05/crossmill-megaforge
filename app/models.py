from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from app import config


# ---- OBSERVATION MODEL ----

class Observation(BaseModel):
    """18 normalised state variables + agent-visible metadata."""

    # 18 normalised state variables (each in [0,1])
    hot_metal_temp: float = Field(ge=0.0, le=1.0)
    hearth_temp: float = Field(ge=0.0, le=1.0)
    blast_temp: float = Field(ge=0.0, le=1.0)
    oxygen_flow: float = Field(ge=0.0, le=1.0)
    carbon: float = Field(ge=0.0, le=1.0)
    silicon: float = Field(ge=0.0, le=1.0)
    sulfur: float = Field(ge=0.0, le=1.0)
    top_pressure: float = Field(ge=0.0, le=1.0)
    co_co2_ratio: float = Field(ge=0.0, le=1.0)
    coke_rate: float = Field(ge=0.0, le=1.0)
    ore_coke_ratio: float = Field(ge=0.0, le=1.0)
    energy: float = Field(ge=0.0, le=1.0)
    production_rate: float = Field(ge=0.0, le=1.0)
    wall_temp: float = Field(ge=0.0, le=1.0)
    thermal_stress: float = Field(ge=0.0, le=1.0)
    slag_basicity: float = Field(ge=0.0, le=1.0)
    emissions_co2: float = Field(ge=0.0, le=1.0)
    equip_health: float = Field(ge=0.0, le=1.0)

    # Agent-visible helpers
    step_idx: int = Field(ge=0)
    task_id: Literal['easy', 'medium', 'hard']
    observed_mask: List[bool] = Field(min_length=18, max_length=18)
    assay_ready: bool
    tapping_just_occurred: bool
    valid_actions: List[str]

    model_config = {
        'json_schema_extra': {
            'example': {
                'hot_metal_temp': 0.75,
                'hearth_temp': 0.68,
                'blast_temp': 0.55,
                'oxygen_flow': 0.60,
                'carbon': 0.64,
                'silicon': 0.40,
                'sulfur': 0.30,
                'top_pressure': 0.50,
                'co_co2_ratio': 0.75,
                'coke_rate': 0.55,
                'ore_coke_ratio': 0.48,
                'energy': 0.50,
                'production_rate': 0.65,
                'wall_temp': 0.45,
                'thermal_stress': 0.35,
                'slag_basicity': 0.40,
                'emissions_co2': 0.52,
                'equip_health': 0.88,
                'step_idx': 42,
                'task_id': 'medium',
                'observed_mask': [True] * 18,
                'assay_ready': False,
                'tapping_just_occurred': False,
                'valid_actions': ['NORMAL'],
            }
        }
    }


# ---- ACTION MODEL ----

class Action(BaseModel):
    """Hybrid continuous + discrete actions."""

    # 8 continuous actions (normalised [0,1])
    oxygen_flow_delta: float
    blast_temp: float
    coke_feed_delta: float
    ore_feed_delta: float
    limestone_addition: float
    tapping_interval: float
    temp_ramp_rate: float
    pressure_target: float

    # 2 discrete actions
    alloy_timing: int = Field(ge=0, le=3)
    emergency_cooling: int = Field(ge=0, le=1)

    @model_validator(mode='after')
    def clamp_continuous_actions(self):
        """Clamp continuous actions to [0,1] instead of raising."""
        self.oxygen_flow_delta = max(0.0, min(1.0, self.oxygen_flow_delta))
        self.blast_temp = max(0.0, min(1.0, self.blast_temp))
        self.coke_feed_delta = max(0.0, min(1.0, self.coke_feed_delta))
        self.ore_feed_delta = max(0.0, min(1.0, self.ore_feed_delta))
        self.limestone_addition = max(0.0, min(1.0, self.limestone_addition))
        self.tapping_interval = max(0.0, min(1.0, self.tapping_interval))
        self.temp_ramp_rate = max(0.0, min(1.0, self.temp_ramp_rate))
        self.pressure_target = max(0.0, min(1.0, self.pressure_target))
        return self

    model_config = {
        'json_schema_extra': {
            'example': {
                'oxygen_flow_delta': 0.50,
                'blast_temp': 0.55,
                'coke_feed_delta': 0.48,
                'ore_feed_delta': 0.52,
                'limestone_addition': 0.60,
                'tapping_interval': 0.50,
                'temp_ramp_rate': 0.40,
                'pressure_target': 0.50,
                'alloy_timing': 1,
                'emergency_cooling': 0,
            }
        }
    }


# ---- REWARD BREAKDOWN MODEL ----

class RewardBreakdown(BaseModel):
    """Per-step reward components."""

    dense_progress: float = Field(ge=-0.05, le=0.05)
    terminal_score: float = Field(ge=0.0, le=1.0)
    safety_penalty: float = Field(le=0.0)
    total: float


# ---- REWARD MODEL ----

class Reward(BaseModel):
    """Reward value + breakdown."""

    value: float
    breakdown: RewardBreakdown


# ---- STEP RESPONSE MODEL ----

class StepResponse(BaseModel):
    """What env.step() returns (OpenEnv-compatible)."""

    observation: Observation
    reward: Reward
    done: bool
    truncated: bool
    info: dict = Field(default_factory=dict)

    model_config = {
        'json_schema_extra': {
            'example': {
                'observation': {
                    'hot_metal_temp': 0.75,
                    'hearth_temp': 0.68,
                    'blast_temp': 0.55,
                    'oxygen_flow': 0.60,
                    'carbon': 0.64,
                    'silicon': 0.40,
                    'sulfur': 0.30,
                    'top_pressure': 0.50,
                    'co_co2_ratio': 0.75,
                    'coke_rate': 0.55,
                    'ore_coke_ratio': 0.48,
                    'energy': 0.50,
                    'production_rate': 0.65,
                    'wall_temp': 0.45,
                    'thermal_stress': 0.35,
                    'slag_basicity': 0.40,
                    'emissions_co2': 0.52,
                    'equip_health': 0.88,
                    'step_idx': 42,
                    'task_id': 'medium',
                    'observed_mask': [True] * 18,
                    'assay_ready': False,
                    'tapping_just_occurred': False,
                    'valid_actions': ['NORMAL'],
                },
                'reward': {
                    'value': 0.025,
                    'breakdown': {
                        'dense_progress': 0.025,
                        'terminal_score': 0.0,
                        'safety_penalty': 0.0,
                        'total': 0.025,
                    }
                },
                'done': False,
                'truncated': False,
                'info': {
                    'safety_violation': False,
                    'safety_violation_reason': None,
                    'regime_id': 0,
                    'assay_queue_depth': 0,
                    'tapping_just_occurred': False,
                    'raw_state': None,
                },
            }
        }
    }


# ---- EPISODE RESULT MODEL ----

class EpisodeResult(BaseModel):
    """Grader-facing episode evaluation result."""

    task_id: str
    total_reward: float
    steps_taken: int
    terminated_safely: bool
    catastrophic_failure: bool
    final_carbon_error_pct: float
    final_coke_rate_kgpt: float
    total_co2_emissions_kgpt: float
    equipment_health_end: float = Field(ge=0.0, le=1.0)
    tapping_events: int
    mean_production_rate_tph: float


if __name__ == '__main__':
    print("=== Observation Schema ===")
    print(Observation.model_json_schema())
    print("\n=== Action Schema ===")
    print(Action.model_json_schema())
    print("\n=== StepResponse Schema ===")
    print(StepResponse.model_json_schema())
