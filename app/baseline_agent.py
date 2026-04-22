"""
Domain-informed PID-style baseline controller for CrossMill-MegaForge.

Strategy:
  - Hold blast temp at nominal 1100 °C (normalised 0.50)
  - Keep oxygen flow at nominal, zero delta (0.50)
  - Small adaptive tweaks based on carbon, equipment health
  - Emergency cooling only if thermal_stress exceeds 0.75
  - Respect maintenance window (forced safe action)
"""

from app.models import Observation


class PIDBaselineAgent:
    """
    Domain-informed baseline: safe, stable operational setpoints with minimal
    adaptive feedback.
    """

    def __init__(self):
        pass

    def __call__(self, obs: Observation) -> list:
        """
        obs : Observation (pydantic model with normalised fields in [0,1])

        Returns
        -------
        list of length 10
            Normalised action [oxygen_flow_delta, blast_temp, coke_feed_delta,
                              ore_feed_delta, limestone_addition, tapping_interval,
                              temp_ramp_rate, pressure_target, alloy_timing,
                              emergency_cooling]
        """
        # Maintenance window override: safe do-nothing
        if 'MAINTENANCE_ONLY' in obs.valid_actions:
            return [0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0, 0]

        # Nominal setpoints (all centre / baseline values)
        oxygen  = 0.50   # 0 % delta
        blast_t = 0.50   # 1100 °C
        coke_d  = 0.50   # 0 % delta
        ore_d   = 0.50   # 0 % delta
        lime    = 0.50   # 10 %
        tap     = 0.50   # 4 h
        ramp    = 0.33   # 20 °C/h
        pres    = 0.50   # 2.5 bar
        alloy   = 1      # early timing
        estop   = 0      # normal operation

        # Adaptive tweaks

        # Carbon control: bump coke if low, reduce if high
        if obs.carbon < 0.45:
            coke_d = min(1.0, coke_d + 0.10)
        elif obs.carbon > 0.65:
            coke_d = max(0.0, coke_d - 0.10)

        # Equipment protection: cool refractory if health failing
        if obs.equip_health < 0.60:
            blast_t = max(0.0, blast_t - 0.10)

        # Thermal safety: emergency cooling if stress high
        if obs.thermal_stress > 0.75:
            estop = 1

        return [oxygen, blast_t, coke_d, ore_d, lime, tap, ramp, pres,
                alloy, estop]


# --------------------------------------------------------------------------- #
# Sanity check                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import sys
    from app.grader import grader

    task_id = 'easy'
    if len(sys.argv) > 1:
        if sys.argv[1] == '--task' and len(sys.argv) > 2:
            task_id = sys.argv[2]

    agent = PIDBaselineAgent()
    print(f'Evaluating PID baseline on {task_id} task (10 episodes)...\n')
    out = grader(agent, task_id=task_id, num_eval_episodes=10, base_seed=3000)

    print(f'PID BASELINE on {task_id} (10 eps):\n')
    for k, v in out.items():
        if k != 'episodes':
            if isinstance(v, float):
                print(f'  {k:30s} = {v:.4f}')
            else:
                print(f'  {k:30s} = {v}')

    print('\nSummary:')
    print(f'  Grader score         = {out["grader_score"]:.4f} / 1.0')
    print(f'  Safety rate          = {1.0 - out["safety_violation_rate"]:.1%}')
    print(f'  Safety violation rate= {out["safety_violation_rate"]:.4f}')
    print(f'  Catastrophic         = {out["catastrophic_rate"]:.1%}')
    print(f'  Mean reward          = {out["mean_reward"]:.4f}')
