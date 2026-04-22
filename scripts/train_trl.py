"""
CrossMill-MegaForge training pipeline.

Stack split (honest):
  - sb3-contrib    : owns the RL training loop (RecurrentPPO / MlpLstmPolicy)
                     because the env is a continuous-action POMDP and only
                     SB3-contrib has RecurrentPPO for this shape.
  - TensorBoard    : real-time metrics + monitor.csv (via SB3 VecMonitor).
  - matplotlib     : renders reward curve PNG from monitor.csv.
  - huggingface_hub: pushes model.zip + reward_curve.png + README to the
                     HF Hub. This is the HuggingFace ecosystem library and
                     is the correct tool for non-LLM RL artifact publishing.
                     (TRL is designed for LLM RLHF, not continuous POMDP RL,
                     so we do not fake a TRL wrapper.)

Target: Google Colab T4 (or any Linux with PyTorch).
Runtime: ~30 min for Easy at 100k timesteps.
Usage:   python scripts/train_trl.py --task easy --timesteps 100000
              --push_to_hub --hf_repo_id user/crossmill-megaforge-easy
"""

import argparse
import os
import sys
import time
import random
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # headless-safe; MUST be BEFORE pyplot import
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from sb3_contrib import RecurrentPPO

from app.environment import MegaForgeEnv
from app.config import STATE_DIM, ACTION_DIM
from app.baseline_agent import PIDBaselineAgent
from app.grader import grader


# --------------------------------------------------------------------------- #
# Utilities                                                                    #
# --------------------------------------------------------------------------- #

def seed_everything(seed: int):
    """Seed python, numpy, and torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Gymnasium shim                                                               #
# --------------------------------------------------------------------------- #

class MegaForgeGymShim(gym.Env):
    """Thin gym.Env shim over MegaForgeEnv so SB3 RecurrentPPO can consume it."""

    metadata = {}

    def __init__(self, task_id='easy', seed=None):
        super().__init__()
        self.inner = MegaForgeEnv(task_id=task_id, seed=seed)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(STATE_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )

    def _obs_to_vec(self, obs):
        """Convert Observation pydantic model to numpy array."""
        return np.array([
            obs.hot_metal_temp, obs.hearth_temp, obs.blast_temp,
            obs.oxygen_flow, obs.carbon, obs.silicon, obs.sulfur,
            obs.top_pressure, obs.co_co2_ratio, obs.coke_rate,
            obs.ore_coke_ratio, obs.energy, obs.production_rate,
            obs.wall_temp, obs.thermal_stress, obs.slag_basicity,
            obs.emissions_co2, obs.equip_health,
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs = self.inner.reset(seed=seed)
        return self._obs_to_vec(obs), {}

    def step(self, action):
        resp = self.inner.step(action.tolist())
        return (
            self._obs_to_vec(resp.observation),
            float(resp.reward.value),
            bool(resp.done and not resp.truncated),
            bool(resp.truncated),
            dict(resp.info),
        )


def make_env(task_id, rank, seed=0):
    """Factory for vectorised env."""
    def _init():
        return MegaForgeGymShim(task_id=task_id, seed=seed + rank)
    return _init


# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #

def plot_reward_curve(monitor_csv_path: str, out_png_path: str,
                      task_id: str, baseline_score: float,
                      final_score: float, window: int = 50):
    """
    Read the VecMonitor CSV and plot a rolling-mean episode reward curve.
    VecMonitor CSV has one '#' comment header line, then columns r, l, t.
    """
    df = pd.read_csv(monitor_csv_path, skiprows=1)
    df['rolling'] = df['r'].rolling(window=window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df.index, df['r'], alpha=0.25, linewidth=0.8,
            label='episode reward')
    ax.plot(df.index, df['rolling'], linewidth=2.0,
            label=f'rolling mean (window={window})')
    ax.axhline(baseline_score, linestyle='--', color='gray',
               label=f'PID baseline ({baseline_score:.3f})')
    ax.axhline(final_score, linestyle='--', color='green',
               label=f'final grader score ({final_score:.3f})')

    ax.set_title(f'CrossMill MegaForge — {task_id} training reward curve')
    ax.set_xlabel('episode')
    ax.set_ylabel('total reward')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# HuggingFace Hub push                                                         #
# --------------------------------------------------------------------------- #

def push_artifacts_to_hub(repo_id: str, model_zip_path: str,
                          curve_png_path: str, summary: dict):
    """
    Push model, reward curve PNG, and auto-generated model card to the HF Hub.
    Requires HF_TOKEN (via huggingface-cli login or notebook_login).
    """
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    create_repo(repo_id, repo_type='model', exist_ok=True)

    card = (
        f"# {repo_id}\n\n"
        f"RecurrentPPO (LSTM) policy trained on CrossMill-MegaForge "
        f"({summary['task_id']} task).\n\n"
        f"## Results\n\n"
        f"- Pre-training (PID baseline) grader score: "
        f"{summary['pre_score']:.3f}\n"
        f"- Post-training grader score: {summary['post_score']:.3f}\n"
        f"- Delta: {summary['delta']:+.3f}\n"
        f"- Safety violation rate: {summary['safety_violation_rate']:.3f}\n"
        f"- Catastrophic failure rate: {summary['catastrophic_rate']:.3f}\n"
        f"- Mean carbon error (% abs): "
        f"{summary['mean_carbon_error_pct']:.4f}\n"
        f"- Mean coke rate (kg/tonne): {summary['mean_coke_rate_kgpt']:.1f}\n"
        f"- Mean CO2 emissions (kg/tonne): "
        f"{summary['mean_co2_emissions_kgpt']:.1f}\n\n"
        f"## Files\n\n"
        f"- model.zip — trained RecurrentPPO checkpoint (SB3 format)\n"
        f"- reward_curve.png — training reward curve\n\n"
        f"## Reproduce\n\n"
        f"See the CrossMill repo for the full env + scripts/train_trl.py.\n"
    )

    card_path = os.path.join(os.path.dirname(model_zip_path), 'README.md')
    with open(card_path, 'w') as f:
        f.write(card)

    for path in (model_zip_path, curve_png_path, card_path):
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=os.path.basename(path),
            repo_id=repo_id,
            repo_type='model',
        )

    print(f'Pushed {len([model_zip_path, curve_png_path, card_path])} '
          f'artifacts to https://huggingface.co/{repo_id}')


# --------------------------------------------------------------------------- #
# Policy adapter for grading                                                   #
# --------------------------------------------------------------------------- #

class PolicyAdapter:
    """Wraps RecurrentPPO for use with the grader."""

    def __init__(self, model, shim):
        self.model = model
        self.shim = shim
        self.lstm_state = None
        self.episode_starts = np.ones((1,), dtype=bool)

    def __call__(self, obs):
        vec = self.shim._obs_to_vec(obs).reshape(1, -1)
        action, self.lstm_state = self.model.predict(
            vec,
            state=self.lstm_state,
            episode_start=self.episode_starts,
            deterministic=True,
        )
        self.episode_starts = np.zeros((1,), dtype=bool)
        return action[0].tolist()


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description='Train RecurrentPPO on CrossMill-MegaForge')
    ap.add_argument('--task', default='easy',
                    choices=['easy', 'medium', 'hard'])
    ap.add_argument('--timesteps', type=int, default=100_000)
    ap.add_argument('--n_envs', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--log_dir', default='./runs/megaforge')
    ap.add_argument('--push_to_hub', action='store_true',
                    help='Push model + reward curve PNG to HF Hub')
    ap.add_argument('--hf_repo_id', default=None,
                    help='e.g. username/crossmill-megaforge-easy')
    args = ap.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)

    # ---- Pre-training baseline (PID heuristic) ----
    print('\n' + '=' * 60)
    print('PRE-TRAINING BASELINE (PID heuristic)')
    print('=' * 60)
    baseline = PIDBaselineAgent()
    pre = grader(baseline, task_id=args.task,
                 num_eval_episodes=10, base_seed=5000)
    print(f"  grader_score: {pre['grader_score']:.3f}")
    print(f"  mean_reward:  {pre['mean_reward']:.3f}")
    print(f"  carbon_err:   {pre['mean_carbon_error_pct']:.4f}")
    print(f"  coke_rate:    {pre['mean_coke_rate_kgpt']:.1f} kg/t")

    # ---- Training ----
    print('\n' + '=' * 60)
    print('TRAINING RecurrentPPO')
    print('=' * 60)

    monitor_path = os.path.join(args.log_dir, 'monitor')
    vec = DummyVecEnv(
        [make_env(args.task, i, args.seed) for i in range(args.n_envs)])
    vec = VecMonitor(vec, filename=monitor_path)

    model = RecurrentPPO(
        'MlpLstmPolicy', vec,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        verbose=1,
        seed=args.seed,
        # tensorboard_log=args.log_dir,  # omit if tensorboard not installed
    )

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps)
    train_time = time.time() - t0
    print(f'\nTraining finished in {train_time / 60:.1f} min')

    model_zip_path = os.path.join(args.log_dir,
                                  f'megaforge-{args.task}-ppo.zip')
    model.save(model_zip_path)
    print(f'Model saved: {model_zip_path}')

    # ---- Post-training grader ----
    print('\n' + '=' * 60)
    print('POST-TRAINING GRADER (RecurrentPPO policy)')
    print('=' * 60)

    shim = MegaForgeGymShim(task_id=args.task, seed=args.seed)
    adapter = PolicyAdapter(model, shim)
    post = grader(adapter, task_id=args.task,
                  num_eval_episodes=50, base_seed=6000)
    print(f"  grader_score: {post['grader_score']:.3f}")
    print(f"  mean_reward:  {post['mean_reward']:.3f}")
    print(f"  safety_viol:  {post['safety_violation_rate']:.3f}")
    print(f"  catastrophic: {post['catastrophic_rate']:.3f}")
    print(f"  carbon_err:   {post['mean_carbon_error_pct']:.4f}")
    print(f"  coke_rate:    {post['mean_coke_rate_kgpt']:.1f} kg/t")
    print(f"  CO2:          {post['mean_co2_emissions_kgpt']:.1f} kg/t")

    # ---- Reward curve PNG ----
    print('\n' + '=' * 60)
    print('PLOTTING REWARD CURVE')
    print('=' * 60)

    curve_png_path = os.path.join(args.log_dir,
                                  f'reward_curve_{args.task}.png')
    csv_path = monitor_path + '.monitor.csv'
    if os.path.exists(csv_path):
        plot_reward_curve(csv_path, curve_png_path,
                          task_id=args.task,
                          baseline_score=pre['mean_reward'],
                          final_score=post['mean_reward'])
        print(f'Saved: {curve_png_path}')
    else:
        print(f'WARNING: monitor CSV not found at {csv_path}; '
              f'reward curve skipped')

    # ---- Improvement summary ----
    summary = {
        'task_id': args.task,
        'pre_score': pre['grader_score'],
        'post_score': post['grader_score'],
        'delta': post['grader_score'] - pre['grader_score'],
        'safety_violation_rate': post['safety_violation_rate'],
        'catastrophic_rate': post['catastrophic_rate'],
        'mean_carbon_error_pct': post['mean_carbon_error_pct'],
        'mean_coke_rate_kgpt': post['mean_coke_rate_kgpt'],
        'mean_co2_emissions_kgpt': post['mean_co2_emissions_kgpt'],
    }
    summary_path = os.path.join(args.log_dir, f'summary_{args.task}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n' + '=' * 60)
    print('IMPROVEMENT')
    print('=' * 60)
    print(f'  grader_score: {pre["grader_score"]:.3f} -> '
          f'{post["grader_score"]:.3f}  '
          f'(delta {summary["delta"]:+.3f})')
    print(f'  summary_json: {summary_path}')
    print(f'  reward_curve: {curve_png_path}')

    # ---- HuggingFace Hub push (optional) ----
    if args.push_to_hub:
        if not args.hf_repo_id:
            print('ERROR: --push_to_hub requires --hf_repo_id user/repo-name')
            sys.exit(1)
        print(f'\n' + '=' * 60)
        print(f'PUSHING ARTIFACTS TO HF HUB ({args.hf_repo_id})')
        print('=' * 60)
        push_artifacts_to_hub(repo_id=args.hf_repo_id,
                              model_zip_path=model_zip_path,
                              curve_png_path=curve_png_path,
                              summary=summary)

    print('\nDone.')


if __name__ == '__main__':
    main()
