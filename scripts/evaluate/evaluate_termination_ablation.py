#!/usr/bin/env python3
"""Evaluation-time ablation of terrain-relative vs world-frame termination.

Takes the trained S2 (stairs) and S4 (circuit+stairs) checkpoints — both
trained WITH terrain-relative health checks (check_healthy_z_relative=true,
healthy_z_range [1.0, 2.0]) — and evaluates each policy under both termination
modes, changing nothing else:

  * relative:   as trained (reproduces the paper's Table III numbers)
  * world:      check_healthy_z_relative=false, same [1.0, 2.0] range applied
                to absolute z (the pre-Dec-24 world-frame check)

For the world-frame mode it additionally records where each episode ends
(final x position and terrain height there), to show terminations cluster on
the elevated terrain rather than at genuine falls.

Usage:
    python scripts/evaluate/evaluate_termination_ablation.py \
        [--episodes 100] [--seed 123] [--out outputs_best/termination_ablation.json]
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.make_env import make_single_env
from utils.vecnorm_io import maybe_load_vecnormalize

import envs  # noqa: F401  (registers custom environments)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RUNS = [
    {"key": "S2", "task": "Stair climbing", "run": "outputs_best/2025-12-06/17-36-50"},
    {"key": "S4", "task": "Circuit + stairs", "run": "outputs_best/2025-12-24/16-29-37"},
]


def load_run_config(run_dir: str) -> dict:
    with open(os.path.join(run_dir, ".hydra", "config.yaml")) as f:
        return yaml.safe_load(f)


def episode_success(key: str, make_kwargs: dict, length: int, last_info: dict) -> bool:
    if key == "S2":
        return int(last_info.get("max_step_reached", 0)) >= int(make_kwargs["num_steps"])
    # S4
    n_wp = len(make_kwargs["waypoints"])
    return bool(last_info.get("circuit_complete", False)) or (
        int(last_info.get("waypoints_reached", 0)) >= n_wp
    )


def evaluate(cfg_run: dict, mode: str, episodes: int, seed: int) -> dict:
    run_dir = os.path.join(ROOT, cfg_run["run"])
    cfg = load_run_config(run_dir)
    env_id = cfg["env"]["name"]
    make_kwargs = dict(cfg["env"]["make_kwargs"] or {})
    make_kwargs["render_mode"] = None
    if mode == "world":
        make_kwargs["check_healthy_z_relative"] = False

    venv = DummyVecEnv([make_single_env(env_id, make_kwargs, monitor=False, seed=seed)])
    venv = maybe_load_vecnormalize(venv, os.path.join(run_dir, "vecnormalize_final.pkl"))
    model = PPO.load(os.path.join(run_dir, "eval", "best_model.zip"), device="cpu")

    base_env = venv.venv.envs[0].unwrapped if hasattr(venv, "venv") else venv.envs[0].unwrapped

    rewards, lengths, successes = [], [], []
    final_x, final_terrain_h, progress = [], [], []
    try:
        for _ in range(episodes):
            obs = venv.reset()
            done, ep_rew, steps = False, 0.0, 0
            last_info: dict = {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, dones, infos = venv.step(action)
                ep_rew += float(r[0])
                steps += 1
                done = bool(dones[0])
                last_info = infos[0]
            rewards.append(ep_rew)
            lengths.append(steps)
            successes.append(episode_success(cfg_run["key"], make_kwargs, steps, last_info))
            x = float(last_info.get("x_position", np.nan))
            y = float(last_info.get("y_position", np.nan))
            final_x.append(x)
            if hasattr(base_env, "_get_terrain_height_at") and np.isfinite(x):
                final_terrain_h.append(float(base_env._get_terrain_height_at(x, y)))
            if cfg_run["key"] == "S2":
                progress.append(int(last_info.get("max_step_reached", 0)))
            else:
                progress.append(int(last_info.get("waypoints_reached", 0)))
    finally:
        venv.close()

    terrain_h = np.array(final_terrain_h) if final_terrain_h else np.array([np.nan])
    out = {
        "key": cfg_run["key"],
        "task": cfg_run["task"],
        "mode": mode,
        "episodes": episodes,
        "seed": seed,
        "success_rate_pct": float(100.0 * np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
        "mean_progress": float(np.mean(progress)),
        "final_x_mean": float(np.nanmean(final_x)),
        "final_x_median": float(np.nanmedian(final_x)),
        "pct_end_on_elevated_terrain": float(100.0 * np.mean(terrain_h > 1e-6)),
        "final_terrain_h_mean": float(np.nanmean(terrain_h)),
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    args = ap.parse_args()

    results = []
    for cfg_run in RUNS:
        for mode in ("relative", "world"):
            print(f"\n=== {cfg_run['key']} {cfg_run['task']} [{mode} termination] ===", flush=True)
            res = evaluate(cfg_run, mode, args.episodes, args.seed)
            results.append(res)
            print(
                f"success {res['success_rate_pct']:.1f}%  "
                f"reward {res['mean_reward']:.0f} +/- {res['std_reward']:.0f}  "
                f"len {res['mean_length']:.0f} (median {res['median_length']:.0f})  "
                f"progress {res['mean_progress']:.2f}  "
                f"final x {res['final_x_median']:.2f}  "
                f"ends on elevated terrain {res['pct_end_on_elevated_terrain']:.0f}%",
                flush=True,
            )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
