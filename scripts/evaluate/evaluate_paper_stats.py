#!/usr/bin/env python3
"""Task-success evaluation of the paper's four best checkpoints (S1-S4).

Unlike evaluate_stats.py (which builds each env with default kwargs), this
script reconstructs every environment from the run's own .hydra/config.yaml,
so S2-S4 are evaluated on exactly the geometry and reward configuration they
were trained on. It reports, per task and per policy mode
(deterministic/stochastic):

  * mean +/- std episode reward and mean episode length over N episodes;
  * a true task-completion success rate:
      - S1  (Humanoid-v5):          episode survives the full 1000-step horizon
      - S2  (stairs):               agent reaches the top step (max_step_reached
                                    >= num_steps from the training config)
      - S3/S4 (circuit):            all waypoints reached (circuit_complete)
  * mean waypoints reached for the circuit tasks.

Usage:
    python scripts/evaluate/evaluate_paper_stats.py [--episodes 100] [--seed 123]
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
    {"key": "S1", "task": "Flat locomotion", "run": "outputs_best/2025-10-28/17-47-25"},
    {"key": "S2", "task": "Stair climbing", "run": "outputs_best/2025-12-06/17-36-50"},
    {"key": "S3", "task": "Circuit (flat)", "run": "outputs_best/2025-12-23/14-37-50"},
    {"key": "S4", "task": "Circuit + stairs", "run": "outputs_best/2025-12-24/16-29-37"},
]


def load_run_config(run_dir: str) -> dict:
    with open(os.path.join(run_dir, ".hydra", "config.yaml")) as f:
        return yaml.safe_load(f)


def episode_success(key: str, make_kwargs: dict, length: int, last_info: dict) -> bool:
    if key == "S1":
        return length >= 1000  # survived the full horizon (no health termination)
    if key == "S2":
        return int(last_info.get("max_step_reached", 0)) >= int(make_kwargs["num_steps"])
    # S3 / S4
    n_wp = len(make_kwargs["waypoints"])
    return bool(last_info.get("circuit_complete", False)) or (
        int(last_info.get("waypoints_reached", 0)) >= n_wp
    )


def evaluate(cfg_run: dict, episodes: int, seed: int, deterministic: bool) -> dict:
    run_dir = os.path.join(ROOT, cfg_run["run"])
    cfg = load_run_config(run_dir)
    env_id = cfg["env"]["name"]
    make_kwargs = dict(cfg["env"]["make_kwargs"] or {})
    make_kwargs["render_mode"] = None

    venv = DummyVecEnv([make_single_env(env_id, make_kwargs, monitor=False, seed=seed)])
    venv = maybe_load_vecnormalize(venv, os.path.join(run_dir, "vecnormalize_final.pkl"))
    model = PPO.load(os.path.join(run_dir, "eval", "best_model.zip"), device="cpu")

    rewards, lengths, successes, waypoints = [], [], [], []
    try:
        for _ in range(episodes):
            obs = venv.reset()
            done, ep_rew, steps = False, 0.0, 0
            last_info: dict = {}
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, r, dones, infos = venv.step(action)
                ep_rew += float(r[0])
                steps += 1
                done = bool(dones[0])
                last_info = infos[0]
            rewards.append(ep_rew)
            lengths.append(steps)
            successes.append(episode_success(cfg_run["key"], make_kwargs, steps, last_info))
            if cfg_run["key"] in ("S3", "S4"):
                waypoints.append(int(last_info.get("waypoints_reached", 0)))
    finally:
        venv.close()

    out = {
        "key": cfg_run["key"],
        "task": cfg_run["task"],
        "env_id": env_id,
        "episodes": episodes,
        "deterministic": deterministic,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "cv_pct": float(100.0 * np.std(rewards) / abs(np.mean(rewards))),
        "mean_length": float(np.mean(lengths)),
        "success_rate_pct": float(100.0 * np.mean(successes)),
    }
    if waypoints:
        out["mean_waypoints_reached"] = float(np.mean(waypoints))
        out["n_waypoints"] = len(make_kwargs["waypoints"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    ap.add_argument("--only", type=str, default=None, help="Comma-separated keys, e.g. S1,S2")
    args = ap.parse_args()

    keys = set(args.only.split(",")) if args.only else None
    results = []
    for cfg_run in RUNS:
        if keys and cfg_run["key"] not in keys:
            continue
        for deterministic in (True, False):
            mode = "deterministic" if deterministic else "stochastic"
            print(f"\n=== {cfg_run['key']} {cfg_run['task']} [{mode}] ===", flush=True)
            res = evaluate(cfg_run, args.episodes, args.seed, deterministic)
            results.append(res)
            wp = (
                f"  waypoints {res['mean_waypoints_reached']:.2f}/{res['n_waypoints']}"
                if "mean_waypoints_reached" in res
                else ""
            )
            print(
                f"reward {res['mean_reward']:.0f} +/- {res['std_reward']:.0f} "
                f"(CV {res['cv_pct']:.1f}%)  len {res['mean_length']:.0f}  "
                f"success {res['success_rate_pct']:.1f}%{wp}",
                flush=True,
            )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
