#!/usr/bin/env bash
# Multiseed task-success analysis for the paper's four best checkpoints (S1-S4).
#
# Runs scripts/evaluate/evaluate_paper_stats.py once per evaluation seed, saves a
# per-seed JSON, then aggregates mean +/- std of the success rate (and reward)
# across seeds for every policy (S1-S4) and mode (deterministic/stochastic).
#
# All four runs are single training seed 42; varying the *evaluation* seed probes
# how robust the reported success rates are to environment/policy sampling noise.
#
# Usage:
#   scripts/evaluate/run_multiseed.sh                 # defaults below
#   SEEDS="1 2 3 4 5" EPISODES=100 scripts/evaluate/run_multiseed.sh
#   scripts/evaluate/run_multiseed.sh --only S1,S2    # extra args go to the eval script
set -euo pipefail

# --- resolve repo root (this script lives in scripts/evaluate/) ---------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

# --- config (override via env) ------------------------------------------------
PY="${PY:-$ROOT/.venv/bin/python}"
SEEDS="${SEEDS:-0 1 2 3 4}"
EPISODES="${EPISODES:-100}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs_best/multiseed}"
EXTRA_ARGS=("$@")   # e.g. --only S1,S2

mkdir -p "$OUT_DIR"

echo "python   : $PY"
echo "seeds    : $SEEDS"
echo "episodes : $EPISODES"
echo "out dir  : $OUT_DIR"
[ "${#EXTRA_ARGS[@]}" -gt 0 ] && echo "extra    : ${EXTRA_ARGS[*]}" || true
echo

# --- per-seed evaluation ------------------------------------------------------
for seed in $SEEDS; do
  echo "======================================================================"
  echo "  seed $seed"
  echo "======================================================================"
  "$PY" scripts/evaluate/evaluate_paper_stats.py \
    --episodes "$EPISODES" \
    --seed "$seed" \
    --out "$OUT_DIR/seed_${seed}.json" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
done

# --- aggregate across seeds ---------------------------------------------------
echo
echo "======================================================================"
echo "  AGGREGATE across seeds: $SEEDS"
echo "======================================================================"
"$PY" - "$OUT_DIR" <<'PY'
import glob, json, os, sys
import numpy as np

out_dir = sys.argv[1]
files = sorted(glob.glob(os.path.join(out_dir, "seed_*.json")))
rows = []
for f in files:
    with open(f) as fh:
        rows.extend(json.load(fh))

# group by (key, task, deterministic)
groups = {}
for r in rows:
    k = (r["key"], r["task"], r["deterministic"])
    groups.setdefault(k, []).append(r)

hdr = f"{'pol':<4}{'task':<20}{'mode':<14}{'success% (mean±std)':<22}{'reward (mean±std)':<22}{'n':>4}"
print(hdr)
print("-" * len(hdr))
for (key, task, det), items in sorted(groups.items(), key=lambda x: (x[0][0], not x[0][2])):
    succ = np.array([i["success_rate_pct"] for i in items])
    rew  = np.array([i["mean_reward"] for i in items])
    mode = "deterministic" if det else "stochastic"
    print(f"{key:<4}{task:<20}{mode:<14}"
          f"{succ.mean():6.1f} ± {succ.std():4.1f}        "
          f"{rew.mean():7.0f} ± {rew.std():4.0f}      {len(items):>4}")

agg_path = os.path.join(out_dir, "aggregate.json")
agg = []
for (key, task, det), items in groups.items():
    succ = [i["success_rate_pct"] for i in items]
    rew  = [i["mean_reward"] for i in items]
    agg.append({
        "key": key, "task": task,
        "mode": "deterministic" if det else "stochastic",
        "n_seeds": len(items),
        "success_rate_mean": float(np.mean(succ)),
        "success_rate_std": float(np.std(succ)),
        "reward_mean": float(np.mean(rew)),
        "reward_std": float(np.std(rew)),
    })
with open(agg_path, "w") as fh:
    json.dump(agg, fh, indent=2)
print(f"\nWrote {agg_path}")
PY

echo
echo "Done. Per-seed JSON + aggregate.json in: $OUT_DIR"
