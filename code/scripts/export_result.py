"""Turn a BasicTS training log into ONE results JSON (see results/_schema.json).

Example (run from the BasicTS folder):
  python ../scripts/export_result.py --log-dir checkpoints --model STID \
      --dataset PEMS04 --seed 0 --config baselines/STID/PEMS04.py --out /content/results_out

It looks for lines like:  "... horizon 12, Test MAE: x, Test RMSE: y, Test MAPE: z"
If your BasicTS version prints test metrics differently, the script says so and
exits; paste the last ~30 log lines to fix the pattern.
"""
import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

PAT = re.compile(
    r"horizon\s*(\d+)\D+?MAE\D*?(\d+\.?\d*)\D+?RMSE\D*?(\d+\.?\d*)\D+?MAPE\D*?(\d+\.?\d*)",
    re.I,
)


def newest_log(log_dir):
    logs = sorted(Path(log_dir).rglob("*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def git_sha():
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--status", default="done", help="done | smoke_test | pending")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    log = newest_log(a.log_dir)
    if log is None:
        sys.exit(f"No *.log file found under {a.log_dir}")
    text = log.read_text(errors="ignore")
    per_h = {}
    for h, mae, rmse, mape in PAT.findall(text):
        per_h[int(h)] = {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}  # last one wins
    if not per_h:
        sys.exit(f"No test-metric lines matched in {log}.\nPaste the last 30 lines of that file to the assistant.")

    hs = sorted(per_h)
    n = len(hs)
    point = {
        "mae": sum(per_h[h]["mae"] for h in hs) / n,
        "rmse": math.sqrt(sum(per_h[h]["rmse"] ** 2 for h in hs) / n),  # sqrt of mean MSE
        "mape": sum(per_h[h]["mape"] for h in hs) / n,
    }
    horizon = {f"h{h}": per_h[h] for h in (3, 6, 12) if h in per_h}

    result = {
        "model": a.model,
        "dataset": a.dataset,
        "seed": a.seed,
        "source": "reproduced",
        "status": a.status,
        "commit": git_sha(),
        "config": a.config,
        "epochs": a.epochs,
        "log_file": str(log),
        "horizons_found": hs,
        "point_metrics": point,
        "horizon_metrics": horizon,
        "uq_metrics": {"picp": None, "mpiw": None, "nll": None},
        "train_time_s": None,
        "mape_note": "Value as printed by BasicTS; check whether it is a fraction (0.12) or percent (12.0) before comparing with the paper.",
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if a.status == "smoke_test" else ""
    path = out / f"{a.model}_{a.dataset}_seed{a.seed}{suffix}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Wrote", path)
    print(json.dumps(point, indent=2))


if __name__ == "__main__":
    main()
