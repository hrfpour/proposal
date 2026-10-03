"""Turn a BasicTS 1.0 run into ONE results JSON (see results/_schema.json).

Reads the newest `training_log_*.log` under --ckpt-root and takes the LAST line
  "... Result <test>: [test/time: .., test/MAE: .., test/MSE: .., test/RMSE: .., test/MAPE: .., test/WAPE: ..]"
(= the final evaluation of the best checkpoint). Also copies cfg.json and
test_metrics.json next to the output for provenance.

Example (Colab):
  python code/scripts/export_result.py --ckpt-root code/BasicTS/checkpoints \
     --model STID --dataset PEMS04 --seed 42 --epochs 2 --status smoke_test \
     --device cpu --out /content/results_out
Standard library only.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

RESULT = re.compile(r"Result <test>: \[(.*?)\]")
PAIR = re.compile(r"test/(\w+): ([-+0-9.eEnaif]+)")
TRAIN_TIME = re.compile(r"train/time: ([0-9.]+)")
PARAMS = re.compile(r"Total parameters: (\d+)")


def git_sha():
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None, help="cpu | gpu (for the record)")
    ap.add_argument("--status", default="done", help="done | smoke_test | pending")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    logs = sorted(Path(a.ckpt_root).rglob("training_log_*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        sys.exit(f"No training_log_*.log under {a.ckpt_root}")
    log = logs[-1]
    text = log.read_text(errors="ignore")

    blocks = RESULT.findall(text)
    if not blocks:
        sys.exit(f"No 'Result <test>' line in {log}. Paste its last 30 lines to the assistant.")
    m = {k: float(v) for k, v in PAIR.findall(blocks[-1])}   # last block = best model on test set
    for key in ("MAE", "RMSE", "MAPE"):
        if key not in m:
            sys.exit(f"{key} not found in: {blocks[-1]}")

    params = PARAMS.findall(text)
    train_time = sum(float(x) for x in TRAIN_TIME.findall(text))
    result = {
        "model": a.model,
        "dataset": a.dataset,
        "seed": a.seed,
        "source": "reproduced",
        "status": a.status,
        "commit": git_sha(),
        "epochs": a.epochs,
        "device": a.device,
        "num_parameters": int(params[0]) if params else None,
        "log_file": str(log),
        "point_metrics": {"mae": m["MAE"], "rmse": m["RMSE"], "mape": m["MAPE"],
                          "mse": m.get("MSE"), "wape": m.get("WAPE")},
        "horizon_metrics": {},
        "uq_metrics": {"picp": None, "mpiw": None, "nll": None},
        "train_time_s": round(train_time, 2),
        "mape_note": "As printed by BasicTS. Check whether it is a fraction (0.12) or percent (12.0) before comparing with papers.",
    }

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if a.status == "smoke_test" else ""
    stem = f"{a.model}_{a.dataset}_seed{a.seed}{suffix}"
    (out / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name in ("cfg.json", "test_metrics.json"):
        src = log.parent / name
        if src.exists():
            shutil.copy(src, out / f"{stem}_{name}")
    print("Wrote", out / f"{stem}.json")
    print(json.dumps(result["point_metrics"], indent=2))


if __name__ == "__main__":
    main()
