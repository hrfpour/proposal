"""Turn a BasicTS 1.0 run into ONE results JSON (see results/_schema.json).

What it does
  1. Finds the newest run folder under --ckpt-root and reads, from its newest
     training_log_*.log, the LAST line "Result <test>: [...]" (= best checkpoint on the test set).
  2. Sums training time / counts epochs over ALL logs of that folder (BasicTS resumes
     automatically, so one run can have several log files).
  3. If numpy is available, RECOMPUTES MAE/RMSE/MAPE from test_results/prediction.npy and
     targets.npy (zeros masked), overall and for horizons 3/6/12, and compares with BasicTS.
  4. Copies cfg.json and test_metrics.json next to the output for provenance.

Example (Colab):
  python code/scripts/export_result.py --ckpt-root code/BasicTS/checkpoints \
     --model STID --dataset PEMS04 --seed 42 --epochs 2 --status smoke_test \
     --device cpu --out /content/results_out
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

RESULT = re.compile(r"Result <test>: \[(.*?)\]")
TRAIN_RESULT = re.compile(r"Result <train>: \[")
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


def recompute(run_dir, null_val):
    """Independent metrics from the saved arrays. Returns a dict (or {'error': ...})."""
    try:
        import numpy as np
    except ImportError:
        return {"error": "numpy not available"}
    d = Path(run_dir) / "test_results"
    if not (d / "prediction.npy").exists() or not (d / "targets.npy").exists():
        return {"error": "test_results/prediction.npy or targets.npy not found"}
    p = np.load(d / "prediction.npy").astype("float64")
    t = np.load(d / "targets.npy").astype("float64")
    info = {"prediction_shape": list(p.shape), "targets_shape": list(t.shape)}
    if p.shape != t.shape:
        return {**info, "error": "shape mismatch"}

    def metrics(pp, tt):
        mask = np.abs(tt - null_val) > 1e-5
        if not mask.any():
            return None
        e = pp[mask] - tt[mask]
        return {"mae": float(np.abs(e).mean()),
                "rmse": float(np.sqrt((e ** 2).mean())),
                "mape": float((np.abs(e) / np.abs(tt[mask])).mean())}

    out = {**info, "null_val": null_val, "overall": metrics(p, t), "horizons": {}}
    if p.ndim >= 3:
        for h in (3, 6, 12):
            if h <= p.shape[1]:
                out["horizons"][f"h{h}"] = metrics(p[:, h - 1], t[:, h - 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None, help="cpu | gpu (for the record)")
    ap.add_argument("--null-val", type=float, default=0.0)
    ap.add_argument("--status", default="done", help="done | smoke_test | pending")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    all_logs = sorted(Path(a.ckpt_root).rglob("training_log_*.log"), key=lambda p: p.stat().st_mtime)
    if not all_logs:
        sys.exit(f"No training_log_*.log under {a.ckpt_root}")
    run_dir = all_logs[-1].parent
    run_logs = sorted(run_dir.glob("training_log_*.log"), key=lambda p: p.stat().st_mtime)

    log, block = None, None
    for lg in reversed(run_logs):                       # newest log that contains a test result
        found = RESULT.findall(lg.read_text(errors="ignore"))
        if found:
            log, block = lg, found[-1]
            break
    if block is None:
        sys.exit(f"No 'Result <test>' line in {run_dir}. Paste the last 30 lines of the newest log.")
    m = {k: float(v) for k, v in PAIR.findall(block)}
    for key in ("MAE", "RMSE", "MAPE"):
        if key not in m:
            sys.exit(f"{key} not found in: {block}")

    texts = [lg.read_text(errors="ignore") for lg in run_logs]
    train_time = sum(float(x) for t in texts for x in TRAIN_TIME.findall(t))
    epochs_trained = sum(len(TRAIN_RESULT.findall(t)) for t in texts)
    params = next((PARAMS.findall(t)[0] for t in texts if PARAMS.findall(t)), None)

    rec = recompute(run_dir, a.null_val)
    bt = {"mae": m["MAE"], "rmse": m["RMSE"], "mape": m["MAPE"]}
    check = None
    if rec.get("overall"):
        check = {k: abs(rec["overall"][k] - bt[k]) / max(abs(bt[k]), 1e-12) for k in bt}

    result = {
        "model": a.model, "dataset": a.dataset, "seed": a.seed,
        "source": "reproduced", "status": a.status, "commit": git_sha(),
        "epochs_requested": a.epochs, "epochs_trained": epochs_trained,
        "device": a.device,
        "num_parameters": int(params) if params else None,
        "run_dir": str(run_dir), "log_files": [str(x) for x in run_logs],
        "point_metrics": {"mae": m["MAE"], "rmse": m["RMSE"], "mape": m["MAPE"],
                          "mse": m.get("MSE"), "wape": m.get("WAPE")},
        "horizon_metrics": {k: v for k, v in rec.get("horizons", {}).items() if v},
        "uq_metrics": {"picp": None, "mpiw": None, "nll": None},
        "train_time_s": round(train_time, 2),
        "recomputed_from_arrays": rec,
        "relative_diff_vs_basicts": check,
        "notes": ["MAPE is a fraction (0.30 = 30%); multiply by 100 to compare with papers (to be confirmed by the recomputed MAPE).",
                  "WAPE is not used: its value looked inconsistent with MAE (possibly computed on the normalized scale)."],
    }

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if a.status == "smoke_test" else ""
    stem = f"{a.model}_{a.dataset}_seed{a.seed}{suffix}"
    (out / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name in ("cfg.json", "test_metrics.json"):
        if (run_dir / name).exists():
            shutil.copy(run_dir / name, out / f"{stem}_{name}")

    print("Wrote", out / f"{stem}.json")
    print("BasicTS  :", {k: round(v, 4) for k, v in bt.items()})
    if rec.get("overall"):
        print("Recomputed:", {k: round(v, 4) for k, v in rec["overall"].items()})
        print("Rel. diff :", {k: f"{v:.2%}" for k, v in check.items()})
        for h, v in result["horizon_metrics"].items():
            print(f"  {h}:", {k: round(x, 4) for k, x in v.items()})
    else:
        print("Recompute skipped:", rec.get("error"))
    print(f"epochs trained (all logs): {epochs_trained} | train time: {result['train_time_s']} s")


if __name__ == "__main__":
    main()
