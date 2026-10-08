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

sys.path.insert(0, str(Path(__file__).resolve().parent))   # to import uq_metrics.py (same folder)
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


def to_array(x):
    """Turn whatever np.load(allow_pickle=True) returned into one float ndarray."""
    import numpy as np
    if hasattr(x, "detach"):                                   # torch tensor
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        if x.dtype != object:
            return x
        if x.ndim == 0:                                         # 0-d object array wrapping something
            return to_array(x.item())
        return np.concatenate([to_array(e) for e in x.tolist()], axis=0)   # list of batches
    if isinstance(x, dict):
        if len(x) == 1:
            return to_array(next(iter(x.values())))
        raise ValueError(f"dict with several keys: {list(x)}")
    if isinstance(x, (list, tuple)):
        return np.concatenate([to_array(e) for e in x], axis=0)
    return np.asarray(x)


def describe(x):
    import numpy as np
    d = {"type": type(x).__name__}
    if isinstance(x, np.ndarray):
        d.update(dtype=str(x.dtype), shape=list(x.shape))
        if x.dtype == object and x.ndim > 0 and len(x):
            d["first_element"] = describe(x.flat[0])
        elif x.dtype == object and x.ndim == 0:
            d["item"] = describe(x.item())
    elif isinstance(x, dict):
        d["keys"] = list(x)[:10]
    elif isinstance(x, (list, tuple)):
        d["len"] = len(x)
        if x:
            d["first_element"] = describe(x[0])
    elif hasattr(x, "shape"):
        d["shape"] = list(x.shape)
    return d


def load_array(path, horizon, nodes):
    """BasicTS 1.0 writes test_results/*.npy as RAW float32 dumps (no .npy header) -> shape (N, horizon, nodes).
    Evidence (PEMS04): size = 3376*12*307*4 bytes, no b'\\x93NUMPY' magic, first values 104.8 / 112.0 (flows).
    A normal .npy file (with header) is also accepted."""
    import numpy as np
    with open(path, "rb") as f:
        magic = f.read(6)
    if magic == b"\x93NUMPY":
        return to_array(np.load(path, allow_pickle=True)), "npy"
    raw = np.fromfile(path, dtype="<f4")
    per_sample = horizon * nodes
    if raw.size % per_sample:
        raise ValueError(f"{raw.size} floats is not a multiple of horizon*nodes={per_sample}")
    return raw.reshape(-1, horizon, nodes), "raw_float32"


def recompute(run_dir, null_val, horizon, nodes):
    """Independent metrics from the saved arrays. Never raises: returns {'error': ..., 'describe': ...}."""
    try:
        import numpy as np
    except ImportError:
        return {"error": "numpy not available"}
    d = Path(run_dir) / "test_results"
    if not (d / "prediction.npy").exists() or not (d / "targets.npy").exists():
        return {"error": "test_results/prediction.npy or targets.npy not found"}
    try:
        p, kind_p = load_array(d / "prediction.npy", horizon, nodes)
        t, kind_t = load_array(d / "targets.npy", horizon, nodes)
        p, t = p.astype("float64"), t.astype("float64")
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    info = {"prediction_shape": list(p.shape), "targets_shape": list(t.shape),
            "file_format": kind_p, "layout_assumed": "(samples, horizon, nodes)"}
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
    per_h = [metrics(p[:, h], t[:, h]) for h in range(p.shape[1])]
    out["mae_by_horizon"] = [round(x["mae"], 4) if x else None for x in per_h]
    maes = [x["mae"] for x in per_h if x]
    # If the layout assumption is right, error should (mostly) grow with the horizon.
    out["mae_grows_with_horizon"] = bool(len(maes) > 1 and maes[-1] > maes[0] and
                                         sum(b >= a for a, b in zip(maes, maes[1:])) >= 0.8 * (len(maes) - 1))
    for h in (3, 6, 12):
        if h <= p.shape[1]:
            out["horizons"][f"h{h}"] = per_h[h - 1]
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
    ap.add_argument("--output-len", type=int, default=12)
    ap.add_argument("--num-nodes", type=int, default=None, help="default: num_vars from datasets/<dataset>/meta.json")
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

    nodes = a.num_nodes
    if nodes is None:
        meta = Path(a.ckpt_root).resolve().parent / "datasets" / a.dataset / "meta.json"
        nodes = json.loads(meta.read_text())["num_vars"] if meta.exists() else None
    rec = recompute(run_dir, a.null_val, a.output_len, nodes) if nodes else {"error": "num_nodes unknown (pass --num-nodes)"}
    bt = {"mae": m["MAE"], "rmse": m["RMSE"], "mape": m["MAPE"]}
    check = None
    if rec.get("overall"):
        check = {k: abs(rec["overall"][k] - bt[k]) / max(abs(bt[k]), 1e-12) for k in bt}

    uq_full, uq_error = None, None
    lv_path = run_dir / "test_results" / "log_var.npy"
    if lv_path.exists() and nodes:
        try:
            from uq_metrics import compute_uq
            tdir = run_dir / "test_results"
            p_arr, _ = load_array(tdir / "prediction.npy", a.output_len, nodes)
            t_arr, _ = load_array(tdir / "targets.npy", a.output_len, nodes)
            lv_arr, _ = load_array(lv_path, a.output_len, nodes)
            uq_full = compute_uq(p_arr, t_arr, lv_arr, a.null_val)
        except Exception as e:
            uq_error = f"{type(e).__name__}: {e}"
    uq_metrics = {"picp": None, "mpiw": None, "nll": None}
    if uq_full:
        uq_metrics = {"picp": uq_full["picp_95"], "mpiw": uq_full["mpiw_95"], "nll": uq_full["nll"], **uq_full}
    elif uq_error:
        uq_metrics["error"] = uq_error

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
        "uq_metrics": uq_metrics,
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
        print("MAE by horizon:", rec.get("mae_by_horizon"), "| grows with horizon:", rec.get("mae_grows_with_horizon"))
        for h, v in result["horizon_metrics"].items():
            print(f"  {h}:", {k: round(x, 4) for k, x in v.items()})
    else:
        print("Recompute skipped:", rec.get("error"))
        if rec.get("describe"):
            print("What the .npy files contain:", json.dumps(rec["describe"]))
    if uq_full:
        print("UQ (95%% interval, original unit): PICP %.3f | MPIW %.2f | NLL %.3f | CRPS %.3f | ECE %.3f | PICP90 %.3f"
              % (uq_full["picp_95"], uq_full["mpiw_95"], uq_full["nll"], uq_full["crps"], uq_full["ece"], uq_full["picp_90"]))
    elif uq_error:
        print("UQ metrics failed:", uq_error)
    print(f"epochs trained (all logs): {epochs_trained} | train time: {result['train_time_s']} s")


if __name__ == "__main__":
    main()
