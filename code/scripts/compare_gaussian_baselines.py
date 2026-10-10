"""Is the LEARNED predictive variance better than a constant-variance Gaussian? (numpy only, no GPU)

Uses the arrays BasicTS and our hooks saved at the end of training (test_results/prediction.npy, targets.npy,
log_var.npy, all raw float32, original unit). All variants are scored with the same metrics (code/scripts/uq_metrics.py).

A) SAME MEAN (the reference probabilistic model's own mean) -> isolates the variance model:
     learned sigma | constant sigma = validation RMSE (no test information) |
     per-horizon sigma, per-node sigma, per-horizon x node sigma  (ORACLE: estimated on the test residuals, so they
     are optimistic reference points for what any static variance can do)
B) other probabilistic runs (e.g. MC Dropout) with their own mean and learned variance
C) deterministic AGCRN + constant sigma (validation RMSE) and + oracle per-horizon x node sigma

Proper scoring rules (NLL, CRPS; lower = better) are comparable between rows. PICP/MPIW must be read together;
`MPIW@95%` is the width after rescaling sigma so that coverage is exactly 95% (same oracle rescaling for every row).

Example (Colab, Drive mounted):
  python code/scripts/compare_gaussian_baselines.py --ckpt-root /content/drive/MyDrive/proposal_ckpt \
      --dataset PEMS04 --seed 42 --num-nodes 307 --prob AGCRN_PROB --prob AGCRN_PROB_DROP10 --det AGCRN \
      --out-dir /content/drive/MyDrive/proposal_runs/full
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_result import load_array      # noqa: E402  (raw float32 loader)
from uq_metrics import compute_uq          # noqa: E402

VAL = re.compile(r"Result <val>: \[(.*?)\]")
PAIR = re.compile(r"val/(\w+): ([-+0-9.eE]+)")


def find_run_dir(base):
    """Newest run folder (the one that contains test_results/prediction.npy) under base."""
    cands = [p.parent.parent for p in Path(base).rglob("prediction.npy") if p.parent.name == "test_results"]
    if not cands:
        raise FileNotFoundError(f"No test_results/prediction.npy under {base}")
    return max(cands, key=lambda p: (p / "test_results" / "prediction.npy").stat().st_mtime)


def best_val_rmse(run_dir):
    """Validation RMSE of the epoch with the lowest validation MAE (= the checkpoint used for the test set)."""
    epochs = []
    for lg in sorted(Path(run_dir).glob("training_log_*.log"), key=lambda p: p.stat().st_mtime):
        for block in VAL.findall(lg.read_text(errors="ignore")):
            m = {k: float(v) for k, v in PAIR.findall(block)}
            if "MAE" in m and "RMSE" in m:
                epochs.append((m["MAE"], m["RMSE"]))
    if not epochs:
        raise ValueError(f"No validation lines in the logs of {run_dir}")
    return min(epochs)[1]


def load_run(base, horizon, nodes, need_log_var):
    run = find_run_dir(base)
    t = run / "test_results"
    pred, _ = load_array(t / "prediction.npy", horizon, nodes)
    targ, _ = load_array(t / "targets.npy", horizon, nodes)
    lv = None
    if need_log_var:
        if not (t / "log_var.npy").exists():
            raise FileNotFoundError(f"{t / 'log_var.npy'} missing: this run did not save its predicted variance")
        lv, _ = load_array(t / "log_var.npy", horizon, nodes)
    return run, pred, targ, lv


def point_metrics(pred, targ, null_val):
    m = np.abs(targ.astype("float64") - null_val) > 1e-5
    e = (pred.astype("float64") - targ.astype("float64"))[m]
    return float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean()))


def static_log_vars(pred, targ, null_val, val_rmse):
    """Constant-variance Gaussians. Returns {label: log_var array broadcastable to pred.shape}."""
    mask = (np.abs(targ.astype("float64") - null_val) > 1e-5).astype("float64")
    e2 = (pred.astype("float64") - targ.astype("float64")) ** 2 * mask
    floor = 1e-6

    def var(axes):
        return np.maximum(e2.sum(axis=axes, keepdims=True) / np.maximum(mask.sum(axis=axes, keepdims=True), 1.0), floor)

    return {
        "constant sigma = validation RMSE": np.full((1, 1, 1), math.log(val_rmse ** 2)),
        "per-horizon sigma (oracle)": np.log(var((0, 2))),
        "per-node sigma (oracle)": np.log(var((0, 1))),
        "per-horizon x node sigma (oracle)": np.log(var((0,))),
    }


def score(pred, targ, log_var, null_val):
    full = np.broadcast_to(log_var, pred.shape)
    r = compute_uq(pred, targ, full, null_val)
    mae, rmse = point_metrics(pred, targ, null_val)
    return {"mae": mae, "rmse": rmse, "picp_90": r["picp_90"], "picp_95": r["picp_95"], "mpiw_95": r["mpiw_95"],
            "mpiw_95_at_nominal": r["mpiw_95_scaled_to_nominal"], "k95": r["k95_to_nominal"],
            "nll": r["nll"], "crps": r["crps"], "ece": r["ece"], "mean_sigma": r["mean_sigma"]}


def table(rows):
    head = ("| variance model | MAE | PICP90 | PICP95 | MPIW95 | MPIW@95% | k95 | NLL | CRPS | ECE |\n"
            "|---|---|---|---|---|---|---|---|---|---|")
    body = [f"| {lab} | {r['mae']:.2f} | {r['picp_90']:.3f} | {r['picp_95']:.3f} | {r['mpiw_95']:.1f} | "
            f"{r['mpiw_95_at_nominal']:.1f} | {r['k95']:.2f} | {r['nll']:.3f} | {r['crps']:.3f} | {r['ece']:.3f} |"
            for lab, r in rows]
    return "\n".join([head] + body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--dataset", default="PEMS04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-nodes", type=int, required=True)
    ap.add_argument("--output-len", type=int, default=12)
    ap.add_argument("--null-val", type=float, default=0.0)
    ap.add_argument("--prob", action="append", required=True, help="model folder of a probabilistic run (repeatable); the first is the reference")
    ap.add_argument("--det", default=None, help="model folder of the deterministic run, e.g. AGCRN")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    def base(name):
        return Path(a.ckpt_root) / name / a.dataset / f"seed{a.seed}"

    sections, report = {}, {}
    ref_name = a.prob[0]
    ref_dir, pred, targ, lv = load_run(base(ref_name), a.output_len, a.num_nodes, True)
    print(f"reference run: {ref_dir}")
    rows_a = [(f"learned sigma ({ref_name})", score(pred, targ, lv, a.null_val))]
    val_rmse = best_val_rmse(ref_dir)
    for label, log_var in static_log_vars(pred, targ, a.null_val, val_rmse).items():
        rows_a.append((label, score(pred, targ, log_var, a.null_val)))
    sections["A"] = rows_a

    rows_b = []
    for name in a.prob[1:]:
        _, p2, t2, lv2 = load_run(base(name), a.output_len, a.num_nodes, True)
        rows_b.append((f"learned sigma ({name}, own mean)", score(p2, t2, lv2, a.null_val)))
    if rows_b:
        sections["B"] = rows_b

    if a.det:
        det_dir, pd, td, _ = load_run(base(a.det), a.output_len, a.num_nodes, False)
        det_val = best_val_rmse(det_dir)
        stat = static_log_vars(pd, td, a.null_val, det_val)
        sections["C"] = [(f"{a.det} + constant sigma = validation RMSE", score(pd, td, stat["constant sigma = validation RMSE"], a.null_val)),
                         (f"{a.det} + per-horizon x node sigma (oracle)", score(pd, td, stat["per-horizon x node sigma (oracle)"], a.null_val))]

    titles = {"A": f"A) Same mean ({ref_name}); only the variance model changes",
              "B": "B) Other probabilistic runs (own mean, learned variance)",
              "C": "C) Deterministic AGCRN with a constant variance"}
    md = [f"# Predictive-variance comparison: {a.dataset}, seed {a.seed}", "",
          "Original unit (vehicles / 5 min). Lower is better for MAE, NLL, CRPS, ECE. Nominal PICP90 = 0.90, PICP95 = 0.95.",
          "(oracle) rows use test residuals to set sigma: optimistic reference points, NOT usable methods.",
          "MPIW@95% = width after rescaling sigma by k95 so that coverage is exactly 95% (oracle rescaling, same for all rows).", ""]
    for key in ("A", "B", "C"):
        if key in sections:
            md += [f"## {titles[key]}", "", table(sections[key]), ""]
    learned = rows_a[0][1]
    non_leaky = rows_a[1][1]
    best_oracle = min((r for _, r in rows_a[2:]), key=lambda r: r["crps"])
    md += ["## Reading guide (automatic)", "",
           f"- learned sigma vs constant sigma (validation RMSE, no test info): CRPS {learned['crps']:.3f} vs {non_leaky['crps']:.3f} "
           f"({(non_leaky['crps'] - learned['crps']) / non_leaky['crps'] * 100:+.1f}% lower for learned), "
           f"NLL {learned['nll']:.3f} vs {non_leaky['nll']:.3f}.",
           f"- learned sigma vs the best ORACLE static sigma: CRPS {learned['crps']:.3f} vs {best_oracle['crps']:.3f} "
           f"({(best_oracle['crps'] - learned['crps']) / best_oracle['crps'] * 100:+.1f}% lower for learned). "
           "If this is not positive, a static variance already explains what the model learned.",
           f"- width at exactly 95% coverage: learned {learned['mpiw_95_at_nominal']:.1f} vs constant {non_leaky['mpiw_95_at_nominal']:.1f}.", ""]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"uq_comparison_{a.dataset}_seed{a.seed}"
    report = {k: [{"label": lab, **r} for lab, r in v] for k, v in sections.items()}
    (out / f"{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / f"{stem}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nWrote {out / (stem + '.md')} and .json")


if __name__ == "__main__":
    main()
