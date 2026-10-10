"""Aggregate results/*.json (status == "done") into mean +- std tables.

Run from the repo root (standard library only):
  python code/scripts/summarize_results.py
Writes results/summary.md and results/summary.json and prints the table.
MAPE is stored by BasicTS as a fraction; it is shown here in %.
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"

# Paper-reported numbers (NOT reproduced). Source: Table 2 of the base paper (Li et al., Symmetry 2025, 17, 1007).
REFERENCE = {
    ("AGCRN", "PEMS04"): (19.83, 32.26, 12.97, "base paper Table 2"),
    ("AGCRN", "PEMS08"): (15.95, 25.22, 10.09, "base paper Table 2"),
    ("DGCRAN", "PEMS04"): (18.12, 30.02, 11.93, "base paper Table 2 (claimed, no code)"),
    ("DGCRAN", "PEMS08"): (13.53, 23.23, 8.86, "base paper Table 2 (claimed, no code)"),
}


def ms(values):
    return st.mean(values), (st.stdev(values) if len(values) > 1 else 0.0)


def main():
    runs = {}
    for f in sorted(RES.glob("*.json")):
        if f.name.endswith(("_cfg.json", "_test_metrics.json")) or f.name.startswith(("_", "summary")):
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("status") != "done" or "point_metrics" not in d:
            continue
        runs.setdefault((d["model"], d["dataset"]), []).append(d)
    if not runs:
        sys.exit("No finished (status == done) results found in results/")

    rows = []
    for (model, dataset), ds in sorted(runs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        ds.sort(key=lambda d: d["seed"])
        pm = [d["point_metrics"] for d in ds]
        mae, rmse, mape = (ms([p[k] * (100 if k == "mape" else 1) for p in pm]) for k in ("mae", "rmse", "mape"))
        rows.append({
            "model": model, "dataset": dataset, "n_seeds": len(ds), "seeds": [d["seed"] for d in ds],
            "mae_mean": mae[0], "mae_std": mae[1], "rmse_mean": rmse[0], "rmse_std": rmse[1],
            "mape_pct_mean": mape[0], "mape_pct_std": mape[1],
            "epochs_trained_mean": st.mean([d.get("epochs_trained") or 0 for d in ds]),
            "train_minutes_mean": st.mean([(d.get("train_time_s") or 0) / 60 for d in ds]),
            "num_parameters": ds[0].get("num_parameters"),
        })

    uq_rows = []
    for (model, dataset), ds in sorted(runs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        with_uq = [d for d in ds if (d.get("uq_metrics") or {}).get("picp") is not None]
        if not with_uq:
            continue
        u = lambda key: ms([d["uq_metrics"][key] for d in with_uq])
        row = {"model": model, "dataset": dataset, "n_seeds": len(with_uq)}
        for key in ("picp_90", "picp_95", "mpiw_95", "nll", "crps", "ece", "k95_to_nominal", "mpiw_95_scaled_to_nominal"):
            if all(key in d["uq_metrics"] for d in with_uq):
                row[key + "_mean"], row[key + "_std"] = u(key)
        if all("epistemic_share" in d["uq_metrics"] for d in with_uq):
            row["epistemic_share_mean"], row["epistemic_share_std"] = u("epistemic_share")
        uq_rows.append(row)

    lines = ["| Dataset | Model | seeds | MAE | RMSE | MAPE (%) | params | min/run |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['model']} | {r['n_seeds']} | {r['mae_mean']:.2f} ± {r['mae_std']:.2f} | "
                     f"{r['rmse_mean']:.2f} ± {r['rmse_std']:.2f} | {r['mape_pct_mean']:.2f} ± {r['mape_pct_std']:.2f} | "
                     f"{r['num_parameters']} | {r['train_minutes_mean']:.1f} |")
    lines += ["", "Reproduced under the BasicTS protocol (6:2:2 split, zeros masked, 12 -> 12 steps, original scale); mean ± sample std over seeds.",
              "", "Paper-reported (NOT reproduced), base paper Table 2:", "",
              "| Dataset | Model | MAE | RMSE | MAPE (%) | note |", "|---|---|---|---|---|---|"]
    for (model, dataset), (a, b, c, note) in sorted(REFERENCE.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        lines.append(f"| {dataset} | {model} | {a} | {b} | {c} | {note} |")

    if uq_rows:
        lines += ["", "Uncertainty (original unit; nominal PICP90 = 0.90, PICP95 = 0.95; mean ± sample std over seeds):", "",
                  "| Dataset | Model | seeds | PICP90 | PICP95 | MPIW95 | NLL | CRPS | ECE | epistemic share |", "|---|---|---|---|---|---|---|---|---|---|"]
        f = lambda r, k, nd=3: (f"{r[k + '_mean']:.{nd}f} ± {r[k + '_std']:.{nd}f}" if k + "_mean" in r else "-")
        for r in uq_rows:
            share = f"{100 * r['epistemic_share_mean']:.1f} %" if "epistemic_share_mean" in r else "-"
            lines.append(f"| {r['dataset']} | {r['model']} | {r['n_seeds']} | {f(r, 'picp_90')} | {f(r, 'picp_95')} | "
                         f"{f(r, 'mpiw_95', 1)} | {f(r, 'nll')} | {f(r, 'crps')} | {f(r, 'ece')} | {share} |")

    # claimed DGCRAN improvement over our best reproduced baseline (MAE)
    lines += ["", "Claimed DGCRAN MAE improvement over the best REPRODUCED baseline (single claim, unverified):", ""]
    for ds_name in sorted({r["dataset"] for r in rows}):
        ref = REFERENCE.get(("DGCRAN", ds_name))
        cand = [r for r in rows if r["dataset"] == ds_name]
        if ref and cand:
            best = min(cand, key=lambda r: r["mae_mean"])
            gain = (best["mae_mean"] - ref[0]) / best["mae_mean"] * 100
            lines.append(f"- {ds_name}: best baseline {best['model']} {best['mae_mean']:.2f} -> DGCRAN claim {ref[0]} ({gain:.1f}% lower MAE)")

    text = "\n".join(lines)
    (RES / "summary.md").write_text(text + "\n", encoding="utf-8")
    (RES / "summary.json").write_text(json.dumps({"point": rows, "uncertainty": uq_rows}, indent=2), encoding="utf-8")
    print(text)
    print("\nWrote results/summary.md and results/summary.json")


if __name__ == "__main__":
    main()
