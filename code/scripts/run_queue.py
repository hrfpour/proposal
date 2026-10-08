"""Run a queue of (model, dataset, seed) jobs one after another, with resume support.

For every job it (1) trains with run_forecast.py (checkpoints on --ckpt-root, ideally on Google Drive),
(2) exports ONE results JSON with export_result.py into --results-dir (also on Drive).
A job whose results JSON already exists is SKIPPED, so after a disconnected Colab session you just run
the same command again: finished jobs are skipped and the interrupted job continues from its last
checkpoint (BasicTS resumes automatically when the config and checkpoint folder are unchanged).

Example (Colab, with the Python 3.11 environment):
  /content/venv/bin/python code/scripts/run_queue.py --models stid agcrn --datasets PEMS04 PEMS08 --seeds 42 \
      --epochs 100 --gpus 0 --ckpt-root /content/drive/MyDrive/proposal_ckpt \
      --results-dir /content/drive/MyDrive/proposal_runs/full
Order: seed by seed, dataset by dataset, so a complete set of baselines for seed 42 finishes first.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BT = ROOT / "code" / "BasicTS"
SCRIPTS = ROOT / "code" / "scripts"
SHOW = ("Epoch ", "Result <test>", "Result <val>", "Traceback", "Error", "error", "PATCH", "model:",
        "model_config:", "Total parameters", "early stopping", "Early", "Resume", "Set ckpt save dir", "Using meta.json")


def run_streaming(cmd, log_path):
    """Run cmd, write the full output to log_path, print only the informative lines. Returns the exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for raw in p.stdout:
            for line in raw.replace("\r", "\n").splitlines():
                log.write(line + "\n")
                if any(k in line for k in SHOW):
                    print(line[:260], flush=True)
        return p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["stid", "agcrn"], choices=["stid", "agcrn", "agcrn_prob"])
    ap.add_argument("--datasets", nargs="+", default=["PEMS04", "PEMS08"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--gpus", default="0", help='"0" for GPU 0, "none" for CPU')
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--status", default="done", help="done | smoke_test")
    ap.add_argument("--tiny", action="store_true", help="agcrn / agcrn_prob: 8 units, 1 layer (pipeline check only)")
    ap.add_argument("--dry-run", action="store_true", help="only list the jobs and what would be skipped")
    a = ap.parse_args()

    results = Path(a.results_dir)
    suffix = "_smoke" if a.status == "smoke_test" else ""
    jobs = [(m, d, s) for s in a.seeds for d in a.datasets for m in a.models]
    print(f"{len(jobs)} job(s): epochs={a.epochs} gpus={a.gpus} status={a.status} tiny={a.tiny}")
    done, failed, t_all = [], [], time.time()

    for i, (m, d, s) in enumerate(jobs, 1):
        stem = f"{m.upper()}_{d}_seed{s}{suffix}"
        if (results / f"{stem}.json").exists():
            print(f"[{i}/{len(jobs)}] SKIP {stem} (results JSON already exists)")
            done.append(stem)
            continue
        print(f"[{i}/{len(jobs)}] {'WOULD RUN' if a.dry_run else 'RUN'} {stem}")
        if a.dry_run:
            continue

        ckpt = Path(a.ckpt_root) / m.upper() / d / f"seed{s}"
        cmd = [sys.executable, str(SCRIPTS / "run_forecast.py"), "--model", m, "--dataset", d,
               "--epochs", str(a.epochs), "--gpus", a.gpus, "--seed", str(s), "--ckpt-dir", str(ckpt)]
        if a.tiny and m.startswith("agcrn"):
            cmd.append("--tiny")
        t0 = time.time()
        rc = run_streaming(cmd, results / "logs" / f"{stem}.log")
        if rc != 0:
            print(f"!!! TRAINING FAILED for {stem} (exit {rc}); full log: {results / 'logs' / (stem + '.log')}")
            failed.append(stem)
            continue

        nodes = json.loads((BT / "datasets" / d / "meta.json").read_text())["num_vars"]
        exp = [sys.executable, str(SCRIPTS / "export_result.py"), "--ckpt-root", str(ckpt),
               "--model", m.upper(), "--dataset", d, "--seed", str(s), "--epochs", str(a.epochs),
               "--device", "cpu" if a.gpus.lower() == "none" else "gpu", "--num-nodes", str(nodes),
               "--status", a.status, "--out", str(results)]
        rc = run_streaming(exp, results / "logs" / f"{stem}_export.log")
        if rc != 0 or not (results / f"{stem}.json").exists():
            print(f"!!! EXPORT FAILED for {stem}; see {results / 'logs' / (stem + '_export.log')}")
            failed.append(stem)
            continue
        for big in ckpt.rglob("inputs.npy"):          # not needed any more; saves ~50 MB per run on Drive
            big.unlink()
        print(f"[{i}/{len(jobs)}] DONE {stem} in {(time.time() - t0) / 60:.1f} min")
        done.append(stem)

    print(f"\nFinished {len(done)} / {len(jobs)} in {(time.time() - t_all) / 60:.1f} min. Failed: {failed or 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
