"""Run one BasicTS 1.0 forecasting job (STID or AGCRN) on a PEMS dataset.

Run it with the Python 3.11 environment that has BasicTS' requirements, e.g. in Colab:
  /content/venv/bin/python code/scripts/run_forecast.py --model agcrn --dataset PEMS04 --epochs 2 --gpus none --tiny
  /content/venv/bin/python code/scripts/run_forecast.py --model agcrn_prob --dataset PEMS04 --gpus 0   # probabilistic AGCRN (NLL)
  /content/venv/bin/python code/scripts/run_forecast.py --model agcrn --dataset PEMS04 --gpus 0 --ckpt-dir /content/drive/MyDrive/proposal_ckpt

--gpus none -> CPU, --gpus 0 -> GPU 0.
--ckpt-dir  -> where checkpoints/logs go (put it on Drive so a disconnected session can resume).
--tiny      -> (agcrn only) hidden size 8 and 1 layer: a fast pipeline check, NOT a real baseline.
Hyper-parameters (checked against the official repositories):
  AGCRN_PROB = AGCRN + a log-variance head trained with the Gaussian NLL; --dropout p --mc-samples T adds MC Dropout.
  AGCRN = LeiBAI/AGCRN PEMSD4/PEMSD8 conf: embed_dim 10/2, 64 units, 2 layers, cheb_k 2, Adam lr 0.003 (no weight decay),
          100 epochs, early stopping patience 15, no gradient clipping.
  STID  = zezhishao/STID stid/PEMS04.py: 3 layers, hidden 32, node/time-of-day/day-of-week embeddings (288 / 7),
          Adam lr 0.002 weight decay 1e-4, MultiStepLR milestones [1, 50, 80] gamma 0.5, gradient clipping max_norm 5, 100 epochs.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # repo root (proposal/)
BT = ROOT / "code" / "BasicTS"               # BasicTS submodule
sys.path.insert(0, str(ROOT / "code"))       # our own models (code/models/...)
sys.path.insert(0, str(BT / "src"))
os.chdir(BT)                                 # BasicTS looks for datasets/<name>; default output dir is checkpoints/

AGCRN_EMBED_DIM = {"PEMS04": 10, "PEMS08": 2}   # official PEMSD4_AGCRN.conf / PEMSD8_AGCRN.conf


def patch_zscore_scaler():
    """Work around a BasicTS 1.0 bug (commit c2bb6e3): with norm_each_channel=False the mean/std are
    numpy scalars and `torch.Tensor(<numpy scalar>)` raises 'TypeError: new(): data must be a sequence'."""
    f = BT / "src" / "basicts" / "scaler" / "z_score_scaler.py"
    old = "torch.Tensor(mean), torch.Tensor(std)"
    new = "torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(std, dtype=torch.float32)"
    text = f.read_text()
    if old in text:
        f.write_text(text.replace(old, new))
        print("PATCH applied: z_score_scaler.py (torch.Tensor -> torch.as_tensor)")
    elif new in text:
        print("PATCH already applied: z_score_scaler.py")
    else:
        print("WARNING: expected line not found in z_score_scaler.py; BasicTS code may have changed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["stid", "agcrn", "agcrn_prob"], default="agcrn")
    ap.add_argument("--dataset", default="PEMS04")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--gpus", default="0", help='"0" for GPU 0, "none" for CPU')
    ap.add_argument("--input-len", type=int, default=12)
    ap.add_argument("--output-len", type=int, default=12)
    ap.add_argument("--lr", type=float, default=None, help="default: 0.003 (agcrn) / 0.002 (stid)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=None, help="early stopping patience; default 15 (agcrn), off (stid); 0 = off")
    ap.add_argument("--ckpt-dir", default=None, help="checkpoint root (e.g. a Drive folder)")
    ap.add_argument("--tiny", action="store_true", help="agcrn / agcrn_prob: 8 units, 1 layer (pipeline check only)")
    ap.add_argument("--dropout", type=float, default=0.0, help="agcrn / agcrn_prob: dropout between layers (0 = off)")
    ap.add_argument("--mc-samples", type=int, default=0, help="agcrn_prob: MC-Dropout passes in the final test evaluation (0 = off; needs --dropout > 0)")
    a = ap.parse_args()

    patch_zscore_scaler()   # must run before basicts is imported

    import numpy as np
    import torch
    print("python", sys.version.split()[0], "| torch", torch.__version__,
          "| numpy", np.__version__, "| cuda", torch.cuda.is_available())

    meta_path = BT / "datasets" / a.dataset / "meta.json"
    rs = json.loads(meta_path.read_text()).get("regular_settings", {})
    norm_each_channel = rs.get("norm_each_channel", False)
    rescale = rs.get("rescale", True)
    null_val = rs.get("null_val", 0.0)
    print(f"Using meta.json regular settings: norm_each_channel={norm_each_channel} "
          f"rescale={rescale} null_val={null_val} seed={a.seed}")
    arr = np.load(BT / "datasets" / a.dataset / "train_data.npy", mmap_mode="r")
    num_nodes = arr.shape[1]
    print("train_data shape:", arr.shape)

    from basicts.configs import BasicTSForecastingConfig
    from basicts.launcher import BasicTSLauncher
    from basicts.runners.callback import EarlyStopping, GradientClipping
    from torch.optim.lr_scheduler import MultiStepLR

    callbacks, scheduler, scheduler_params, extra = [], None, None, {}
    if a.model in ("agcrn", "agcrn_prob"):
        units, layers = (8, 1) if a.tiny else (64, 2)
        common = dict(input_len=a.input_len, output_len=a.output_len, num_features=num_nodes,
                      embed_dim=AGCRN_EMBED_DIM.get(a.dataset, 10), rnn_units=units, num_layers=layers,
                      cheb_k=2, dropout=a.dropout)
        if a.model == "agcrn":
            from models.agcrn import AGCRN, AGCRNConfig
            model_cls, model_config = AGCRN, AGCRNConfig(**common)
        else:
            from models.agcrn_prob import (AGCRNProb, AGCRNProbConfig, ProbForecastingTaskFlow,
                                           gaussian_nll, install_uq_hooks)
            install_uq_hooks()                           # final test evaluation also writes log_var.npy / epi_var.npy
            model_cls, model_config = AGCRNProb, AGCRNProbConfig(**common, mc_samples=a.mc_samples)
            extra = {"loss": gaussian_nll, "taskflow": ProbForecastingTaskFlow()}   # NLL loss + log-variance in original unit
        lr = a.lr if a.lr is not None else 0.003
        optimizer_params = {"lr": lr, "weight_decay": 0.0}          # official AGCRN: plain Adam
        patience = 15 if a.patience is None else a.patience
        if patience > 0:
            callbacks.append(EarlyStopping(patience=patience))
    else:
        from basicts.models.STID import STID, STIDConfig
        model_cls = STID
        model_config = STIDConfig(input_len=a.input_len, output_len=a.output_len, num_features=num_nodes,
                                  num_layers=3, if_time_in_day=True, if_day_in_week=True,
                                  num_time_in_day=288, num_day_in_week=7)
        lr = a.lr if a.lr is not None else 0.002
        optimizer_params = {"lr": lr, "weight_decay": 1e-4}
        scheduler, scheduler_params = MultiStepLR, {"milestones": [1, 50, 80], "gamma": 0.5}
        callbacks.append(GradientClipping(max_norm=5.0))
        if a.patience:
            callbacks.append(EarlyStopping(patience=a.patience))
    print("model:", a.model, "| optimizer_params:", optimizer_params, "| scheduler:", scheduler_params,
          "| callbacks:", [type(c).__name__ for c in callbacks])
    print("model_config:", dict(model_config))

    gpus = None if a.gpus.lower() == "none" else a.gpus
    kwargs = dict(
        model=model_cls,
        dataset_name=a.dataset,
        model_config=model_config,
        gpus=gpus,
        num_epochs=a.epochs,
        input_len=a.input_len,
        output_len=a.output_len,
        optimizer_params=optimizer_params,   # explicit: BasicTS default weight_decay would be 5e-4
        lr_scheduler=scheduler,
        lr_scheduler_params=scheduler_params,
        use_timestamps=True,                 # STID needs time-of-day / day-of-week; AGCRN ignores them
        seed=a.seed,
        norm_each_channel=norm_each_channel,   # BasicTS default is True -> would report normalized-scale metrics
        rescale=rescale,                       # BasicTS default is False -> must be True to report original units
        null_val=null_val,                     # mask zero (missing) values, as the dataset specifies
    )
    kwargs.update(extra)
    if callbacks:
        kwargs["callbacks"] = callbacks
    if a.ckpt_dir:
        kwargs["ckpt_save_dir"] = a.ckpt_dir
    BasicTSLauncher.launch_training(BasicTSForecastingConfig(**kwargs))


if __name__ == "__main__":
    main()
