"""Run one BasicTS 1.0 forecasting job with STID (pipeline check).

Run it with the Python 3.11 environment that has BasicTS' requirements, e.g. in Colab:
  /content/venv/bin/python code/scripts/run_stid.py --dataset PEMS04 --epochs 2 --gpus none

--gpus none  -> CPU (no GPU quota needed)
--gpus 0     -> GPU 0
"""
import argparse
import dataclasses
import inspect
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # repo root (proposal/)
BT = ROOT / "code" / "BasicTS"               # BasicTS submodule
sys.path.insert(0, str(BT / "src"))
os.chdir(BT)                                 # BasicTS looks for datasets/<name> and writes checkpoints/ here


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="PEMS04")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--gpus", default="0", help='"0" for GPU 0, "none" for CPU')
    ap.add_argument("--input-len", type=int, default=12)
    ap.add_argument("--output-len", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    import numpy as np
    import torch
    print("python", sys.version.split()[0], "| torch", torch.__version__,
          "| numpy", np.__version__, "| cuda", torch.cuda.is_available())

    import json
    meta_path = BT / "datasets" / a.dataset / "meta.json"
    print("meta.json:", meta_path.read_text())
    rs = json.loads(meta_path.read_text()).get("regular_settings", {})
    norm_each_channel = rs.get("norm_each_channel", False)
    rescale = rs.get("rescale", True)
    null_val = rs.get("null_val", 0.0)
    print(f"Using meta.json regular settings: norm_each_channel={norm_each_channel} "
          f"rescale={rescale} null_val={null_val} seed={a.seed}")
    arr = np.load(BT / "datasets" / a.dataset / "train_data.npy", mmap_mode="r")
    print("train_data shape:", arr.shape)

    from basicts.configs import BasicTSForecastingConfig
    from basicts.launcher import BasicTSLauncher
    from basicts.models.STID import STID, STIDConfig

    print("--- BasicTSForecastingConfig fields ---")
    try:
        for f in dataclasses.fields(BasicTSForecastingConfig):
            if f.default is not dataclasses.MISSING:
                d = f.default
            elif f.default_factory is not dataclasses.MISSING:
                d = "<factory>"
            else:
                d = "<required>"
            print(f.name, "=", d)
    except TypeError:
        print(inspect.signature(BasicTSForecastingConfig.__init__))
    print("--- end of fields ---", flush=True)

    gpus = None if a.gpus.lower() == "none" else a.gpus
    model_config = STIDConfig(
        input_len=a.input_len,
        output_len=a.output_len,
        num_features=arr.shape[1],   # number of sensors (nodes)
    )
    cfg = BasicTSForecastingConfig(
        model=STID,
        dataset_name=a.dataset,
        model_config=model_config,
        gpus=gpus,
        num_epochs=a.epochs,
        input_len=a.input_len,
        output_len=a.output_len,
        lr=a.lr,
        seed=a.seed,
        norm_each_channel=norm_each_channel,   # BasicTS default is True: metrics would be on the normalized scale
        rescale=rescale,                       # BasicTS default is False: must be True to report original units
        null_val=null_val,                     # mask zero (missing) values in the metrics, as the dataset specifies
    )
    BasicTSLauncher.launch_training(cfg)


if __name__ == "__main__":
    main()
