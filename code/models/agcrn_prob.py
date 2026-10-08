"""Probabilistic AGCRN (heteroscedastic Gaussian output) for BasicTS 1.0.

Same encoder and mean head as the deterministic AGCRN (code/models/agcrn.py); ONE extra head predicts the
log-variance of every (horizon step, node). Training minimises the Gaussian negative log-likelihood (NLL),
so the model learns the mean AND how uncertain it is (aleatoric / data uncertainty).

How it plugs into BasicTS 1.0 (all checked in the BasicTS source, commit c2bb6e3):
  * forward returns {"prediction": mean, "log_var": log_var}.
  * `gaussian_nll(prediction, targets, targets_mask, log_var)` is passed as cfg.loss; the runner fills the
    arguments by NAME from the dict returned by forward. (The model cannot compute the loss itself: in validation
    and test BasicTS hands it an EMPTY `targets` tensor.) The runner copies the loss function with
    types.FunctionType(code, globals, name) -> the function must have no default arguments and no closure.
  * Loss and training are in NORMALISED units. `ProbForecastingTaskFlow.postprocess` additionally converts the
    log-variance to the ORIGINAL unit (log var_orig = log var + 2 log std, exact for the global z-score scaler)
    and `install_logvar_saver()` makes the final test evaluation write it to test_results/log_var.npy
    (raw float32, same layout as prediction.npy), where code/scripts/export_result.py computes PICP/MPIW/NLL.
"""
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from basicts.runners.taskflow import BasicTSForecastingTaskFlow

from .agcrn import AGCRN, AGCRNConfig


@dataclass
class AGCRNProbConfig(AGCRNConfig):
    """AGCRNConfig + limits of the predicted log-variance (normalised units)."""

    min_log_var: float = field(default=-10.0, metadata={"help": "Lower clamp of the predicted log-variance."})
    max_log_var: float = field(default=5.0, metadata={"help": "Upper clamp of the predicted log-variance."})


class AGCRNProb(AGCRN):
    """AGCRN with a second regression head for the log-variance. dropout (config) is applied between the stacked
    layers (inherited) and on the last hidden state before BOTH heads; it is 0 (off) unless set."""

    def __init__(self, config: AGCRNProbConfig):
        super().__init__(config)                      # builds encoder + mean head and applies the official init
        self.min_log_var = config.min_log_var
        self.max_log_var = config.max_log_var
        self.head_drop = nn.Dropout(config.dropout)
        self.var_conv = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        nn.init.xavier_uniform_(self.var_conv.weight)
        nn.init.zeros_(self.var_conv.bias)            # start with log-variance 0 (variance 1 in normalised units)

    def _head(self, conv, hidden):
        out = conv(hidden)                                                       # [B, horizon*C, N, 1]
        out = out.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        return out.permute(0, 1, 3, 2).squeeze(-1)                               # [B, horizon, N]

    def forward(self, inputs: torch.Tensor):
        """Args: inputs [batch, input_len, num_features].
        Returns: {"prediction": mean [B, horizon, N], "log_var": log-variance [B, horizon, N]} (normalised units)."""
        x = inputs.unsqueeze(-1)                                                 # [B, T, N, 1]
        init_state = x.new_zeros(self.num_layers, x.shape[0], self.num_node, self.hidden_dim)
        output = self.encoder(x, init_state, self._supports(), self.node_embeddings)   # [B, T, N, hidden]
        hidden = self.head_drop(output[:, -1:, :, :])                            # [B, 1, N, hidden]
        mean = self._head(self.end_conv, hidden)
        log_var = self._head(self.var_conv, hidden).clamp(self.min_log_var, self.max_log_var)
        return {"prediction": mean, "log_var": log_var}


def gaussian_nll(prediction, targets, targets_mask, log_var):
    """Masked Gaussian negative log-likelihood, up to the constant 0.5*log(2*pi). Normalised units.
    NOTE: BasicTS re-creates this function with types.FunctionType -> no default args, no closures, only globals."""
    mask = targets_mask.to(prediction.dtype)
    nll = 0.5 * (log_var + (targets - prediction) ** 2 * torch.exp(-log_var))
    return (nll * mask).sum() / mask.sum().clamp_min(1.0)


class ProbForecastingTaskFlow(BasicTSForecastingTaskFlow):
    """Forecasting task flow that also converts the predicted log-variance to the original unit."""

    def postprocess(self, runner, forward_return):
        log_var = forward_return.get("log_var")
        forward_return = super().postprocess(runner, forward_return)
        if log_var is not None and runner.scaler is not None and runner.cfg.rescale:
            std = torch.as_tensor(runner.scaler.stats["std"], dtype=log_var.dtype, device=log_var.device)
            forward_return["log_var_orig"] = log_var + 2.0 * torch.log(std)
        return forward_return


def install_logvar_saver():
    """Make the final test evaluation also write test_results/log_var.npy (raw float32 memmap, like prediction.npy).
    Call once, before BasicTSLauncher.launch_training (the runner runs in the same process)."""
    from basicts.runners import BasicTSRunner

    if getattr(BasicTSRunner, "_logvar_saver_installed", False):
        return
    original = BasicTSRunner._save_results

    def patched(self, batch_idx, batch_data):
        original(self, batch_idx, batch_data)
        lv = batch_data.get("log_var_orig")
        if lv is None:
            return
        lv = lv.detach().cpu().numpy().astype("float32")
        total = len(self.test_data_loader.dataset)
        path = os.path.join(self.ckpt_save_dir, "test_results", "log_var.npy")
        if batch_idx == 0:
            self._logvar_memmap = np.memmap(path, dtype="float32", mode="w+", shape=(total, *lv.shape[1:]))
        start = batch_idx * lv.shape[0]
        self._logvar_memmap[start:start + lv.shape[0]] = lv
        if start + lv.shape[0] >= total:
            self._logvar_memmap.flush()

    BasicTSRunner._save_results = patched
    BasicTSRunner._logvar_saver_installed = True
