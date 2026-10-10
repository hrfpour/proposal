"""Probabilistic AGCRN (heteroscedastic Gaussian output, optional MC Dropout) for BasicTS 1.0.

Same encoder and mean head as the deterministic AGCRN (code/models/agcrn.py); ONE extra head predicts the
log-variance of every (horizon step, node). Training minimises the Gaussian negative log-likelihood (NLL), so the
model learns the mean AND how uncertain it is (aleatoric / data uncertainty).

Epistemic (model) uncertainty by MC Dropout (Gal & Ghahramani 2016; Kendall & Gal 2017):
  * train with config.dropout = p > 0 (dropout between the stacked layers and before both heads);
  * config.mc_samples = T > 1 makes the FINAL evaluation of the best model (BasicTSRunner.eval, run once after
    training) draw T stochastic forward passes with dropout ON. The predictive distribution is then
        mean      = mean_t(mu_t)
        variance  = mean_t(sigma_t^2)  +  var_t(mu_t)         (aleatoric + epistemic, law of total variance)
    and the model also returns the epistemic part separately. Validation during training uses ONE deterministic pass.

How it plugs into BasicTS 1.0 (all read in the BasicTS source, commit c2bb6e3):
  * forward returns {"prediction": mean, "log_var": log_var[, "epi_var": epistemic variance]}.
  * `gaussian_nll(prediction, targets, targets_mask, log_var)` is passed as cfg.loss; the runner fills the arguments
    by NAME from the dict returned by forward. (The model cannot compute the loss itself: in validation and test
    BasicTS hands it an EMPTY `targets` tensor.) The runner copies the loss function with
    types.FunctionType(code, globals, name) -> no default arguments, no closure.
  * `ProbForecastingTaskFlow.postprocess` converts the variances to the ORIGINAL unit
    (log var_orig = log var + 2 log std; var_orig = var * std^2; exact for the global z-score scaler) and
    `install_uq_hooks()` makes the final test evaluation write test_results/log_var.npy (and epi_var.npy) as raw
    float32 in the same layout as prediction.npy; code/scripts/export_result.py turns them into PICP/MPIW/NLL/...
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
    """AGCRNConfig + limits of the predicted log-variance (normalised units) + MC Dropout samples."""

    min_log_var: float = field(default=-10.0, metadata={"help": "Lower clamp of the predicted log-variance."})
    max_log_var: float = field(default=5.0, metadata={"help": "Upper clamp of the predicted log-variance."})
    mc_samples: int = field(default=0, metadata={"help": "Stochastic passes (dropout ON) in the FINAL test evaluation; "
                                                         "0 or 1 = a single deterministic pass."})


class AGCRNProb(AGCRN):
    """AGCRN with a second regression head for the log-variance (see module docstring)."""

    def __init__(self, config: AGCRNProbConfig):
        super().__init__(config)                      # builds encoder + mean head and applies the official init
        self.min_log_var = config.min_log_var
        self.max_log_var = config.max_log_var
        self.mc_final_samples = int(config.mc_samples)   # used by the final evaluation only (see install_uq_hooks)
        self.mc_eval_samples = 1                         # switched to mc_final_samples during the final evaluation
        self.head_drop = nn.Dropout(config.dropout)
        self.var_conv = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        nn.init.xavier_uniform_(self.var_conv.weight)
        nn.init.zeros_(self.var_conv.bias)            # start with log-variance 0 (variance 1 in normalised units)

    def _head(self, conv, hidden):
        out = conv(hidden)                                                       # [B, horizon*C, N, 1]
        out = out.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        return out.permute(0, 1, 3, 2).squeeze(-1)                               # [B, horizon, N]

    def _forward_once(self, inputs):
        x = inputs.unsqueeze(-1)                                                 # [B, T, N, 1]
        init_state = x.new_zeros(self.num_layers, x.shape[0], self.num_node, self.hidden_dim)
        output = self.encoder(x, init_state, self._supports(), self.node_embeddings)   # [B, T, N, hidden]
        hidden = self.head_drop(output[:, -1:, :, :])                            # [B, 1, N, hidden]
        mean = self._head(self.end_conv, hidden)
        log_var = self._head(self.var_conv, hidden).clamp(self.min_log_var, self.max_log_var)
        return mean, log_var

    def forward(self, inputs: torch.Tensor):
        """Args: inputs [batch, input_len, num_features].
        Returns {"prediction": mean, "log_var": log-variance [, "epi_var": epistemic variance]} (normalised units)."""
        if self.training or self.mc_eval_samples <= 1:
            mean, log_var = self._forward_once(inputs)
            return {"prediction": mean, "log_var": log_var}

        # MC Dropout: the model is in eval mode; switch ONLY the dropout layers on for T stochastic passes.
        drops = [m for m in self.modules() if isinstance(m, nn.Dropout)]
        for m in drops:
            m.train()
        try:
            means, variances = [], []
            for _ in range(self.mc_eval_samples):
                mu, lv = self._forward_once(inputs)
                means.append(mu)
                variances.append(torch.exp(lv))
        finally:
            for m in drops:
                m.eval()
        means = torch.stack(means, dim=0)                                        # [T, B, horizon, N]
        epi_var = means.var(dim=0, unbiased=False)                               # epistemic: spread of the means
        ale_var = torch.stack(variances, dim=0).mean(dim=0)                      # aleatoric: mean predicted variance
        total_var = (ale_var + epi_var).clamp_min(1e-12)
        return {"prediction": means.mean(dim=0), "log_var": torch.log(total_var), "epi_var": epi_var}


def gaussian_nll(prediction, targets, targets_mask, log_var):
    """Masked Gaussian negative log-likelihood, up to the constant 0.5*log(2*pi). Normalised units.
    NOTE: BasicTS re-creates this function with types.FunctionType -> no default args, no closures, only globals."""
    mask = targets_mask.to(prediction.dtype)
    nll = 0.5 * (log_var + (targets - prediction) ** 2 * torch.exp(-log_var))
    return (nll * mask).sum() / mask.sum().clamp_min(1.0)


class ProbForecastingTaskFlow(BasicTSForecastingTaskFlow):
    """Forecasting task flow that also converts the predicted variances to the original unit."""

    def postprocess(self, runner, forward_return):
        log_var = forward_return.get("log_var")
        epi_var = forward_return.get("epi_var")
        forward_return = super().postprocess(runner, forward_return)
        if log_var is not None and runner.scaler is not None and runner.cfg.rescale:
            std = torch.as_tensor(runner.scaler.stats["std"], dtype=log_var.dtype, device=log_var.device)
            forward_return["log_var_orig"] = log_var + 2.0 * torch.log(std)
            if epi_var is not None:
                forward_return["epi_var_orig"] = epi_var * std ** 2
        return forward_return


_SAVED_KEYS = {"log_var_orig": "log_var.npy", "epi_var_orig": "epi_var.npy"}


def install_uq_hooks():
    """Two small patches of BasicTSRunner (it runs in the same process as the training script). Call once, before
    BasicTSLauncher.launch_training:
      1. _save_results: the final test evaluation also writes log_var.npy (and epi_var.npy) as raw float32 memmaps,
         like prediction.npy;
      2. eval: during the final evaluation of the best model the model uses mc_final_samples MC-Dropout passes."""
    from basicts.runners import BasicTSRunner

    if getattr(BasicTSRunner, "_uq_hooks_installed", False):
        return
    original_save = BasicTSRunner._save_results
    original_eval = BasicTSRunner.eval

    def patched_save(self, batch_idx, batch_data):
        original_save(self, batch_idx, batch_data)
        total = len(self.test_data_loader.dataset)
        maps = self.__dict__.setdefault("_uq_memmaps", {})
        for key, filename in _SAVED_KEYS.items():
            value = batch_data.get(key)
            if value is None:
                continue
            value = value.detach().cpu().numpy().astype("float32")
            if batch_idx == 0:
                path = os.path.join(self.ckpt_save_dir, "test_results", filename)
                maps[key] = np.memmap(path, dtype="float32", mode="w+", shape=(total, *value.shape[1:]))
            start = batch_idx * value.shape[0]
            maps[key][start:start + value.shape[0]] = value
            if start + value.shape[0] >= total:
                maps[key].flush()

    def patched_eval(self, ckpt_path=None):
        model = getattr(self.model, "module", self.model)
        samples = getattr(model, "mc_final_samples", 1)
        if samples > 1:
            model.mc_eval_samples = samples
        try:
            return original_eval(self, ckpt_path)
        finally:
            if hasattr(model, "mc_eval_samples"):
                model.mc_eval_samples = 1

    BasicTSRunner._save_results = patched_save
    BasicTSRunner.eval = patched_eval
    BasicTSRunner._uq_hooks_installed = True
