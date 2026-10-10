"""Uncertainty metrics for Gaussian predictive distributions N(mean, exp(log_var)) (numpy only).

compute_uq(pred, target, log_var, null_val) -> dict with
  picp_90/mpiw_90/picp_95/mpiw_95 : coverage and mean width of the central 90% / 95% intervals
  nll                             : Gaussian negative log-likelihood (with the 0.5*log(2*pi) constant), mean over valid points
  crps                            : continuous ranked probability score (closed form for a Gaussian)
  ece                             : mean |empirical coverage - nominal level| over levels 0.05 ... 0.95 (lower = better calibrated)
  mean_sigma                      : mean predicted standard deviation
  k95_to_nominal                  : factor k that makes the 95% interval cover exactly 95% when sigma is multiplied by k
                                    (ORACLE diagnostic: computed on the test set itself; k > 1 = over-confident)
  mpiw_95_scaled_to_nominal       : mean 95% width after that rescaling = sharpness at equal (nominal) coverage
  calibration                     : {"levels": [...], "coverage": [...]} for the calibration plot
  horizons                        : picp_95 / mpiw_95 / nll for horizon steps 3, 6, 12 (if arrays are 3-d)
All arrays are in the ORIGINAL unit. Points with |target - null_val| <= 1e-5 (missing values) are ignored.
"""
import math
from statistics import NormalDist

import numpy as np

_ND = NormalDist()


def _erf(x):
    """Abramowitz-Stegun 7.1.26 (max abs error 1.5e-7)."""
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592
    return sign * (1.0 - poly * t * np.exp(-x * x))


def _z(level):
    return _ND.inv_cdf(0.5 + level / 2.0)


def _core(mu, y, lv):
    sigma = np.exp(0.5 * lv)
    err = y - mu
    zscore = err / sigma
    nll = float(np.mean(0.5 * (lv + zscore ** 2 + math.log(2 * math.pi))))
    cdf = 0.5 * (1.0 + _erf(zscore / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * zscore ** 2) / math.sqrt(2 * math.pi)
    crps = float(np.mean(sigma * (zscore * (2 * cdf - 1) + 2 * pdf - 1.0 / math.sqrt(math.pi))))
    out = {"nll": nll, "crps": crps, "mean_sigma": float(sigma.mean())}
    abs_err = np.abs(err)
    for level in (0.90, 0.95):
        z = _z(level)
        key = int(round(level * 100))
        out[f"picp_{key}"] = float(np.mean(abs_err <= z * sigma))
        out[f"mpiw_{key}"] = float(np.mean(2 * z * sigma))
    z95 = _z(0.95)
    k95 = float(np.quantile(abs_err / sigma, 0.95) / z95)
    out["k95_to_nominal"] = k95
    out["mpiw_95_scaled_to_nominal"] = float(np.mean(2 * z95 * k95 * sigma))
    return out, abs_err, sigma


def compute_uq(pred, target, log_var, null_val=0.0):
    pred = np.asarray(pred, dtype="float64")
    target = np.asarray(target, dtype="float64")
    log_var = np.asarray(log_var, dtype="float64")
    if not (pred.shape == target.shape == log_var.shape):
        raise ValueError(f"shape mismatch: {pred.shape} {target.shape} {log_var.shape}")
    mask = np.abs(target - null_val) > 1e-5
    out, abs_err, sigma = _core(pred[mask], target[mask], log_var[mask])

    levels = [round(0.05 * k, 2) for k in range(1, 20)]            # 0.05 ... 0.95
    coverage = [float(np.mean(abs_err <= _z(l) * sigma)) for l in levels]
    out["ece"] = float(np.mean(np.abs(np.array(coverage) - np.array(levels))))
    out["calibration"] = {"levels": levels, "coverage": coverage}

    out["horizons"] = {}
    if pred.ndim >= 3:
        for h in (3, 6, 12):
            if h <= pred.shape[1]:
                m = np.abs(target[:, h - 1] - null_val) > 1e-5
                core, _, _ = _core(pred[:, h - 1][m], target[:, h - 1][m], log_var[:, h - 1][m])
                out["horizons"][f"h{h}"] = {k: core[k] for k in ("picp_95", "mpiw_95", "nll", "crps", "k95_to_nominal")}
    out["n_valid"] = int(mask.sum())
    return out
