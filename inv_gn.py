"""Gauss-Newton inversion for TEM 1D, using tem_wrapper + jacob_cal."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(_LOCAL.parent))
sys.path.insert(0, str(_LOCAL))

from config import (
    N_LAYERS, LAYER_THICKNESS, MAX_ITER,
    LAMBDA_INITIAL, LAMBDA_DECREASE, LAMBDA_INCREASE,
    LOG10_RHO_MIN, LOG10_RHO_MAX, CALIBRATION_FLAG, TARGET_POINT,
)
from tem_wrapper import tem_forward
from jacob_cal import jacobian


def _roughness_matrix(n: int) -> np.ndarray:
    W = np.zeros((n - 1, n))
    for i in range(n - 1):
        W[i, i] = -1.0
        W[i, i + 1] = 1.0
    return W


def main():
    conf = json.loads((_LOCAL / "data_conf.json").read_text(encoding="utf-8"))
    t_gate = np.array(conf["gated_time"])
    d_obs = np.array(conf["gated_rx"])

    print(f"=== Gauss-Newton — {TARGET_POINT} ===")
    print(f"Layers: {N_LAYERS}, thick: {LAYER_THICKNESS}m, max iter: {MAX_ITER}")
    print(f"Gates: {len(t_gate)} pts, {t_gate[0]*1e3:.2f}–{t_gate[-1]*1e3:.2f} ms")

    thickness = np.full(N_LAYERS - 1, LAYER_THICKNESS)
    log_rho = np.full(N_LAYERS, np.log10(50.0)) + 0.01 * np.random.randn(N_LAYERS)

    f_init = tem_forward(10.0 ** log_rho, thickness)
    if CALIBRATION_FLAG:
        log_cal = float(np.median(np.log10(np.maximum(np.abs(d_obs), 1e-30))
                                  - np.log10(np.maximum(np.abs(f_init), 1e-30))))
        cal = float(10.0 ** log_cal)
        d_work = np.log10(np.maximum(np.abs(d_obs), 1e-30)) - log_cal
    else:
        cal = 1.0
        d_work = np.log10(np.maximum(np.abs(d_obs), 1e-30))

    Wm = _roughness_matrix(N_LAYERS)
    WmTWm = Wm.T @ Wm

    lamb = LAMBDA_INITIAL
    best_rms = np.inf
    best_log_rho = log_rho.copy()
    rms_hist, rho_hist = [], []

    f0_log = np.log10(np.maximum(np.abs(f_init), 1e-30))
    rms0 = float(np.sqrt(np.mean((d_work - f0_log) ** 2)))
    rms_hist.append(rms0)
    rho_hist.append((10.0 ** log_rho).tolist())
    print(f"  Initial rms={rms0:.4e} cal={cal:.2e}")

    for it in range(MAX_ITER):
        t0 = time.perf_counter()
        J = jacobian(10.0 ** log_rho, thickness)
        f_pred = tem_forward(10.0 ** log_rho, thickness)
        f_log = np.log10(np.maximum(np.abs(f_pred), 1e-30))
        residual = d_work - f_log
        rms = float(np.sqrt(np.mean(residual ** 2)))
        if rms < best_rms:
            best_rms = rms
            best_log_rho = log_rho.copy()

        H = J.T @ J + lamb * WmTWm
        rhs = J.T @ residual - lamb * WmTWm @ log_rho
        try:
            dm = np.linalg.solve(H, rhs)
        except np.linalg.LinAlgError:
            lamb = min(1e8, lamb * LAMBDA_INCREASE)
            continue

        accepted = False
        for k in range(15):
            alpha = 1.0 / (2.0 ** k)
            trial_log = log_rho + alpha * dm
            if not np.all((trial_log >= LOG10_RHO_MIN) & (trial_log <= LOG10_RHO_MAX)):
                continue
            f_trial = tem_forward(10.0 ** trial_log, thickness)
            ft_log = np.log10(np.maximum(np.abs(f_trial), 1e-30))
            trms = float(np.sqrt(np.mean((d_work - ft_log) ** 2)))
            if trms < rms:
                log_rho = trial_log
                lamb = max(1e-8, lamb * LAMBDA_DECREASE)
                accepted = True
                rms = trms
                break
        if not accepted:
            lamb = min(1e8, lamb * LAMBDA_INCREASE)

        dt = time.perf_counter() - t0
        print(f"  iter {it+1:2d}/{MAX_ITER}  rms={rms:.4e}  lambda={lamb:.2e}  dt={dt:.1f}s  {'+' if accepted else '-'}")
        rms_hist.append(rms)
        rho_hist.append((10.0 ** log_rho).tolist())

    result = {
        "point": TARGET_POINT,
        "depth": np.cumsum(np.concatenate(([0.0], thickness))).tolist(),
        "rho": (10.0 ** best_log_rho).tolist(),
        "calibration": float(cal),
        "rms": best_rms,
        "n_iterations": MAX_ITER,
        "thicknesses": thickness.tolist(),
        "rms_history": rms_hist,
        "rho_history": rho_hist,
    }
    out = _LOCAL / "inversion_result.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out}")
    print(f"  Final RMS: {best_rms:.4e}")
    print(f"  Rho: {[f'{v:.1f}' for v in result['rho'][:6]]}...")


if __name__ == "__main__":
    main()
