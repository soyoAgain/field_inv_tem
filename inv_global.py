"""Global optimization inversion using differential evolution.

Reads parameters from config.py, data from data_conf.json.
Optimizes log10(rho) for each layer, minimizing log-space RMS.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

_LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(_LOCAL.parent))
sys.path.insert(0, str(_LOCAL))

from config import (
    N_LAYERS, LAYER_THICKNESS, LOG10_RHO_MIN, LOG10_RHO_MAX, TARGET_POINT,
    GLOBAL_RESULT_PATH,
)
from tem_wrapper import tem_forward

_conf = json.loads((_LOCAL / "data_conf.json").read_text(encoding="utf-8"))
_d_obs = np.array(_conf["gated_rx"], dtype=float)
_t_gate = np.array(_conf["gated_time"], dtype=float)
_thickness = np.full(N_LAYERS - 1, LAYER_THICKNESS)

# Pre-compute calibration from a half-space
_f_hs = tem_forward(np.full(N_LAYERS, 50.0), _thickness)
_log_d = np.log10(np.maximum(np.abs(_d_obs), 1e-30))
_log_f_hs = np.log10(np.maximum(np.abs(_f_hs), 1e-30))
_log_cal = float(np.median(_log_d - _log_f_hs))
_cal = float(10.0 ** _log_cal)
_sample_Vobs_log = _log_d - _log_cal
print(f"Cal: {_cal:.2e}, data range (log, cal'd): [{_sample_Vobs_log.min():.3f}, {_sample_Vobs_log.max():.3f}]")


def _objective(log_rho: np.ndarray) -> float:
    """RMS in log-space (lower is better)."""
    rho = 10.0 ** log_rho
    fwd = tem_forward(rho, _thickness)
    fwd_log = np.log10(np.maximum(np.abs(fwd), 1e-30))
    rms = float(np.sqrt(np.mean((_sample_Vobs_log - fwd_log) ** 2)))
    return rms


def _smooth_penalty(log_rho: np.ndarray, weight: float = 0.01) -> float:
    """First-difference roughness penalty."""
    diff = np.diff(log_rho)
    return float(weight * np.sum(diff ** 2))


def main():
    bounds = [(LOG10_RHO_MIN, LOG10_RHO_MAX)] * N_LAYERS

    print(f"=== Global Optimization (DE) — {TARGET_POINT} ===")
    print(f"Layers: {N_LAYERS}, thick: {LAYER_THICKNESS}m")
    print(f"Gates: {len(_d_obs)} pts, {_t_gate[0]*1e3:.2f}–{_t_gate[-1]*1e3:.2f} ms")
    print(f"Bounds: log10(rho) ∈ [{LOG10_RHO_MIN}, {LOG10_RHO_MAX}]")
    print(f"Parameters: {N_LAYERS}")
    print()

    t0 = time.perf_counter()

    result = differential_evolution(
        _objective,
        bounds,
        strategy="best1bin",
        maxiter=50,
        popsize=15,
        tol=1e-8,
        mutation=(0.5, 1.5),
        recombination=0.7,
        seed=42,
        polish=True,
    )

    dt = time.perf_counter() - t0
    best_log_rho = result.x
    best_rho = 10.0 ** best_log_rho
    best_rms = float(result.fun)

    print(f"\nOptimization finished in {dt:.0f}s ({result.nit} iters, {result.nfev} evals)")
    print(f"  RMS: {best_rms:.4e}")
    print(f"  rho: {[f'{v:.2f}' for v in best_rho[:8]]}...")

    # Save
    out = _LOCAL / GLOBAL_RESULT_PATH
    result_dict = {
        "point": TARGET_POINT,
        "depth": np.cumsum(np.concatenate(([0.0], _thickness))).tolist(),
        "rho": best_rho.tolist(),
        "calibration": float(_cal),
        "rms": best_rms,
        "optimizer": "differential_evolution",
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "thicknesses": _thickness.tolist(),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
