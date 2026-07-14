"""Jacobian computation using log-space central finite difference.
Reference: Jacob2_fortran_ver_log_time_aligned in Jac_matrix.py, sign-corrected.

    dm = JACOBIAN_STEP * m[j]              (relative perturbation)
    J[:, j] = (log10(f_plus) - log10(f_minus)) / (2 * dm)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(_LOCAL))

from config import JACOBIAN_STEP
from tem_wrapper import tem_forward, tem_forward_numba


def _jacobian_serial(rho, thickness, forward_fn):
    """serial central-difference Jacobian."""
    rho = np.maximum(np.asarray(rho, dtype=float), 1e-30)
    nlayer = rho.size
    nt = len(forward_fn(rho, thickness))
    m_inv = np.log10(rho)
    J = np.zeros((nt, nlayer))
    dm = JACOBIAN_STEP
    for j in range(nlayer):
        m_neg, m_pos = m_inv.copy(), m_inv.copy()
        m_neg[j] -= dm
        m_pos[j] += dm
        fn = forward_fn(10.0 ** m_neg, thickness)
        fp = forward_fn(10.0 ** m_pos, thickness)
        J[:, j] = (np.log10(np.maximum(np.abs(fp), 1e-30))
                   - np.log10(np.maximum(np.abs(fn), 1e-30))) / (2.0 * dm)
    return J


def jacobian(rho: np.ndarray, thickness: np.ndarray) -> np.ndarray:
    """Compute log-space central-difference Jacobian using Fortran engine."""
    return _jacobian_serial(rho, thickness, tem_forward)


def _worker_column(args):
    """ProcessPool worker: compute one Jacobian column (2 forward calls)."""
    j, m_inv, dm, rho_lin, thickness = args
    m_neg, m_pos = m_inv.copy(), m_inv.copy()
    m_neg[j] -= dm
    m_pos[j] += dm
    fn = tem_forward_numba(10.0 ** m_neg, thickness)
    fp = tem_forward_numba(10.0 ** m_pos, thickness)
    return j, (np.log10(np.maximum(np.abs(fp), 1e-30))
               - np.log10(np.maximum(np.abs(fn), 1e-30))) / (2.0 * dm)


def jacobian_numba(rho: np.ndarray, thickness: np.ndarray,
                   parallel: bool = True) -> np.ndarray:
    """Compute log-space central-difference Jacobian using Python+numba engine.

    Args:
        rho:       resistivity, shape (n_layers,), Ω·m
        thickness: layer thickness, shape (n_layers-1,), m
        parallel:  use ProcessPoolExecutor for parallelism

    Returns:
        J: shape (n_gates, n_layers), ∂log10(response)/∂log10(rho)
    """
    if not parallel:
        return _jacobian_serial(rho, thickness, tem_forward_numba)

    from joblib import Parallel, delayed

    rho = np.maximum(np.asarray(rho, dtype=float), 1e-30)
    nlayer = rho.size
    m_inv = np.log10(rho)
    dm = JACOBIAN_STEP
    nt = len(tem_forward_numba(rho, thickness))

    tasks = [(j, m_inv, dm, rho, thickness) for j in range(nlayer)]
    results = Parallel(n_jobs=min(nlayer, 10))(
        delayed(_worker_column)(t) for t in tasks
    )
    J = np.zeros((nt, nlayer))
    for j, col in results:
        J[:, j] = col
    return J


if __name__ == "__main__":
    import time
    from config import N_LAYERS, LAYER_THICKNESS

    rho = np.logspace(np.log10(10), np.log10(500), N_LAYERS)
    thickness = np.full(N_LAYERS - 1, LAYER_THICKNESS)

    # Accuracy
    Jf = jacobian(rho, thickness)
    Jn = jacobian_numba(rho, thickness, parallel=False)
    print(f"{N_LAYERS} layers, J shape: {Jf.shape}")
    print(f"jacobian (Fortran) norm: {np.linalg.norm(Jf):.4e}")
    print(f"jacobian_numba norm:       {np.linalg.norm(Jn):.4e}")
    print(f"max|diff| = {np.abs(Jf - Jn).max():.2e}")

    # Speed
    n_bench = 5
    t0 = time.perf_counter()
    for _ in range(n_bench):
        jacobian(rho, thickness)
    dt_f = (time.perf_counter() - t0) / n_bench

    t0 = time.perf_counter()
    for _ in range(n_bench):
        jacobian_numba(rho, thickness, parallel=False)
    dt_n = (time.perf_counter() - t0) / n_bench

    t0 = time.perf_counter()
    for _ in range(n_bench):
        jacobian_numba(rho, thickness, parallel=True)
    dt_np = (time.perf_counter() - t0) / n_bench

    print(f"\nSpeed ({n_bench} runs avg):")
    print(f"  jacobian (Fortran):        {dt_f*1e3:.1f}ms")
    print(f"  jacobian_numba (serial):   {dt_n*1e3:.1f}ms")
    print(f"  jacobian_numba (parallel): {dt_np*1e3:.1f}ms")
