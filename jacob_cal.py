"""Jacobian computation using log-space central finite difference.
Reference: Jacob2_fortran_ver_log_time_aligned in Jac_matrix.py, sign-corrected.

    dm = JACOBIAN_STEP * m[j]              (relative perturbation)
    J[:, j] = (log10(f_plus) - log10(f_minus)) / (2 * dm)
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np

_LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(_LOCAL))

from config import JACOBIAN_STEP
from tem_wrapper import tem_forward, tem_forward_numba


def _jacobian_serial(rho, thickness, forward_fn):
    """serial central-difference Jacobian."""
    rho = np.maximum(np.asarray(rho, dtype=float), 1e-30)
    nlayer = rho.size
    if nlayer == 0:
        raise ValueError("rho must contain at least one layer")
    m_inv = np.log10(rho)
    dm = JACOBIAN_STEP
    for j in range(nlayer):
        col = _finite_difference_column(j, m_inv, dm, thickness, forward_fn)
        if j == 0:
            J = np.empty((col.size, nlayer), dtype=float)
        J[:, j] = col
    return J


def jacobian(rho: np.ndarray, thickness: np.ndarray) -> np.ndarray:
    """Compute log-space central-difference Jacobian using Fortran engine."""
    return _jacobian_serial(rho, thickness, tem_forward)


def _finite_difference_column(j, m_inv, dm, thickness, forward_fn):
    """Compute one central finite-difference Jacobian column."""
    m_neg, m_pos = m_inv.copy(), m_inv.copy()
    m_neg[j] -= dm
    m_pos[j] += dm
    fn = forward_fn(10.0 ** m_neg, thickness)
    fp = forward_fn(10.0 ** m_pos, thickness)
    return (np.log10(np.maximum(np.abs(fp), 1e-30))
            - np.log10(np.maximum(np.abs(fn), 1e-30))) / (2.0 * dm)


def _worker_columns(indices, m_inv, dm, thickness):
    """Process worker: compute a batch of columns to reduce IPC overhead."""
    return [
        (int(j), _finite_difference_column(
            int(j), m_inv, dm, thickness, tem_forward_numba
        ))
        for j in indices
    ]


def jacobian_numba(rho: np.ndarray, thickness: np.ndarray,
                   parallel: bool = True,
                   n_jobs: Optional[int] = None) -> np.ndarray:
    """Compute log-space central-difference Jacobian using Python+numba engine.

    Args:
        rho:       resistivity, shape (n_layers,), Ω·m
        thickness: layer thickness, shape (n_layers-1,), m
        parallel:  use joblib processes for parallelism
        n_jobs:    process count; defaults to available logical CPUs, capped
                   by the number of layers

    Returns:
        J: shape (n_gates, n_layers), ∂log10(response)/∂log10(rho)
    """
    if not parallel:
        return _jacobian_serial(rho, thickness, tem_forward_numba)

    from joblib import Parallel, delayed

    rho = np.maximum(np.asarray(rho, dtype=float), 1e-30)
    nlayer = rho.size
    if nlayer == 0:
        raise ValueError("rho must contain at least one layer")
    m_inv = np.log10(rho)
    dm = JACOBIAN_STEP
    workers = min(nlayer, n_jobs or (os.cpu_count() or 1))
    if workers < 1:
        raise ValueError("n_jobs must be at least 1")

    column_batches = [
        batch for batch in np.array_split(np.arange(nlayer), workers)
        if batch.size
    ]
    batches = Parallel(n_jobs=workers)(
        delayed(_worker_columns)(batch, m_inv, dm, thickness)
        for batch in column_batches
    )
    results = [item for batch in batches for item in batch]
    J = np.empty((results[0][1].size, nlayer), dtype=float)
    for j, column in results:
        J[:, j] = column
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
