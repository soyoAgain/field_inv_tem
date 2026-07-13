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


def jacobian_numba(rho: np.ndarray, thickness: np.ndarray,
                   parallel: bool = True) -> np.ndarray:
    """Compute log-space central-difference Jacobian using Python+numba engine.

    Args:
        rho:       resistivity, shape (n_layers,), Ω·m
        thickness: layer thickness, shape (n_layers-1,), m
        parallel:  use parallel workers (ThreadPoolExecutor) for numba engine

    Returns:
        J: shape (n_gates, n_layers), ∂log10(response)/∂log10(rho)
    """
    if not parallel:
        return _jacobian_serial(rho, thickness, tem_forward_numba)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    nlayer = rho.size
    nt = len(tem_forward_numba(rho, thickness))
    m_inv = np.log10(rho)
    dm = JACOBIAN_STEP

    def _worker(j):
        m_neg, m_pos = m_inv.copy(), m_inv.copy()
        m_neg[j] -= dm
        m_pos[j] += dm
        fn = tem_forward_numba(10.0 ** m_neg, thickness)
        fp = tem_forward_numba(10.0 ** m_pos, thickness)
        return j, (np.log10(np.maximum(np.abs(fp), 1e-30))
                   - np.log10(np.maximum(np.abs(fn), 1e-30))) / (2.0 * dm)

    J = np.zeros((nt, nlayer))
    max_workers = min(nlayer, 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, j): j for j in range(nlayer)}
        for future in as_completed(futures):
            j, col = future.result()
            J[:, j] = col
    return J


if __name__ == "__main__":
    rho = np.array([50.0, 100.0, 10.0, 500.0, 200.0, 30.0])
    thickness = np.array([3.0, 5.0, 2.0, 4.0, 1.0])
    Jf = jacobian(rho, thickness)
    Jn = jacobian_numba(rho, thickness, parallel=False)
    print(f"J shape: {Jf.shape}")
    print(f"jacobian (Fortran) norm: {np.linalg.norm(Jf):.4e}")
    print(f"jacobian_numba norm:       {np.linalg.norm(Jn):.4e}")
    print(f"max|diff| = {np.abs(Jf - Jn).max():.2e}")
