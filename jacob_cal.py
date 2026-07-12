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
from tem_wrapper import tem_forward


def jacobian(rho: np.ndarray, thickness: np.ndarray) -> np.ndarray:
    """Compute log-space central-difference Jacobian.

    Args:
        rho:       resistivity, shape (n_layers,), Ω·m
        thickness: layer thickness, shape (n_layers-1,), m

    Returns:
        J: shape (n_gates, n_layers), ∂log10(response)/∂log10(rho)
    """
    nlayer = rho.size
    nt = len(tem_forward(rho, thickness))

    m_inv = np.log10(rho)
    J = np.zeros((nt, nlayer))

    for j in range(nlayer):
        m_neg = m_inv.copy()
        m_pos = m_inv.copy()
        dm = JACOBIAN_STEP * m_inv[j]
        if abs(dm) < 1e-12:
            dm = JACOBIAN_STEP * 1.0
        m_neg[j] -= dm
        m_pos[j] += dm

        f_neg = tem_forward(10.0 ** m_neg, thickness)
        f_pos = tem_forward(10.0 ** m_pos, thickness)

        J[:, j] = (np.log10(np.maximum(np.abs(f_pos), 1e-30))
                   - np.log10(np.maximum(np.abs(f_neg), 1e-30))) / (2.0 * dm)

    return J


if __name__ == "__main__":
    rho = np.array([50.0, 100.0, 10.0, 500.0])
    thickness = np.array([3.0, 5.0, 2.0])
    J = jacobian(rho, thickness)
    print(f"J shape: {J.shape}")
    print(f"J column norms: {[f'{np.linalg.norm(J[:,i]):.3e}' for i in range(J.shape[1])]}")
