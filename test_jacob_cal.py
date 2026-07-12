"""Taylor expansion test for jacob_cal.py: visualise & validate log-space Jacobian."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

_LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(_LOCAL))

from jacob_cal import jacobian
from tem_wrapper import tem_forward
from config import JACOBIAN_STEP

FIG_DIR = _LOCAL / "fig"
FIG_DIR.mkdir(exist_ok=True)

_conf = json.loads((_LOCAL / "data_conf.json").read_text(encoding="utf-8"))
T_GATE = np.array(_conf["gated_time"])

_cjk_font = None
for name in ["PingFang SC", "Heiti SC", "STHeiti", "SimHei", "Arial Unicode MS"]:
    for f in fm.fontManager.ttflist:
        if f.name == name: _cjk_font = f; break
    if _cjk_font: break
if _cjk_font: plt.rcParams["font.family"] = _cjk_font.name
plt.rcParams["axes.unicode_minus"] = False


def main():
    # ---- Model ----
    nlayer = 6
    thickness = np.array([1.5, 1.5, 2.0, 2.0, 3.0])
    np.random.seed(1)
    log_rho = np.full(nlayer, np.log10(50.0)) + 0.02 * np.random.randn(nlayer)
    rho = 10.0 ** log_rho
    nt = len(T_GATE)

    print(f"Model: {nlayer} layers, rho = {[f'{v:.1f}' for v in rho]}")
    print(f"JACOBIAN_STEP = {JACOBIAN_STEP}")
    print()

    # ---- Compute Jacobian ----
    J = jacobian(rho, thickness)
    f0 = tem_forward(rho, thickness)
    log_f0 = np.log10(np.maximum(np.abs(f0), 1e-30))
    print(f"J shape: {J.shape}")
    sv = np.linalg.svd(J, compute_uv=False)
    print(f"J singular values: {[f'{v:.3e}' for v in sv]}")
    print()

    # ---- Taylor test: random direction ----
    np.random.seed(42)
    dm = np.random.randn(nlayer) * 0.02
    epsilons = np.logspace(-3, 1, 10)

    print(f"{'eps':>8s}  {'||Δf_log||':>12s}  {'||J·dm||':>12s}  {'rel_err':>8s}")
    print(f"{'-'*46}")
    rel_errs = []
    for eps in epsilons:
        delta = eps * dm
        m_pert = log_rho + delta
        f_pert = tem_forward(10.0 ** m_pert, thickness)
        dF = np.log10(np.maximum(np.abs(f_pert), 1e-30)) - log_f0
        pred = J @ delta
        rel = np.linalg.norm(dF - pred) / max(np.linalg.norm(dF), 1e-30)
        rel_errs.append(rel)
        print(f"  {eps:8.1e}  {np.linalg.norm(dF):12.4e}  {np.linalg.norm(pred):12.4e}  {rel:8.4f}")

    # ---- Plot 1: Column norms ----
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=120)
    col_norms = np.linalg.norm(J, axis=0)
    ax.bar(range(nlayer), col_norms, color="steelblue")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Column norm")
    ax.set_title("Jacobian column norms (log-space central diff)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "jacob_col_norms.png")
    plt.close(fig)

    # ---- Plot 2: Singular values ----
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=120)
    ax.semilogy(range(1, nlayer+1), sv, "o-", markersize=6, color="steelblue")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.set_title("Jacobian singular value spectrum")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "jacob_singular_values.png")
    plt.close(fig)

    # ---- Plot 3: Taylor residual vs eps ----
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.loglog(epsilons, rel_errs, "o-", markersize=6, color="steelblue")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, label="J·dm = 0")
    ax.set_xlabel("ε (perturbation scale)")
    ax.set_ylabel("Relative error")
    ax.set_title("Taylor test: ∥Δf − J·Δm∥ / ∥Δf∥")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "jacob_taylor_test.png")
    plt.close(fig)

    # ---- Plot 4: Predicted vs Actual (eps=1.0) ----
    delta = 1.0 * dm
    m_pert = log_rho + delta
    f_pert = tem_forward(10.0 ** m_pert, thickness)
    dF = np.log10(np.maximum(np.abs(f_pert), 1e-30)) - log_f0
    pred = J @ delta

    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=120)
    ax.scatter(dF, pred, s=30, color="steelblue", alpha=0.7)
    lim = max(np.abs(dF).max(), np.abs(pred).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8)
    ax.set_xlabel("Actual dlog10(f)")
    ax.set_ylabel("Predicted J*dm")
    ax.set_title(f"Predicted vs Actual (eps=1.0), rel err={rel_errs[-1]:.3f}")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "jacob_pred_vs_actual.png")
    plt.close(fig)

    # ---- Print summary ----
    print(f"\nPlots saved to {FIG_DIR}/")
    for p in ["jacob_col_norms.png", "jacob_singular_values.png",
              "jacob_taylor_test.png", "jacob_pred_vs_actual.png"]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
