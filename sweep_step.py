"""Sweep JACOBIAN_STEP and plot Taylor relative error.
dm_j = step (absolute perturbation, per user spec).
f_pert computed once outside loop with fixed perturbation delta."""
import json, sys, numpy as np
sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tem_wrapper import tem_forward

thk = np.full(19, 0.5)
np.random.seed(1)
log_rho = np.full(20, np.log10(50.0)) + 0.02 * np.random.randn(20)
rho = 10.0 ** log_rho
f0 = tem_forward(rho, thk)
log_f0 = np.log10(np.maximum(np.abs(f0), 1e-30))

steps = np.logspace(-7, 1, 30)

# Fixed Taylor perturbation — spike direction (layer 5 only)
eps = 0.01
delta = np.zeros(20)
delta[5] = eps
f_pert = tem_forward(10.0 ** (log_rho + delta), thk)
dF = np.log10(np.maximum(np.abs(f_pert), 1e-30)) - log_f0
norm_dF = np.linalg.norm(dF)
# 扩展df和norm_dF为矢量，给出代码

rel_errs = []
for step in steps:
    # Jacobian: central difference with dm_j = step (absolute)
    J = np.zeros((len(f0), 20))
    for j in range(20):
        m_neg, m_pos = log_rho.copy(), log_rho.copy()
        dm_j = step
        m_neg[j] -= dm_j
        m_pos[j] += dm_j
        fn = tem_forward(10.0 ** m_neg, thk)
        fp = tem_forward(10.0 ** m_pos, thk)
        J[:, j] = (np.log10(np.maximum(np.abs(fp), 1e-30))
                   - np.log10(np.maximum(np.abs(fn), 1e-30))) / (2 * dm_j)
    pred = J @ delta
    rel = np.linalg.norm(dF - pred) / max(norm_dF, 1e-30)
    rel_errs.append(rel)
    if len(rel_errs) % 5 == 1:
        print(f"  {step:.1e}: rel_err={rel:.4f}")

best = int(np.argmin(rel_errs))
print(f"\nBest: {steps[best]:.2e} (rel_err={rel_errs[best]:.4f})")

fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
ax.loglog(steps, rel_errs, "o-", markersize=4, color="steelblue")
ax.axvline(steps[best], color="C3", linestyle="--", linewidth=0.8, label=f"best={steps[best]:.1e}")
ax.set_xlabel("JACOBIAN_STEP")
ax.set_ylabel("Relative error")
ax.set_title("Taylor rel_err vs JACOBIAN_STEP (eps=0.1, spike layer 5)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("fig/jacobian_step_sweep.png")
plt.close(fig)
print(f"Saved fig/jacobian_step_sweep.png")
