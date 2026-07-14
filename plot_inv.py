"""Plot each inversion iteration: resistivity model + forward-vs-observed."""
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
sys.path.insert(0, str(_LOCAL.parent))
sys.path.insert(0, str(_LOCAL))
from config import RESULTS_DIR, N_LAYERS, LAYER_THICKNESS

FIG_DIR = RESULTS_DIR / "fig_inv"

_cjk_font = None
for name in ["PingFang SC", "Heiti SC", "STHeiti", "SimHei", "Arial Unicode MS"]:
    for f in fm.fontManager.ttflist:
        if f.name == name:
            _cjk_font = f
            break
    if _cjk_font:
        break
if _cjk_font:
    plt.rcParams["font.family"] = _cjk_font.name
plt.rcParams["axes.unicode_minus"] = False


def _stair_rho(depths, rho):
    z = np.array(depths)
    r = np.array(rho)
    z_out, r_out = [], []
    for i in range(len(depths) - 1):
        z_out.extend([z[i], z[i + 1]])
        r_out.extend([r[i], r[i]])
    z_out.append(z[-1])
    r_out.append(r[-1])
    return np.array(z_out), np.array(r_out)


def _plot_one_iteration(args):
    idx, rho_list, depths, t_gate, d_obs = args
    path = FIG_DIR / f"iter_{idx:03d}.png"
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=120,
                                  gridspec_kw={"width_ratios": [1, 2]})

        # subfig1: resistivity model
        ax = axes[0]
        z, r = _stair_rho(depths, rho_list)
        ax.semilogx(r, z, linewidth=2, color="steelblue")
        rmin = max(r[r > 0].min() / 2, 1e-6)
        rmax = r.max() * 2
        ax.set_xlim(rmin, rmax)
        ax.set_xlabel("Resistivity (ohm-m)")
        ax.set_ylabel("Depth (m)")
        ax.set_title(f"Iteration {idx}")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

        # subfig2: forward vs observed (if data available)
        ax = axes[1]
        if t_gate is not None and d_obs is not None:
            try:
                from tem_wrapper import tem_forward_numba
                fwd = tem_forward_numba(np.array(rho_list), np.diff(depths))
                cal = float(np.median(np.abs(d_obs) / np.maximum(np.abs(fwd), 1e-30)))

                t_ms = np.array(t_gate) * 1e3
                ax.loglog(t_ms, np.abs(d_obs), ".", markersize=3, color="gray", alpha=0.5, label="Observed")
                ax.loglog(t_ms, np.abs(fwd * cal), "-", linewidth=1.5, color="C3", label="Forward")
                ax.legend(fontsize=7)
            except Exception:
                ax.text(0.5, 0.5, "Forward failed", transform=ax.transAxes, ha="center")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("|Response|")
        ax.set_title(f"Iter {idx} — Observed vs Forward")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return str(path)
    except Exception:
        return None


def _plot_rms_convergence(rms_hist):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.semilogy(range(len(rms_hist)), rms_hist, "o-", markersize=3, color="steelblue")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMS")
    ax.set_title("RMS convergence")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / "rms_convergence.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    if FIG_DIR.exists():
        import shutil
        shutil.rmtree(FIG_DIR)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "inversion_result.json", encoding="utf-8") as f:
        result = json.load(f)

    rho_hist = result.get("rho_history", [])
    rms_hist = result.get("rms_history", [])
    n_iter = len(rho_hist)
    print(f"Loaded {n_iter} iterations")

    # Depths
    depths = np.cumsum(np.concatenate(([0.0], np.full(N_LAYERS - 1, LAYER_THICKNESS))))

    # Observed data (for forward comparison)
    conf = json.loads((RESULTS_DIR / "data_conf.json").read_text())
    t_gate = conf.get("gated_time", None)
    d_obs = conf.get("gated_rx", None)

    # Parallel plot each iteration using joblib
    from joblib import Parallel, delayed
    tasks = [(i, rho_hist[i], depths.tolist(), t_gate, d_obs) for i in range(n_iter)]
    results = Parallel(n_jobs=-1)(
        delayed(_plot_one_iteration)(t) for t in tasks
    )
    for i, p in enumerate(results):
        if p:
            print(f"  [{i+1}/{n_iter}] {p}")
        else:
            print(f"  [{i+1}/{n_iter}] FAIL")

    # RMS convergence plot
    p_rms = _plot_rms_convergence(rms_hist)
    print(f"  RMS: {p_rms}")


if __name__ == "__main__":
    main()
