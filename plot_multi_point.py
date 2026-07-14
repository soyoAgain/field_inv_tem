"""Plot multi-point inversion results using per-point selected iterations."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import N_LAYERS, LAYER_THICKNESS

RESULTS_DIR = Path("/Users/xiechushu/project/EM_app/TEM_field_forward/point11/results_MGS")
OUT_DIR = RESULTS_DIR
POINT_PARAMS_PATH = Path("/Users/xiechushu/project/EM_app/TEM_field_forward/point11/results_MGS/point_params_MGS.json")

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


def _load_point(point_dir: Path, point_params: dict[str, dict]):
    """Load rho and depth for the configured iteration of one point."""
    point_name = point_dir.name
    if point_name not in point_params:
        raise KeyError(f"{point_name} 未在 point_params.json 中配置")

    point_conf = point_params[point_name]
    iteration_number = point_conf.get("ITERATION_NUMBER")
    if iteration_number is None:
        raise KeyError(f"{point_name} 缺少 ITERATION_NUMBER 配置")
    if not isinstance(iteration_number, int) or iteration_number <= 0:
        raise ValueError(f"{point_name} 的 ITERATION_NUMBER 必须是正整数，当前为 {iteration_number!r}")

    rho_scale = point_conf.get("RHO_SCALE", 1.0)
    if not isinstance(rho_scale, (int, float)) or rho_scale <= 0:
        raise ValueError(f"{point_name} 的 RHO_SCALE 必须是正数，当前为 {rho_scale!r}")

    res_path = point_dir / "inversion_result.json"
    conf_path = point_dir / "data_conf.json"
    if not res_path.exists():
        return None
    with open(res_path) as f:
        result = json.load(f)
    rho_hist = result.get("rho_history", [])
    if iteration_number > len(rho_hist):
        raise ValueError(
            f"{point_name} 请求第 {iteration_number} 次迭代，但只存在 {len(rho_hist)} 次迭代结果"
        )
    iter_idx = iteration_number - 1
    rho = np.array(rho_hist[iter_idx])
    if conf_path.exists():
        with open(conf_path) as f:
            conf = json.load(f)
        thick = conf.get("LAYER_THICKNESS", 0.5)
    else:
        thick = 0.5
    nlayer = len(rho)
    depths = np.cumsum(np.concatenate(([0.0], np.full(nlayer - 1, thick))))
    return {
        "point": point_name,
        "rho": rho,
        "rho_scale": float(rho_scale),
        "depth": depths,
        "rms_history": result.get("rms_history", []),
        "iteration_number": iteration_number,
    }


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


def _sort_key(name):
    digits = "".join(c for c in name if c.isdigit())
    return int(digits) if digits else 0


def main():
    point_params = json.loads(POINT_PARAMS_PATH.read_text(encoding="utf-8"))

    point_dirs = sorted(
        [d for d in RESULTS_DIR.iterdir() if d.is_dir()],
        key=lambda d: _sort_key(d.name),
    )

    all_data = []
    for pd in point_dirs:
        data = _load_point(pd, point_params)
        if data:
            print(
                f"{data['point']}: using iteration {data['iteration_number']}/"
                f"{len(data['rms_history'])}, rho_scale={data['rho_scale']:g}"
            )
            all_data.append(data)
    print(f"Loaded {len(all_data)} points")

    if not all_data:
        print("No data found!")
        return

    points = [d["point"] for d in all_data]

    # ---- Plot 1: all profiles overlaid ----
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    cmap = plt.cm.tab20
    for i, d in enumerate(all_data):
        z, r = _stair_rho(d["depth"], d["rho"] * d["rho_scale"])
        ax.semilogx(r, z, linewidth=1.8, color=cmap(i % 20), label=d["point"])
    ax.set_xlabel("Resistivity (ohm-m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Resistivity profiles — selected iteration per point")
    ax.invert_yaxis()
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "profiles_iter20.png")
    plt.close(fig)

    # ---- Plot 2: section (pcolormesh, finite layers only) ----
    n = len(all_data)
    n_finite = N_LAYERS - 1
    rho_grid = np.zeros((n, n_finite))
    for i, d in enumerate(all_data):
        rho_grid[i, :] = np.array(d["rho"][:n_finite]) * d["rho_scale"]
    depths_all = np.array(all_data[0]["depth"], dtype=float)
    z_edges = depths_all[:N_LAYERS]

    fig, ax = plt.subplots(figsize=(12 if n > 8 else 8, 5), dpi=150)
    X, Y = np.meshgrid(np.arange(n + 1) - 0.5, z_edges)
    pcm = ax.pcolormesh(X, Y, np.log10(rho_grid.T), shading="flat", cmap="Spectral_r")
    cb = fig.colorbar(pcm, ax=ax, label="log10 rho (ohm-m)")
    ax.set_xticks(range(n))
    ax.set_xticklabels(points, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Depth (m)")
    ax.set_title("Resistivity section — selected iteration per point")
    ax.set_ylim(z_edges[-1], 0.0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "section_iter20.png")
    plt.close(fig)

    # ---- Plot 3: RMS convergence ----
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for i, d in enumerate(all_data):
        rms = d.get("rms_history", [])
        if rms:
            ax.semilogy(range(len(rms)), rms, linewidth=1, color=cmap(i % 20), label=d["point"], alpha=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMS")
    ax.set_title("RMS convergence — all points")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rms_all_points.png")
    plt.close(fig)

    for p in ["profiles_iter20.png", "section_iter20.png", "rms_all_points.png"]:
        print(f"  Saved {OUT_DIR / p}")


if __name__ == "__main__":
    main()
