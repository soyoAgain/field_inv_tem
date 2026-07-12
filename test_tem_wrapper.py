"""Test tem_wrapper.py: forward responses for various subsurface models."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

_LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(_LOCAL))

from tem_wrapper import tem_forward

FIG_DIR = _LOCAL / "fig"
FIG_DIR.mkdir(exist_ok=True)

_cjk_font = None
for name in ["PingFang SC", "Heiti SC", "STHeiti", "SimHei", "Arial Unicode MS"]:
    for f in fm.fontManager.ttflist:
        if f.name == name:
            _cjk_font = f; break
    if _cjk_font: break
if _cjk_font:
    plt.rcParams["font.family"] = _cjk_font.name
plt.rcParams["axes.unicode_minus"] = False

# ---- Model definitions ----
MODELS = [
    ("Uniform 0.01 Ω·m",    np.array([0.01]),     np.array([])),
    ("Uniform 0.1 Ω·m",     np.array([0.1]),      np.array([])),
    ("Uniform 1 Ω·m",       np.array([1.0]),      np.array([])),
    ("Uniform 10 Ω·m",      np.array([10.0]),     np.array([])),
    ("Uniform 100 Ω·m",     np.array([100.0]),    np.array([])),
    ("Uniform 1000 Ω·m",    np.array([1000.0]),   np.array([])),
    ("Conductive cap 0.01/100 Ω·m (5m)", np.array([0.01, 100.0]), np.array([5.0])),
    ("Resistive cap 100/0.01 Ω·m (5m)",  np.array([100.0, 0.01]), np.array([5.0])),
    ("3-layer 100/0.01/1000 Ω·m (3m,7m)", np.array([100.0, 0.01, 1000.0]), np.array([3.0, 7.0])),
    ("3-layer 0.01/100/0.01 Ω·m (3m,7m)", np.array([0.01, 100.0, 0.01]), np.array([3.0, 7.0])),
    ("Gradient: 0.001→0.01→0.1→1→10→100→1000 (1.5m each)",
     np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]),
     np.full(6, 1.5)),
]

# ---- Load gated_time for x-axis ----
import json
conf = json.loads((_LOCAL / "data_conf.json").read_text(encoding="utf-8"))
t_gate = np.array(conf["gated_time"]) * 1e3  # ms

# ---- Compute responses ----
results = []
for label, rho, thk in MODELS:
    try:
        resp = tem_forward(rho, thk)
        results.append((label, np.abs(resp)))
        print(f"  {label:45s}  resp range: [{resp.min():.3e}, {resp.max():.3e}]")
    except Exception as e:
        print(f"  {label:45s}  ERROR: {e}")

# ---- Plot 1: all uniform half-spaces ----
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
for label, resp in results[:6]:
    ax.loglog(t_gate, resp, linewidth=1.8, label=label)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("|Response| (V)")
ax.set_title("Half-space forward responses")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
p = FIG_DIR / "test_halfspace.png"
fig.savefig(p); plt.close(fig)
print(f"\nSaved {p}")

# ---- Plot 2: layered models ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
cmap = plt.cm.tab10

# 2-layer
ax = axes[0]
for i, (label, resp) in enumerate(results[6:8]):
    ax.loglog(t_gate, resp, linewidth=1.8, color=cmap(i), label=label)
# add uniform 10 Ω·m for comparison
_, ref = results[3]
ax.loglog(t_gate, ref, "--", linewidth=1, color="gray", label="Uniform 10 Ω·m (ref)")
ax.set_xlabel("Time (ms)"); ax.set_ylabel("|Response| (V)")
ax.set_title("2-layer models (depth ≤ 5m)")
ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.3)

# 3-layer + gradient
ax = axes[1]
for i, (label, resp) in enumerate(results[8:]):
    ax.loglog(t_gate, resp, linewidth=1.8, color=cmap(i+2), label=label)
ax.set_xlabel("Time (ms)"); ax.set_ylabel("|Response| (V)")
ax.set_title("3-layer & gradient models (depth ≤ 10m)")
ax.legend(fontsize=6); ax.grid(True, which="both", alpha=0.3)

fig.tight_layout()
p = FIG_DIR / "test_layered.png"
fig.savefig(p); plt.close(fig)
print(f"Saved {p}")

# ---- Plot 3: all models on one figure ----
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
for i, (label, resp) in enumerate(results):
    ax.loglog(t_gate, resp, linewidth=1.2, color=cmap(i % 10), label=label, alpha=0.85)
ax.set_xlabel("Time (ms)"); ax.set_ylabel("|Response| (V)")
ax.set_title("All forward responses (depth ≤ 10m, ρ ∈ [1e-3, 1e3] Ω·m)")
ax.legend(fontsize=6, ncol=2)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
p = FIG_DIR / "test_all_models.png"
fig.savefig(p); plt.close(fig)
print(f"Saved {p}")
