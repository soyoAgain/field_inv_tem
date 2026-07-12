"""Log-spaced time gating: interpolate rx voltage at log-spaced times,
   plot semilogy, save gated times to data_conf.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

_LOCAL = Path(__file__).resolve().parent
sys_path = __import__("sys").path
sys_path.insert(0, str(_LOCAL.parent))
sys_path.insert(0, str(_LOCAL))

from data_loader import discover_sy6_records, DEFAULT_SY6_DIR

_cjk_font = None
for name in ["PingFang SC", "Heiti SC", "STHeiti", "SimHei", "Arial Unicode MS"]:
    for f in fm.fontManager.ttflist:
        if f.name == name:
            _cjk_font = f; break
    if _cjk_font: break
if _cjk_font:
    plt.rcParams["font.family"] = _cjk_font.name
plt.rcParams["axes.unicode_minus"] = False

# ---- Config ----
from config import (
    SOURCE_DATA_PATH, DATA_FILE_STEM,
    TIME_GATE_START, TIME_GATE_END, TIME_GATE_COUNT,
    USE_DENOISED, CURRENT_SCALE,
)
FIG_DIR = _LOCAL / "fig"
FIG_DIR.mkdir(exist_ok=True)
CONF_PATH = _LOCAL / "data_conf.json"
DATA_DIR = Path(SOURCE_DATA_PATH)

# ---- Load waveform data ----
with open(CONF_PATH, encoding="utf-8") as fh:
    conf = json.load(fh)
t_origin = conf["wave_start_time_real_axis"][0]  # pulse start, matches forward model origin

# ---- Load raw data ----
if USE_DENOISED:
    from data_loader import discover_sy6_records, DEFAULT_SY6_DIR
    from pathlib import Path as _Path
    shot_name = _Path(SOURCE_DATA_PATH).name
    records = discover_sy6_records(DEFAULT_SY6_DIR)
    rec = [r for r in records if r.point == shot_name and r.denoised_decay_path][0]
    loaded = rec.load(current_scale=CURRENT_SCALE)
    t_full = loaded.time  # full time axis (0-250ms)
    if loaded.denoised_full is not None:
        rx_full = loaded.denoised_full
    else:
        raise SystemExit("未找到已降噪信号，程序已退出，禁止对原始信号抽道。")
else:
    t_full = np.load(DATA_DIR / f"{DATA_FILE_STEM}_t.npy")
    rx_full = np.load(DATA_DIR / f"{DATA_FILE_STEM}_rx.npy")

# ---- Log-spaced time gates (absolute acquisition time) ----
t_gate_rel = np.logspace(np.log10(TIME_GATE_START), np.log10(TIME_GATE_END), TIME_GATE_COUNT)
t_gate_abs = t_origin + t_gate_rel

# ---- Extract rx at nearest unique gate times ----
indices = []
last_index = -1
gate_count = t_gate_abs.size
sample_count = t_full.size
for i, t_target in enumerate(t_gate_abs):
    min_index = last_index + 1
    max_index = sample_count - (gate_count - i)
    if max_index < min_index:
        max_index = min_index

    pos = int(np.searchsorted(t_full, t_target))
    candidate = min(max(pos, min_index), max_index)
    if candidate > min_index and abs(t_full[candidate - 1] - t_target) <= abs(t_full[candidate] - t_target):
        candidate -= 1
    if candidate < max_index and abs(t_full[candidate + 1] - t_target) < abs(t_full[candidate] - t_target):
        candidate += 1

    indices.append(candidate)
    last_index = candidate

indices = np.array(indices, dtype=int)
rx_gate = rx_full[indices]
t_gate_abs = t_full[indices]
t_gate_rel = t_gate_abs - t_origin

# ---- Plot ----
fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
t_full_ms = t_full * 1e3
t_off = t_full - t_origin
mask = (t_off >= TIME_GATE_START * 0.8) & (t_off <= TIME_GATE_END * 1.2)
ax.semilogy(t_off[mask] * 1e3, np.abs(rx_full[mask]), linewidth=0.5, color="lightgray", alpha=0.6, label="Raw |rx|")
ax.semilogy(t_gate_rel * 1e3, np.abs(rx_gate), "o-", markersize=5, linewidth=1.5, color="steelblue", label=f"Gated ({TIME_GATE_COUNT} pts)")
ax.set_xlabel("Time after pulse start (ms)")
ax.set_ylabel("|Rx voltage| (V)")
ax.set_title("Log-spaced time gating")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
path = FIG_DIR / "gated_data.png"
fig.savefig(path)
plt.close(fig)
print(f"Plot saved to {path}")

# ---- Save to data_conf.json ----
conf["time_origin"] = t_origin
conf["gated_time"] = t_gate_rel.tolist()
conf["gated_time_abs"] = t_gate_abs.tolist()
conf["gated_rx"] = rx_gate.tolist()
conf["time_gate_start"] = TIME_GATE_START
conf["time_gate_end"] = TIME_GATE_END
conf["time_gate_count"] = TIME_GATE_COUNT

with open(CONF_PATH, "w", encoding="utf-8") as fh:
    json.dump(conf, fh, ensure_ascii=False, indent=2)
print(f"Gated data saved to {CONF_PATH}")
print(f"  time origin (pulse start): {t_origin*1e3:.4f} ms")
print(f"  gates: {TIME_GATE_COUNT} points, {t_gate_rel[0]*1e3:.2f} – {t_gate_rel[-1]*1e3:.2f} ms (rel)")
print(f"  gated_rx range: {rx_gate.min():.4f} – {rx_gate.max():.4f} V")
