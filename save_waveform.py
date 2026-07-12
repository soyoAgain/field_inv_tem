from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(_LOCAL))

from config import SOURCE_DATA_PATH, CURRENT_SCALE, RESULTS_DIR

sys.path.insert(0, str(_LOCAL.parent))
from data_loader import discover_sy6_records, DEFAULT_SY6_DIR

records = discover_sy6_records(DEFAULT_SY6_DIR)
shot_name = Path(SOURCE_DATA_PATH).name
rec = [r for r in records if r.point == shot_name and r.denoised_decay_path][0]
loaded = rec.load(current_scale=CURRENT_SCALE)

current_raw = np.load(rec.current_path) * CURRENT_SCALE
t_full = np.arange(0, 0.250, 4e-6)

# ---- Point 2 (peak) — max I in [200, 202] ms ----
mask2 = (t_full >= 0.200) & (t_full <= 0.202)
idx2_win = np.where(mask2)[0]
i2_rel = int(np.argmax(current_raw[idx2_win]))
i2 = idx2_win[i2_rel]

# ---- Point 1 (start) — in [200, t2], first I > baseline + 5% peak ----
mask1 = (t_full >= 0.200) & (t_full <= t_full[i2])
idx1_win = np.where(mask1)[0]
edge_n = max(1, min(len(idx1_win) // 20, 2000))
baseline = float(np.median(current_raw[idx1_win[:edge_n]]))
thresh = baseline + 0.05 * (current_raw[i2] - baseline)
cand = np.where(current_raw[idx1_win] > thresh)[0]
i1 = idx1_win[cand[0]] if cand.size else idx1_win[0]

# ---- Point 3 (off-start) — in [203, 204] ms, argmax of |diff| ----
mask3 = (t_full >= 0.203) & (t_full <= 0.204)
idx3_win = np.where(mask3)[0]
dc = np.diff(current_raw)
dc_abs = np.abs(dc[idx3_win[:-1]])
i3_rel = int(np.argmax(dc_abs))
i3 = idx3_win[i3_rel]

# ---- Point 4 (off-end) — after i3, first I < 0 ----
after3 = np.where(t_full > t_full[i3])[0]
neg = np.where(current_raw[after3] < 0)[0]
i4 = after3[neg[0]] if neg.size else len(current_raw) - 1

# Build keypoints
key_times = np.array([t_full[i1], t_full[i2], t_full[i3], t_full[i4]], dtype=float)
key_amps = np.array([current_raw[i1], current_raw[i2], current_raw[i3], current_raw[i4]], dtype=float)
waveform_time_origin = float(key_times[0])

out = RESULTS_DIR / "data_conf.json"
if out.exists():
    data = json.loads(out.read_text(encoding="utf-8"))
else:
    data = {}
data.update({
    "wave_start_time": (key_times - waveform_time_origin).tolist(),
    "wave_amp": key_amps.tolist(),
    "wave_start_time_real_axis": key_times.tolist(),
    "waveform_time_origin": waveform_time_origin,
})
out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Saved waveform to {out}")
print(f"  waveform_time_origin: {waveform_time_origin*1e3:.4f} ms")
print(f"  wave_start_time: {[f'{v*1e3:.4f}' for v in (key_times - waveform_time_origin)]} ms")
print(f"  wave_start_time_real_axis: {[f'{v*1e3:.4f}' for v in key_times]} ms")
print(f"  wave_amp: {[f'{v:.4f}' for v in key_amps]} A")
