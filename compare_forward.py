"""Compare Python forwardprocess (tem_forward_1d) vs Fortran tem_forward (tem_wrapper)."""
from __future__ import annotations

import json, sys, time
from pathlib import Path
import numpy as np

_LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(_LOCAL.parent))
sys.path.insert(0, str(_LOCAL))

from config import N_LAYERS, LAYER_THICKNESS, TX_RADIUS, TX_TURNS, TX_HEIGHT, RX_RADIUS, RX_TURNS, RX_HEIGHT, OFFSET
from tem_wrapper import tem_forward as _fwd_fortran
from tem_forward_1d import forwardprocess, get_hankel_filter, get_frt_filter

# ---- Load config ----
conf = json.loads((_LOCAL / "data_conf.json").read_text(encoding="utf-8"))
t_gate = np.array(conf["gated_time"], dtype=np.float64)
w_times = np.array(conf["wave_start_time"], dtype=np.float64)
w_amps = np.array(conf["wave_amp"], dtype=np.float64)
t_end = float(conf.get("time_origin", 0.2) + t_gate[-1] + 0.05)

thickness = np.full(N_LAYERS - 1, LAYER_THICKNESS)

# ---- Build waveform p1-p4 (ns_id=1 format) ----
p1 = w_times[:1].repeat(2).copy()
p1[1] = w_amps[0]
p2 = w_times[1:2].repeat(2).copy()
p2[1] = w_amps[1]
p3 = w_times[2:3].repeat(2).copy()
p3[1] = w_amps[2]
p4 = w_times[3:4].repeat(2).copy()
p4[1] = w_amps[3]

# Fortran uses rho as-is; forwardprocess expects hh length = rho.size - 1
# tem_wrapper internally converts thickness → hh

# ---- Test models ----
models = [
    ("half 10 Ω·m",    np.full(N_LAYERS, 10.0)),
    ("half 50 Ω·m",    np.full(N_LAYERS, 50.0)),
    ("half 100 Ω·m",   np.full(N_LAYERS, 100.0)),
    ("layered up",     np.logspace(np.log10(10), np.log10(1000), N_LAYERS)),
    ("layered down",   np.logspace(np.log10(1000), np.log10(10), N_LAYERS)),
]

print(f"Layers: {N_LAYERS}, thick: {LAYER_THICKNESS}m, gates: {len(t_gate)}")
print()

hankel_filt = get_hankel_filter()
frt_filt = get_frt_filter()

for name, rho in models:
    print(f"{'='*60}")
    print(f"  Model: {name}")
    print(f"  rho[:6]: {[f'{v:.1f}' for v in rho[:6]]}...")

    hh = np.concatenate([thickness, [10.0]])  # Fortran format: nlayer elements

    # --- Speed: Fortran ---
    n_warm, n_bench = 2, 10
    for _ in range(n_warm):
        _ = _fwd_fortran(rho, thickness)
    t0 = time.perf_counter()
    for _ in range(n_bench):
        resp_f = _fwd_fortran(rho, thickness)
    dt_f = (time.perf_counter() - t0) / n_bench

    # --- Speed: Python forwardprocess ---
    for _ in range(n_warm):
        _ = forwardprocess(tlog_a=t_gate, rho=rho.astype(np.float64), hh=hh.astype(np.float64),
                           ns_id=1, p1=p1, p2=p2, p3=p3, p4=p4,
                           ht=TX_HEIGHT, t_ed=t_end,
                           xr=OFFSET, hr=RX_HEIGHT, rt=TX_RADIUS, rr=RX_RADIUS,
                           nturn=TX_TURNS, nturn1=RX_TURNS, ic=3,
                           hankel_filt=hankel_filt, frt_filt=frt_filt)
    t0 = time.perf_counter()
    for _ in range(n_bench):
        resp_p = forwardprocess(tlog_a=t_gate, rho=rho.astype(np.float64), hh=hh.astype(np.float64),
                                ns_id=1, p1=p1, p2=p2, p3=p3, p4=p4,
                                ht=TX_HEIGHT, t_ed=t_end,
                                xr=OFFSET, hr=RX_HEIGHT, rt=TX_RADIUS, rr=RX_RADIUS,
                                nturn=TX_TURNS, nturn1=RX_TURNS, ic=3,
                                hankel_filt=hankel_filt, frt_filt=frt_filt)
    dt_p = (time.perf_counter() - t0) / n_bench

    resp_f = np.asarray(resp_f, dtype=float)
    resp_p = np.asarray(resp_p, dtype=float)

    print(f"  Speed:  Fortran={dt_f*1e3:.1f}ms   Python={dt_p*1e3:.1f}ms   ratio=Fortran/Python={dt_f/dt_p:.2f}x")

    # --- Numerical diff ---
    diff = resp_f - resp_p
    abs_err = float(np.abs(diff).max())
    rel = np.abs(diff) / np.maximum(np.abs(resp_f), 1e-30)
    rel_err = float(rel.max())
    median_rel = float(np.median(rel))
    corr = float(np.corrcoef(resp_f, resp_p)[0, 1])

    print(f"  |diff|_max = {abs_err:.3e}")
    print(f"  |diff|/|F|_max = {rel_err:.3e} ({rel_err*100:.4f}%)")
    print(f"  median rel err = {median_rel:.3e}")
    print(f"  correlation = {corr:.6f}")
    print()

print("Done.")
