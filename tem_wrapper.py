from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_LOCAL = Path(__file__).resolve().parent
_PARENT = _LOCAL.parent

# Local config first
sys.path.insert(0, str(_LOCAL))
from config import (
    TX_RADIUS, TX_TURNS, TX_HEIGHT,
    RX_RADIUS, RX_TURNS, RX_HEIGHT,
    OFFSET, CURRENT_SCALE,
)

# Parent for forwardprocess
sys.path.insert(0, str(_PARENT))

_fortran_dir = Path("/Users/xiechushu/project/py_tem_fdem/思源湖数据/思源湖现场反演/思源湖反演")
if _fortran_dir.exists():
    sys.path.insert(0, str(_fortran_dir))

from forwardprocess import forward_Fortran_TEM_log_time_aligned

from tem_forward_1d import forwardprocess, get_hankel_filter, get_frt_filter

# Load data_conf once at module level
_conf = json.loads((_LOCAL / "data_conf.json").read_text(encoding="utf-8"))
_WAVE_TIMES = np.array(_conf["wave_start_time"], dtype=float)
_WAVE_AMPS = np.array(_conf["wave_amp"], dtype=float)
_GATED_TIME = np.array(_conf["gated_time"], dtype=float)
_GATED_TIME_ABS = np.array(_conf["gated_time_abs"], dtype=float)

# Pre-compute waveform p1-p4 (ns_id=1 format) and filters for numba forward
_p1 = _WAVE_TIMES[:1].repeat(2).copy(); _p1[1] = _WAVE_AMPS[0]
_p2 = _WAVE_TIMES[1:2].repeat(2).copy(); _p2[1] = _WAVE_AMPS[1]
_p3 = _WAVE_TIMES[2:3].repeat(2).copy(); _p3[1] = _WAVE_AMPS[2]
_p4 = _WAVE_TIMES[3:4].repeat(2).copy(); _p4[1] = _WAVE_AMPS[3]
_HANKEL_FILT = get_hankel_filter()
_FRT_FILT = get_frt_filter()

def tem_forward(rho: np.ndarray, thickness: np.ndarray) -> np.ndarray:
    """TEM forward response using Fortran engine.

    Args:
        rho:       resistivity array, shape (n_layers,), unit: Ω·m
        thickness: layer thickness, shape (n_layers-1,), unit: m

    Returns:
        response array, shape (n_gates,), unit: V
    """
    nlayer = rho.size

    hh = np.zeros(nlayer, dtype=float)
    if thickness.size > 0:
        hh[: thickness.size] = thickness
    hh[-1] = 10.0

    nt = _GATED_TIME.size

    time_arr, resp_arr = forward_Fortran_TEM_log_time_aligned(
        log_time_sample=_GATED_TIME_ABS,
        # 为什么把log_time_sample替换为_GATED_TIME_ABS和_GATED_TIME对正演效果没有任何影响？可以检查/Users/xiechushu/project/fortran_forward_log_sample_aligned/tem_forward_log_sample_aligned.f90
        rho=rho,
        hh=hh,
        nlayer=nlayer,
        nt=nt,
        npls=1,
        t_st=float(_GATED_TIME[0]),
        t_ed=float(_GATED_TIME[-1]),
        xr=OFFSET,
        hr=RX_HEIGHT,
        ht=TX_HEIGHT,
        rt=TX_RADIUS,
        rr=RX_RADIUS,
        nturn=TX_TURNS,
        nturn1=RX_TURNS,
        ic=3,
        wave_start_time=_WAVE_TIMES,
        wave_amp=_WAVE_AMPS,
    )
    return resp_arr


def tem_forward_numba(rho: np.ndarray, thickness: np.ndarray) -> np.ndarray:
    """TEM forward response using Python+numba engine (forwardprocess).

    Args:
        rho:       resistivity array, shape (n_layers,), unit: Ω·m
        thickness: layer thickness, shape (n_layers-1,), unit: m

    Returns:
        response array, shape (n_gates,), unit: V
    """
    nlayer = rho.size
    hh = np.zeros(nlayer, dtype=float)
    if thickness.size > 0:
        hh[: thickness.size] = thickness
    hh[-1] = 10.0

    return forwardprocess(
        tlog_a=_GATED_TIME,
        rho=np.asarray(rho, dtype=np.float64),
        hh=np.asarray(hh, dtype=np.float64),
        ns_id=1,
        p1=_p1, p2=_p2, p3=_p3, p4=_p4,
        ht=TX_HEIGHT,
        t_ed=float(_GATED_TIME[-1]),
        xr=OFFSET, hr=RX_HEIGHT, rt=TX_RADIUS, rr=RX_RADIUS,
        nturn=TX_TURNS, nturn1=RX_TURNS, ic=3,
        hankel_filt=_HANKEL_FILT, frt_filt=_FRT_FILT,
    )


if __name__ == "__main__":
    # 对比tem_forward_numba和tem_forward
    rho = np.array([100.0])
    thickness = np.empty(0)
    resp = tem_forward(rho, thickness)
    print(f"Half-space 100 Ω·m: {len(resp)} points")
    print(f"  range: [{resp.min():.4e}, {resp.max():.4e}]")
