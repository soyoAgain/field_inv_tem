from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None

HANKEL_NC = 100
HANKEL_NCNULL = 60
HANKEL_FAC = 0.1 * np.log(10.0)
PI = np.pi
TWO_PI = 2.0 * np.pi
MU0 = 4.0e-7 * np.pi

_H0_VALUES = (
    2.898782880000000e-07, 3.649351440000000e-07, 4.594261260000000e-07, 5.783832260000000e-07,
    7.281413380000000e-07, 9.166756390000000e-07, 1.154026250000000e-06, 1.452832980000000e-06,
    1.829008340000000e-06, 2.302585110000000e-06, 2.898782860000000e-06, 3.649351480000000e-06,
    4.594261190000000e-06, 5.783832360000000e-06, 7.281413220000000e-06, 9.166756640000000e-06,
    1.154026210000000e-05, 1.452833050000000e-05, 1.829008240000000e-05, 2.302585270000000e-05,
    2.898782590000000e-05, 3.649351860000000e-05, 4.594260510000000e-05, 5.783833290000000e-05,
    7.281411440000000e-05, 9.166758819999999e-05, 1.154025730000000e-04, 1.452833540000000e-04,
    1.829006940000000e-04, 2.302586300000000e-04, 2.898778910000000e-04, 3.649353620000000e-04,
    4.594249600000000e-04, 5.783834370000000e-04, 7.281377380000000e-04, 9.166748280000000e-04,
    1.154014530000000e-03, 1.452825610000000e-03, 1.828968260000000e-03, 2.302545350000000e-03,
    2.898639790000000e-03, 3.649167030000000e-03, 4.593733080000000e-03, 5.783032380000000e-03,
    7.279414970000000e-03, 9.163407050000001e-03, 1.153256910000000e-02, 1.451458320000000e-02,
    1.826011990000000e-02, 2.297010420000000e-02, 2.887026190000000e-02, 3.626918100000000e-02,
    4.547940310000000e-02, 5.694081920000000e-02, 7.098730720000000e-02, 8.809954260000000e-02,
    1.082238890000000e-01, 1.312504830000000e-01, 1.550557150000000e-01, 1.763715060000000e-01,
    1.856277380000000e-01, 1.697780440000000e-01, 1.034052450000000e-01, -3.025832330000000e-02,
    -2.275743930000000e-01, -3.621732170000000e-01, -2.055004460000000e-01, 3.373948730000000e-01,
    3.176898970000000e-01, -5.137621600000000e-01, 3.091302640000000e-01, -1.267575920000000e-01,
    4.619678900000000e-02, -1.809686740000000e-02, 8.354260500000000e-03, -4.473683040000000e-03,
    2.619747830000000e-03, -1.601713570000000e-03, 9.977178819999999e-04, -6.262758150000000e-04,
    3.943388180000000e-04, -2.486063540000000e-04, 1.568086040000000e-04, -9.892662880000001e-05,
    6.241523980000000e-05, -3.938053930000000e-05, 2.484723580000000e-05, -1.567749450000000e-05,
    9.891817410000000e-06, -6.241311600000000e-06, 3.938000580000000e-06, -2.484710180000000e-06,
    1.567746090000000e-06, -9.891808959999999e-07, 6.241309480000000e-07, -3.938000050000000e-07,
    2.484710050000000e-07, -1.567746050000000e-07, 9.891808880000000e-08, -6.241309460000000e-08,
)

_H1_VALUES = (
    1.849095570000000e-13, 2.853213270000000e-13, 4.644718080000000e-13, 7.166947710000000e-13,
    1.166700430000000e-12, 1.800255870000000e-12, 2.930618980000000e-12, 4.522038290000000e-12,
    7.361382060000000e-12, 1.135884660000000e-11, 1.849095570000000e-11, 2.853213270000000e-11,
    4.644718080000000e-11, 7.166947710000000e-11, 1.166700430000000e-10, 1.800255870000000e-10,
    2.930618980000000e-10, 4.522038290000000e-10, 7.361382060000000e-10, 1.135884660000000e-09,
    1.849095570000000e-09, 2.853213260000000e-09, 4.644718060000000e-09, 7.166947650000000e-09,
    1.166700420000000e-08, 1.800255830000000e-08, 2.930618890000000e-08, 4.522038070000000e-08,
    7.361381490000000e-08, 1.135884520000000e-07, 1.849095210000000e-07, 2.853212370000000e-07,
    4.644715800000000e-07, 7.166941980000000e-07, 1.166698990000000e-06, 1.800252260000000e-06,
    2.930609900000000e-06, 4.522015490000000e-06, 7.361324770000000e-06, 1.135870270000000e-05,
    1.849059420000000e-05, 2.853122470000000e-05, 4.644490000000000e-05, 7.166374800000000e-05,
    1.166556530000000e-04, 1.799894400000000e-04, 2.929711060000000e-04, 4.519757830000000e-04,
    7.355654350000000e-04, 1.134446150000000e-03, 1.845483060000000e-03, 2.844142570000000e-03,
    4.621947430000000e-03, 7.109805900000000e-03, 1.152369110000000e-02, 1.764344850000000e-02,
    2.840762330000000e-02, 4.297705960000000e-02, 6.803325690000001e-02, 9.978459290000000e-02,
    1.510705440000000e-01, 2.035405810000000e-01, 2.712353770000000e-01, 2.760738710000000e-01,
    2.166919770000000e-01, -7.837237370000000e-02, -3.406756270000000e-01, -3.606936730000000e-01,
    5.130245260000000e-01, -5.947247290000000e-02, -1.951171230000000e-01, 1.992356000000000e-01,
    -1.385215530000000e-01, 8.793208590000000e-02, -5.506971460000000e-02, 3.456378480000000e-02,
    -2.175271800000000e-02, 1.371002910000000e-02, -8.646564169999999e-03, 5.454627580000000e-03,
    -3.441388640000000e-03, 2.171306860000000e-03, -1.369986280000000e-03, 8.643989520000000e-04,
    -5.453978740000000e-04, 3.441225450000000e-04, -2.171265850000000e-04, 1.369975970000000e-04,
    -8.643963640000000e-05, 5.453972240000000e-05, -3.441223820000000e-05, 2.171265440000000e-05,
    -1.369975870000000e-05, 8.643963379999999e-06, -5.453972180000000e-06, 3.441223800000000e-06,
    -2.171265430000000e-06, 1.369975870000000e-06, -8.643963370000000e-07, 5.453972180000000e-07,
)


@dataclass(frozen=True)
class HankelFilter:
    nc: int
    ncnull: int
    h0: np.ndarray
    h1: np.ndarray
    h0_rev: np.ndarray
    h1_rev: np.ndarray
    log_u_base: np.ndarray


def _as_float64_array(values: tuple[float, ...]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).copy()


def build_hankel_filter() -> HankelFilter:
    h0 = _as_float64_array(_H0_VALUES)
    h1 = _as_float64_array(_H1_VALUES)
    if h0.shape != (HANKEL_NC,):
        raise ValueError(f'h0 length mismatch: expected {HANKEL_NC}, got {h0.shape}')
    if h1.shape != (HANKEL_NC,):
        raise ValueError(f'h1 length mismatch: expected {HANKEL_NC}, got {h1.shape}')

    nn = np.arange(1.0, HANKEL_NC + 1.0, dtype=np.float64)
    nnn = HANKEL_NCNULL - HANKEL_NC + nn
    log_u_base = -(nnn - 1.0) * HANKEL_FAC

    return HankelFilter(
        nc=HANKEL_NC,
        ncnull=HANKEL_NCNULL,
        h0=h0,
        h1=h1,
        h0_rev=h0[::-1].copy(),
        h1_rev=h1[::-1].copy(),
        log_u_base=log_u_base,
    )


@lru_cache(maxsize=1)
def get_hankel_filter() -> HankelFilter:
    return build_hankel_filter()


def expc(x: float | np.ndarray) -> float | np.ndarray:
    x_clip = np.clip(x, -650.0, 650.0)
    return np.exp(x_clip)


def safe_exp(x: float | np.ndarray) -> float | np.ndarray:
    return expc(x)


if njit is not None:
    @njit(cache=True)
    def _expc_numba_scalar(x: float) -> float:
        if x > 650.0:
            x = 650.0
        elif x < -650.0:
            x = -650.0
        return math.exp(x)

    @njit(cache=True)
    def _compute_hankel_u_grid_numba(r: float, log_u_base: np.ndarray) -> np.ndarray:
        out = np.empty(log_u_base.size, dtype=np.float64)
        inv_r = 1.0 / r
        for i in range(log_u_base.size):
            out[i] = _expc_numba_scalar(log_u_base[i]) * inv_r
        return out
else:
    _expc_numba_scalar = None
    _compute_hankel_u_grid_numba = None


def compute_hankel_u_grid(r: float, filt: HankelFilter | None = None) -> np.ndarray:
    if r <= 0.0:
        raise ValueError(f'r must be positive, got {r}')
    if filt is None:
        filt = get_hankel_filter()
    if _compute_hankel_u_grid_numba is not None:
        return _compute_hankel_u_grid_numba(float(r), filt.log_u_base)
    return safe_exp(filt.log_u_base) / float(r)


def _validate_layer_model(rho: np.ndarray, hh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rho_arr = np.asarray(rho, dtype=np.float64)
    hh_arr = np.asarray(hh, dtype=np.float64)
    if rho_arr.ndim != 1 or rho_arr.size == 0:
        raise ValueError('rho must be a non-empty 1D array')
    if np.any(rho_arr <= 0.0):
        raise ValueError('rho must be strictly positive')
    if hh_arr.ndim != 1:
        raise ValueError('hh must be a 1D array')
    if hh_arr.size not in (rho_arr.size - 1, rho_arr.size):
        raise ValueError('hh length must be len(rho)-1 or len(rho)')
    if hh_arr.size == rho_arr.size:
        hh_arr = hh_arr[:-1]
    if hh_arr.size and np.any(hh_arr <= 0.0):
        raise ValueError('finite layer thicknesses must be strictly positive')
    return rho_arr, hh_arr


if njit is not None:
    @njit(cache=True)
    def _b_numba(rho: np.ndarray, hh: np.ndarray, f: float, u: float) -> complex:
        omega_term = 1j * MU0 * TWO_PI * f
        bval = np.sqrt(u * u + omega_term / rho[-1])
        nlayer = rho.size
        if nlayer == 1:
            return bval
        for idx in range(nlayer - 2, -1, -1):
            alpha = np.sqrt(u * u + omega_term / rho[idx])
            h = hh[idx]
            s1 = 0.0 + 0.0j
            if (2.0 * alpha * h).real < 400.0:
                s1 = np.exp(-2.0 * alpha * h)
            s2 = (1.0 - s1) / (1.0 + s1)
            bval = alpha * (bval + alpha * s2) / (alpha + bval * s2)
        return bval

    @njit(cache=True)
    def _hankel_sum_numba(
        rho: np.ndarray,
        hh: np.ndarray,
        f: float,
        z: float,
        r: float,
        weights_rev: np.ndarray,
        log_u_base: np.ndarray,
        power_u: int,
    ) -> complex:
        total = 0.0 + 0.0j
        inv_r = 1.0 / r
        for i in range(log_u_base.size):
            u = _expc_numba_scalar(log_u_base[i]) * inv_r
            bval = _b_numba(rho, hh, f, u)
            ref = (bval - u) / (bval + u)
            atten = _expc_numba_scalar(-u * z)
            if power_u == 1:
                total += ref * atten * u * weights_rev[i]
            else:
                total += ref * atten * u * u * weights_rev[i]
        return total * inv_r
else:
    _b_numba = None
    _hankel_sum_numba = None


def b(rho: np.ndarray, hh: np.ndarray, f: float, u: float) -> complex:
    rho_arr, hh_arr = _validate_layer_model(rho, hh)
    if f <= 0.0:
        raise ValueError(f'f must be positive, got {f}')
    if u <= 0.0:
        raise ValueError(f'u must be positive, got {u}')
    if _b_numba is not None:
        return _b_numba(rho_arr, hh_arr, float(f), float(u))

    omega_term = 1j * MU0 * TWO_PI * float(f)
    bval = np.sqrt(u * u + omega_term / rho_arr[-1])
    if rho_arr.size == 1:
        return complex(bval)
    for idx in range(rho_arr.size - 2, -1, -1):
        alpha = np.sqrt(u * u + omega_term / rho_arr[idx])
        h = hh_arr[idx]
        s1 = 0.0 + 0.0j
        if (2.0 * alpha * h).real < 400.0:
            s1 = np.exp(-2.0 * alpha * h)
        s2 = (1.0 - s1) / (1.0 + s1)
        bval = alpha * (bval + alpha * s2) / (alpha + bval * s2)
    return complex(bval)


def _hankel_sum(
    rho: np.ndarray,
    hh: np.ndarray,
    f: float,
    z: float,
    r: float,
    weights_rev: np.ndarray,
    power_u: int,
    filt: HankelFilter,
) -> complex:
    rho_arr, hh_arr = _validate_layer_model(rho, hh)
    if f <= 0.0:
        raise ValueError(f'f must be positive, got {f}')
    if r <= 0.0:
        raise ValueError(f'r must be positive, got {r}')
    if z < 0.0:
        raise ValueError(f'z must be non-negative, got {z}')
    if _hankel_sum_numba is not None:
        return _hankel_sum_numba(rho_arr, hh_arr, float(f), float(z), float(r), weights_rev, filt.log_u_base, power_u)

    total = 0.0 + 0.0j
    u_grid = compute_hankel_u_grid(float(r), filt)
    for i, u in enumerate(u_grid):
        bval = b(rho_arr, hh_arr, float(f), float(u))
        ref = (bval - u) / (bval + u)
        atten = float(expc(-u * float(z)))
        if power_u == 1:
            total += ref * atten * u * weights_rev[i]
        else:
            total += ref * atten * u * u * weights_rev[i]
    return total / float(r)


def t3(rho: np.ndarray, hh: np.ndarray, f: float, z: float, r: float, filt: HankelFilter | None = None) -> complex:
    if filt is None:
        filt = get_hankel_filter()
    return _hankel_sum(rho, hh, f, z, r, filt.h0_rev, 2, filt)


def t5(rho: np.ndarray, hh: np.ndarray, f: float, z: float, r: float, filt: HankelFilter | None = None) -> complex:
    if filt is None:
        filt = get_hankel_filter()
    return _hankel_sum(rho, hh, f, z, r, filt.h1_rev, 1, filt)


def t6(rho: np.ndarray, hh: np.ndarray, f: float, z: float, r: float, filt: HankelFilter | None = None) -> complex:
    if filt is None:
        filt = get_hankel_filter()
    return _hankel_sum(rho, hh, f, z, r, filt.h1_rev, 2, filt)


def _forward_from_kernels(item: int, zplus: float, r: float, t3_val: complex | None = None, t5_val: complex | None = None, t6_val: complex | None = None) -> complex:
    if item == 1:
        hf = t6_val / (4.0 * PI)
    elif item == 2:
        hf = -t3_val / (4.0 * PI)
    elif item == 3:
        hf = (-t3_val + t5_val / r) / (4.0 * PI)
        if zplus < 0.0:
            hf = -hf
    elif item == 4:
        hf = t5_val / (4.0 * PI * r)
        if zplus < 0.0:
            hf = -hf
    elif item == 5:
        hf = -t6_val / (4.0 * PI)
        if zplus < 0.0:
            hf = -hf
    else:
        raise ValueError(f'item must be an integer between 1 and 5, got {item}')
    return hf * MU0


def forward(
    rho: np.ndarray,
    hh: np.ndarray,
    f: float,
    item: int,
    zplus: float,
    zminus: float,
    r: float,
    filt: HankelFilter | None = None,
) -> complex:
    if item not in (1, 2, 3, 4, 5):
        raise ValueError(f'item must be an integer between 1 and 5, got {item}')
    if f <= 0.0:
        raise ValueError(f'f must be positive, got {f}')
    if r <= 0.0:
        raise ValueError(f'r must be positive, got {r}')
    if zminus < 0.0:
        raise ValueError(f'zminus must be non-negative, got {zminus}')
    if filt is None:
        filt = get_hankel_filter()

    t3_val = t5_val = t6_val = None
    if item == 1:
        t6_val = t6(rho, hh, f, zminus, r, filt)
    elif item == 2:
        t3_val = t3(rho, hh, f, zminus, r, filt)
    elif item == 3:
        t3_val = t3(rho, hh, f, zminus, r, filt)
        t5_val = t5(rho, hh, f, zminus, r, filt)
    elif item == 4:
        t5_val = t5(rho, hh, f, zminus, r, filt)
    elif item == 5:
        t6_val = t6(rho, hh, f, zminus, r, filt)

    return _forward_from_kernels(item, float(zplus), float(r), t3_val, t5_val, t6_val)


if njit is not None:
    @njit(cache=True)
    def _splin1_numba(y: np.ndarray) -> np.ndarray:
        n = y.size
        c = np.zeros((3, n), dtype=np.float64)
        n1 = n - 1
        for i in range(1, n1):
            c[0, i] = y[i + 1] - 2.0 * y[i] + y[i - 1]
        c[1, 0] = 0.0
        c[2, 0] = 0.0
        for i in range(1, n1):
            p = 4.0 + c[1, i - 1]
            c[1, i] = -1.0 / p
            c[2, i] = (c[0, i] - c[2, i - 1]) / p
        c[0, n - 1] = 0.0
        for ii in range(1, n1):
            i = n - 1 - ii
            c[0, i] = c[1, i] * c[0, i + 1] + c[2, i]
        c[0, 0] = 0.0
        for i in range(n1):
            c[1, i] = y[i + 1] - y[i] - c[0, i + 1] + c[0, i]
            c[2, i] = y[i] - c[0, i]
        c[2, n - 1] = y[n - 1]
        return c

    @njit(cache=True)
    def _splin2_numba(c: np.ndarray, xint: float, x1: float, x2: float) -> float:
        n = c.shape[1]
        h = (x2 - x1) / float(n - 1)
        if xint < x1:
            p = (xint - x1) / h
            return c[1, 0] * p + c[2, 0]
        if xint >= x2:
            p = (xint - x2) / h
            return c[1, n - 2] * p + c[2, n - 1]
        u = (xint - x1) / h
        i0 = int(u)
        p = u - i0
        q = 1.0 - p
        return c[0, i0] * q ** 3 + c[0, i0 + 1] * p ** 3 + c[1, i0] * p + c[2, i0]

    @njit(cache=True)
    def _spl_numba(x: np.ndarray, fx: np.ndarray, x2: np.ndarray) -> np.ndarray:
        c = _splin1_numba(fx)
        xl1 = math.log10(x[0])
        xl2 = math.log10(x[-1])
        out = np.empty(x2.size, dtype=np.float64)
        for i in range(x2.size):
            out[i] = _splin2_numba(c, math.log10(x2[i]), xl1, xl2)
        return out
else:
    _splin1_numba = None
    _splin2_numba = None
    _spl_numba = None


def _validate_log_grid(x: np.ndarray, fx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1 or x_arr.size < 2:
        raise ValueError('x must be a 1D array with at least 2 points')
    if np.any(x_arr <= 0.0):
        raise ValueError('x must be strictly positive for log10 interpolation')
    if fx is None:
        fx_arr = None
    else:
        fx_arr = np.asarray(fx, dtype=np.float64)
        if fx_arr.ndim != 1 or fx_arr.size != x_arr.size:
            raise ValueError('fx must be a 1D array with the same length as x')
    logx = np.log10(x_arr)
    dlog = np.diff(logx)
    if not np.allclose(dlog, dlog[0], rtol=2e-7, atol=1e-10):
        raise ValueError('x must be approximately uniform in log10 space')
    return x_arr, fx_arr


def splin1(y: np.ndarray) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim != 1 or y_arr.size < 2:
        raise ValueError('y must be a 1D array with at least 2 points')
    if _splin1_numba is not None:
        return _splin1_numba(y_arr)

    n = y_arr.size
    c = np.zeros((3, n), dtype=np.float64)
    n1 = n - 1
    for i in range(1, n1):
        c[0, i] = y_arr[i + 1] - 2.0 * y_arr[i] + y_arr[i - 1]
    c[1, 0] = 0.0
    c[2, 0] = 0.0
    for i in range(1, n1):
        p = 4.0 + c[1, i - 1]
        c[1, i] = -1.0 / p
        c[2, i] = (c[0, i] - c[2, i - 1]) / p
    c[0, n - 1] = 0.0
    for ii in range(1, n1):
        i = n - 1 - ii
        c[0, i] = c[1, i] * c[0, i + 1] + c[2, i]
    c[0, 0] = 0.0
    for i in range(n1):
        c[1, i] = y_arr[i + 1] - y_arr[i] - c[0, i + 1] + c[0, i]
        c[2, i] = y_arr[i] - c[0, i]
    c[2, n - 1] = y_arr[n - 1]
    return c


def splin2(c: np.ndarray, xint: float, x1: float, x2: float) -> float:
    c_arr = np.asarray(c, dtype=np.float64)
    if c_arr.ndim != 2 or c_arr.shape[0] != 3 or c_arr.shape[1] < 2:
        raise ValueError('c must have shape (3, n) with n >= 2')
    if _splin2_numba is not None:
        return float(_splin2_numba(c_arr, float(xint), float(x1), float(x2)))

    n = c_arr.shape[1]
    h = (x2 - x1) / float(n - 1)
    if xint < x1:
        p = (xint - x1) / h
        return float(c_arr[1, 0] * p + c_arr[2, 0])
    if xint >= x2:
        p = (xint - x2) / h
        return float(c_arr[1, n - 2] * p + c_arr[2, n - 1])
    u = (xint - x1) / h
    i0 = int(u)
    p = u - i0
    q = 1.0 - p
    return float(c_arr[0, i0] * q ** 3 + c_arr[0, i0 + 1] * p ** 3 + c_arr[1, i0] * p + c_arr[2, i0])


def spl(x: np.ndarray, fx: np.ndarray, x2: np.ndarray) -> np.ndarray:
    x_arr, fx_arr = _validate_log_grid(x, fx)
    x2_arr = np.asarray(x2, dtype=np.float64)
    if x2_arr.ndim != 1:
        raise ValueError('x2 must be a 1D array')
    if np.any(x2_arr <= 0.0):
        raise ValueError('x2 must be strictly positive for log10 interpolation')
    if _spl_numba is not None:
        return _spl_numba(x_arr, fx_arr, x2_arr)

    c = splin1(fx_arr)
    xl1 = math.log10(x_arr[0])
    xl2 = math.log10(x_arr[-1])
    out = np.empty(x2_arr.size, dtype=np.float64)
    for i in range(x2_arr.size):
        out[i] = splin2(c, math.log10(x2_arr[i]), xl1, xl2)
    return out



FRT_Q = 1.258925412
FRT_NC = 160
FRT_NCNULL = 80

_FRT_FRQ_VALUES = (
    1.000000000000000e-03, 1.467799300000000e-03, 2.154434700000000e-03, 3.162277700000000e-03,
    4.641588800000000e-03, 6.812920700000000e-03, 1.000000000000000e-02, 1.467799300000000e-02,
    2.154434700000000e-02, 3.162277700000000e-02, 4.641588800000000e-02, 6.812920700000000e-02,
    1.000000000000000e-01, 1.467799300000000e-01, 2.154434700000000e-01, 3.162277700000000e-01,
    4.641588800000000e-01, 6.812920700000000e-01, 1.000000000000000e+00, 1.467799300000000e+00,
    2.154434700000000e+00, 3.162277700000000e+00, 4.641588800000000e+00, 6.812920700000000e+00,
    1.000000000000000e+01, 1.467799300000000e+01, 2.154434700000000e+01, 3.162277700000000e+01,
    4.641588800000000e+01, 6.812920699999999e+01, 1.000000000000000e+02, 1.467799300000000e+02,
    2.154434700000000e+02, 3.162277700000000e+02, 4.641588800000000e+02, 6.812920700000000e+02,
    1.000000000000000e+03, 1.467799300000000e+03, 2.154434700000000e+03, 3.162277700000000e+03,
    4.641588800000000e+03, 6.812920700000000e+03, 1.000000000000000e+04, 1.467799300000000e+04,
    2.154434700000000e+04, 3.162277700000000e+04, 4.641588800000000e+04, 6.812920699999999e+04,
    1.000000000000000e+05, 1.467799300000000e+05, 2.154434700000000e+05, 3.162277700000000e+05,
    4.641588800000000e+05, 6.812920699999999e+05, 1.000000000000000e+06, 1.467799300000000e+06,
    2.154434700000000e+06, 3.162277700000000e+06, 4.641588800000000e+06, 6.812920700000000e+06,
    1.000000000000000e+07, 1.467799300000000e+07, 2.154434700000000e+07, 3.162277700000000e+07,
    4.641588800000000e+07, 6.812920700000000e+07, 1.000000000000000e+08,
)

_FRT_H_VALUES = (
    2.595111399388290e-13, 3.665687713235550e-13, 5.177928766162420e-13, 7.314007304057910e-13,
    1.033132811562350e-12, 1.459336000883870e-12, 2.061371462346990e-12, 2.911757339624180e-12,
    4.112978044578700e-12, 5.809717711179840e-12, 8.206473230997420e-12, 1.159190583893650e-11,
    1.637407465477800e-11, 2.312888039304310e-11, 3.267059389022880e-11, 4.614815207210980e-11,
    6.518645450470520e-11, 9.207758995325450e-11, 1.300642009802190e-10, 1.837187473962550e-10,
    2.595125123778840e-10, 3.665665961542420e-10, 5.177963240272790e-10, 7.313952666275010e-10,
    1.033141471067360e-09, 1.459322276493330e-09, 2.061393214040130e-09, 2.911722865513800e-09,
    4.113032682361580e-09, 5.809631116129750e-09, 8.206610474902851e-09, 1.159168832200510e-08,
    1.637441939588180e-08, 2.312833401521440e-08, 3.267145984072990e-08, 4.614677963305560e-08,
    6.847447288677200e-08, 5.465746774903740e-08, 1.133198987774930e-07, 2.165299741575270e-07,
    2.886299422141400e-07, 3.428727280511250e-07, 4.791194887062620e-07, 7.420894188897520e-07,
    1.077365205352710e-06, 1.463832313065750e-06, 2.017276821346680e-06, 2.890581976174310e-06,
    4.152378088670220e-06, 5.844489893617420e-06, 8.180294303484191e-06, 1.154208544814940e-05,
    1.638970171453220e-05, 2.317690961138900e-05, 3.268726763313300e-05, 4.607868667018510e-05,
    6.518273213516360e-05, 9.208625895400371e-05, 1.301691426159510e-04, 1.835874811116270e-04,
    2.595955443937230e-04, 3.663243837193230e-04, 5.182106974625010e-04, 7.307299695625310e-04,
    1.033852391323890e-03, 1.457387640447300e-03, 2.062982564027320e-03, 2.906064015789590e-03,
    4.114679578837400e-03, 5.790342533211200e-03, 8.200057212352200e-03, 1.151938923331040e-02,
    1.630393989007890e-02, 2.282568109844870e-02, 3.222485551636920e-02, 4.478651016700110e-02,
    6.273306748745450e-02, 8.570586728474711e-02, 1.174181794076050e-01, 1.536326458323050e-01,
    1.977181118951020e-01, 2.288499242632470e-01, 2.403109050124220e-01, 1.654090719294040e-01,
    2.847096851671140e-03, -2.880158462696870e-01, -3.690973918532250e-01, -2.501098659226010e-02,
    5.718111095004260e-01, -3.922613902127690e-01, 7.632827742973269e-02, 5.162336929278510e-02,
    -6.480151605764320e-02, 4.890455225025520e-02, -3.269343077947500e-02, 2.105425709497450e-02,
    -1.338628489347360e-02, 8.470988014792590e-03, -5.351345159197510e-03, 3.378140238063490e-03,
    -2.131573640024700e-03, 1.345063524745580e-03, -8.489297437718030e-04, 5.355218223567130e-04,
    -3.377447999863820e-04, 2.132687926332040e-04, -1.346299697231560e-04, 8.477374166792790e-05,
    -5.349406358270960e-05, 3.390441629819100e-05, -2.133156383587940e-05, 1.334409116250190e-05,
    -8.516290738256340e-06, 5.443626722732110e-06, -3.321122784178960e-06, 2.071471908523860e-06,
    -1.420094125555110e-06, 8.782477549980040e-07, -4.556628047370300e-07, 3.385981030400090e-07,
    -2.874078307722510e-07, 1.078661505456990e-07, -2.472402418535800e-08, 5.355351103960300e-08,
    -3.378998113137800e-08, 2.132003675318200e-08, -1.345203377400750e-08, 8.487659507905460e-09,
    -5.355351103960180e-09, 3.378998111313830e-09, -2.132003675318190e-09, 1.345203377400750e-09,
    -8.487659507905760e-10, 5.355351103960150e-10, -3.378998111313820e-10, 2.132003675318110e-10,
    -1.345203377400790e-10, 8.487659507905721e-11, -5.355351103960340e-11, 3.378998111313810e-11,
    -2.132003675318180e-11, 1.345203377400740e-11, -8.487659507905710e-12, 5.355351103960310e-12,
    -3.378998111313790e-12, 2.132003675318170e-12, -1.345203377400730e-12, 8.487659507905670e-13,
    -5.355351103960290e-13, 3.378998111313770e-13, -2.132003675318160e-13, 1.345203377400780e-13,
    -8.487659507905960e-14, 5.355351103960070e-14, -3.378998111313770e-14, 2.132003675318160e-14,
    -1.345203377400830e-14, 8.487655079055799e-15, -5.355351103960250e-15, 3.378998111313890e-15,
)


@dataclass(frozen=True)
class FRTFilter:
    q: float
    nc: int
    ncnull: int
    frq: np.ndarray
    h: np.ndarray
    h_rev: np.ndarray


def build_frt_filter() -> FRTFilter:
    frq = _as_float64_array(_FRT_FRQ_VALUES)
    h = _as_float64_array(_FRT_H_VALUES)
    if frq.shape != (67,):
        raise ValueError(f'frq length mismatch: expected 67, got {frq.shape}')
    if h.shape != (FRT_NC,):
        raise ValueError(f'h length mismatch: expected {FRT_NC}, got {h.shape}')
    return FRTFilter(
        q=FRT_Q,
        nc=FRT_NC,
        ncnull=FRT_NCNULL,
        frq=frq,
        h=h,
        h_rev=h[::-1].copy(),
    )


@lru_cache(maxsize=1)
def get_frt_filter() -> FRTFilter:
    return build_frt_filter()


def compute_frt_frequency_grid(t: float, filt: FRTFilter | None = None) -> tuple[np.ndarray, np.ndarray]:
    if t <= 0.0:
        raise ValueError(f't must be positive, got {t}')
    if filt is None:
        filt = get_frt_filter()
    nn = np.arange(1.0, filt.nc + 1.0, dtype=np.float64)
    n = -filt.nc + filt.ncnull + nn
    omega = np.power(filt.q, -(n - 1.0)) / float(t)
    f = omega / TWO_PI
    return omega, f


if njit is not None:
    @njit(cache=True)
    def _frt_accumulate_numba(
        omega: np.ndarray,
        funr1: np.ndarray,
        funi1: np.ndarray,
        h_rev: np.ndarray,
        ic: int,
        t: float,
    ) -> float:
        total = 0.0
        for i in range(omega.size):
            if ic == 0:
                imag_fun = funi1[i]
            else:
                imag_fun = funr1[i] / omega[i]
            total += imag_fun * math.sqrt(omega[i]) * h_rev[i]
        return -total * math.sqrt(2.0 / (PI * t))
else:
    _frt_accumulate_numba = None


def frt_from_spectrum(
    t: float,
    func_item: np.ndarray,
    ic: int,
    frq: np.ndarray | None = None,
    filt: FRTFilter | None = None,
) -> float:
    if t <= 0.0:
        raise ValueError(f't must be positive, got {t}')
    if ic not in (0, 1):
        raise ValueError(f'ic must be 0 or 1, got {ic}')
    func_arr = np.asarray(func_item, dtype=np.complex128)
    if func_arr.ndim != 1:
        raise ValueError('func_item must be a 1D complex array')
    if filt is None:
        filt = get_frt_filter()
    if frq is None:
        frq_arr = filt.frq
    else:
        frq_arr = np.asarray(frq, dtype=np.float64)
        if frq_arr.ndim != 1 or frq_arr.size != func_arr.size:
            raise ValueError('frq must be a 1D array with the same length as func_item')

    omega, f_target = compute_frt_frequency_grid(float(t), filt)
    funr1 = spl(frq_arr, func_arr.real, f_target)
    funi1 = spl(frq_arr, func_arr.imag, f_target)

    if _frt_accumulate_numba is not None:
        return float(_frt_accumulate_numba(omega, funr1, funi1, filt.h_rev, ic, float(t)))

    total = 0.0
    for i in range(omega.size):
        if ic == 0:
            imag_fun = funi1[i]
        else:
            imag_fun = funr1[i] / omega[i]
        total += imag_fun * math.sqrt(omega[i]) * filt.h_rev[i]
    return float(-total * math.sqrt(2.0 / (PI * float(t))))


def frt(
    rho: np.ndarray,
    hh: np.ndarray,
    t: float,
    item: int,
    zplus: float,
    zminus: float,
    r: float,
    ic: int,
    frq: np.ndarray | None = None,
    hankel_filt: HankelFilter | None = None,
    frt_filt: FRTFilter | None = None,
) -> float:
    if frt_filt is None:
        frt_filt = get_frt_filter()
    if frq is None:
        frq_arr = frt_filt.frq
    else:
        frq_arr = np.asarray(frq, dtype=np.float64)
    func_item = np.empty(frq_arr.size, dtype=np.complex128)
    for i, fi in enumerate(frq_arr):
        func_item[i] = forward(rho, hh, float(fi), item, zplus, zminus, r, hankel_filt)
    return frt_from_spectrum(float(t), func_item, ic, frq_arr, frt_filt)


def _extract_waveform_point(p: np.ndarray, ns_id: int) -> tuple[float, float]:
    p_arr = np.asarray(p, dtype=np.float64)
    if p_arr.ndim != 1 or p_arr.size < ns_id * 2:
        raise ValueError(f'waveform point array is too short for ns_id={ns_id}')
    idx = (ns_id - 1) * 2
    return float(p_arr[idx]), float(p_arr[idx + 1])


def _prepare_trapezoid_waveform(ns_id: int, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray, t_ed: float) -> tuple[float, float, float, float, float, float, float]:
    t1, a1 = _extract_waveform_point(p1, ns_id)
    t2, a2 = _extract_waveform_point(p2, ns_id)
    t3v, a3 = _extract_waveform_point(p3, ns_id)
    t4, a4 = _extract_waveform_point(p4, ns_id)

    tt1 = t2 - t1
    tt2 = t3v - t2
    tt3 = t4 - t3v
    tt4 = t_ed - tt1 - tt2 - tt3
    if tt1 <= 0.0 or tt2 <= 0.0 or tt3 <= 0.0 or tt4 < 0.0:
        raise ValueError('invalid trapezoid timing derived from p1..p4 and t_ed')

    slope1 = (a2 - a1) / tt1
    slope2 = (a2 - a3) / tt2
    slope3 = (a3 - a4) / tt3
    period = tt1 + tt2 + tt3 + tt4
    return tt1, tt2, tt3, tt4, slope1, slope2, slope3, period


def _frt_shifted_from_spectrum(t_shift: float, func_item: np.ndarray, frt_filt: FRTFilter, ic: int) -> float:
    if t_shift <= 0.0:
        return 0.0
    return frt_from_spectrum(t_shift, func_item, ic, frt_filt.frq, frt_filt)


def forwardprocess(
    tlog_a: np.ndarray,
    rho: np.ndarray,
    hh: np.ndarray,
    ns_id: int,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
    ht: float,
    t_ed: float,
    xr: float,
    hr: float,
    rt: float,
    rr: float,
    nturn: int,
    nturn1: int,
    ic: int = 3,
    hankel_filt: HankelFilter | None = None,
    frt_filt: FRTFilter | None = None,
) -> np.ndarray:
    if ic != 3:
        raise NotImplementedError(f'forwardprocess currently supports only ic=3, got {ic}')
    t_arr = np.asarray(tlog_a, dtype=np.float64)
    if t_arr.ndim != 1 or t_arr.size == 0:
        raise ValueError('tlog_a must be a non-empty 1D array')
    if np.any(t_arr <= 0.0):
        raise ValueError('tlog_a must be strictly positive')
    if ns_id < 1:
        raise ValueError(f'ns_id must be >= 1, got {ns_id}')
    if hankel_filt is None:
        hankel_filt = get_hankel_filter()
    if frt_filt is None:
        frt_filt = get_frt_filter()

    zplus = float(ht - hr)
    zminus = float(ht + hr)
    r = float(np.sqrt(xr * xr))

    tt1, tt2, tt3, tt4, slope1, slope2, slope3, period = _prepare_trapezoid_waveform(ns_id, p1, p2, p3, p4, float(t_ed))

    func_item = np.empty(frt_filt.frq.size, dtype=np.complex128)
    for i, fi in enumerate(frt_filt.frq):
        func_item[i] = forward(rho, hh, float(fi), 2, zplus, zminus, r, hankel_filt)

    geom = (4.0 * rt * rt * nturn) * (PI * rr * rr * nturn1)
    hz1 = np.empty_like(t_arr)

    for i, ti in enumerate(t_arr):
        kpls = int(np.floor(ti / period)) + 1
        if kpls < 1:
            kpls = 1
        ss2 = 0.0

        for ip in range(1, kpls):
            base = ti - (ip - 1) * period
            ss2 += slope1 * _frt_shifted_from_spectrum(base, func_item, frt_filt, 1)
            ss2 += -(slope1 + slope2) * _frt_shifted_from_spectrum(base - tt1, func_item, frt_filt, 1)
            ss2 += -(slope3 - slope2) * _frt_shifted_from_spectrum(base - tt1 - tt2, func_item, frt_filt, 1)
            ss2 += slope3 * _frt_shifted_from_spectrum(base - tt1 - tt2 - tt3, func_item, frt_filt, 1)

        base = ti - (kpls - 1) * period
        if base > 0.0:
            ss2 += slope1 * _frt_shifted_from_spectrum(base, func_item, frt_filt, 1)
            if base > tt1:
                ss2 += -(slope1 + slope2) * _frt_shifted_from_spectrum(base - tt1, func_item, frt_filt, 1)
                if base > tt1 + tt2:
                    ss2 += -(slope3 - slope2) * _frt_shifted_from_spectrum(base - tt1 - tt2, func_item, frt_filt, 1)
                    if base > tt1 + tt2 + tt3:
                        ss2 += slope3 * _frt_shifted_from_spectrum(base - tt1 - tt2 - tt3, func_item, frt_filt, 1)

        hz1[i] = geom * ss2

    return hz1


def _check_close(name: str, value: float, expected: float, atol: float = 1e-15) -> None:
    if not np.isclose(value, expected, rtol=0.0, atol=atol):
        raise AssertionError(f'{name} mismatch: got {value:.16e}, expected {expected:.16e}')


def _run_self_check() -> None:
    filt = get_hankel_filter()
    assert filt.nc == 100
    assert filt.ncnull == 60
    assert filt.h0.shape == (100,)
    assert filt.h1.shape == (100,)
    assert filt.h0.dtype == np.float64
    assert filt.h1.dtype == np.float64

    _check_close('h0[0]', filt.h0[0], 2.89878288e-07)
    _check_close('h0[47]', filt.h0[47], 1.45145832e-02)
    _check_close('h0[48]', filt.h0[48], 1.82601199e-02)
    _check_close('h0[-1]', filt.h0[-1], -6.24130946e-08)

    _check_close('h1[0]', filt.h1[0], 1.84909557e-13)
    _check_close('h1[47]', filt.h1[47], 4.51975783e-04)
    _check_close('h1[48]', filt.h1[48], 7.35565435e-04)
    _check_close('h1[-1]', filt.h1[-1], 5.45397218e-07)

    u_grid = compute_hankel_u_grid(1.0, filt)
    _check_close('u_grid[0]', u_grid[0], 1.0e4, atol=1e-10)
    _check_close('u_grid[-1]', u_grid[-1], 10.0 ** (-5.9), atol=1e-15)

    _check_close('expc(0.0)', float(expc(0.0)), 1.0)
    _check_close('expc(1.0)', float(expc(1.0)), math.e)
    _check_close('expc(700.0)', float(expc(700.0)), math.exp(650.0), atol=1e-12 * math.exp(650.0))
    _check_close('expc(-700.0)', float(expc(-700.0)), math.exp(-650.0), atol=1e-15)
    if _expc_numba_scalar is not None:
        _check_close('numba_expc(700.0)', _expc_numba_scalar(700.0), math.exp(650.0), atol=1e-12 * math.exp(650.0))
        _check_close('numba_expc(-700.0)', _expc_numba_scalar(-700.0), math.exp(-650.0), atol=1e-15)

    rho_half = np.array([100.0], dtype=np.float64)
    hh_half = np.empty(0, dtype=np.float64)
    u0 = 2.5
    f0 = 1000.0
    b_expected = np.sqrt(u0 * u0 + 1j * MU0 * TWO_PI * f0 / rho_half[0])
    b_val = b(rho_half, hh_half, f0, u0)
    if not np.allclose(b_val, b_expected, rtol=1e-12, atol=1e-12):
        raise AssertionError(f'b half-space mismatch: {b_val} vs {b_expected}')
    if _b_numba is not None:
        b_numba = _b_numba(rho_half, hh_half, f0, u0)
        if not np.allclose(b_numba, b_val, rtol=1e-12, atol=1e-12):
            raise AssertionError(f'b numba mismatch: {b_numba} vs {b_val}')

    rho_small = np.array([100.0, 10.0], dtype=np.float64)
    hh_small = np.array([5.0], dtype=np.float64)
    r0 = 0.58
    z0 = 0.1
    t3_val = t3(rho_small, hh_small, f0, z0, r0, filt)
    t5_val = t5(rho_small, hh_small, f0, z0, r0, filt)
    t6_val = t6(rho_small, hh_small, f0, z0, r0, filt)
    if not (np.isfinite(t3_val.real) and np.isfinite(t3_val.imag)):
        raise AssertionError('t3 returned non-finite value')
    if not (np.isfinite(t5_val.real) and np.isfinite(t5_val.imag)):
        raise AssertionError('t5 returned non-finite value')
    if not (np.isfinite(t6_val.real) and np.isfinite(t6_val.imag)):
        raise AssertionError('t6 returned non-finite value')

    sample_path = '/Users/xiechushu/project/fortran_forward_log_sample_aligned/f_副本.json'
    with open(sample_path, encoding='utf-8') as fh:
        sample = json.load(fh)
    rho_real = np.asarray(sample['rho'], dtype=np.float64)
    hh_real = np.asarray(sample['hh'], dtype=np.float64)
    if hh_real.size == rho_real.size:
        hh_real = hh_real[:-1]
    real_t3 = t3(rho_real, hh_real, 1000.0, 0.1, 0.58, filt)
    real_t5 = t5(rho_real, hh_real, 1000.0, 0.1, 0.58, filt)
    real_t6 = t6(rho_real, hh_real, 1000.0, 0.1, 0.58, filt)
    if not (np.isfinite(real_t3.real) and np.isfinite(real_t3.imag)):
        raise AssertionError('real-case t3 returned non-finite value')
    if not (np.isfinite(real_t5.real) and np.isfinite(real_t5.imag)):
        raise AssertionError('real-case t5 returned non-finite value')
    if not (np.isfinite(real_t6.real) and np.isfinite(real_t6.imag)):
        raise AssertionError('real-case t6 returned non-finite value')

    forward_item2 = forward(rho_small, hh_small, f0, 2, zplus=0.1, zminus=z0, r=r0, filt=filt)
    manual_item2 = _forward_from_kernels(2, 0.1, r0, t3_val=t3_val)
    if not np.allclose(forward_item2, manual_item2, rtol=1e-12, atol=1e-12):
        raise AssertionError(f'forward item2 mismatch: {forward_item2} vs {manual_item2}')

    for item in (1, 2, 3, 4, 5):
        fwd_pos = forward(rho_small, hh_small, f0, item, zplus=0.1, zminus=z0, r=r0, filt=filt)
        if item in (1, 2):
            fwd_neg = forward(rho_small, hh_small, f0, item, zplus=-0.1, zminus=z0, r=r0, filt=filt)
            if not np.allclose(fwd_pos, fwd_neg, rtol=1e-12, atol=1e-12):
                raise AssertionError(f'item {item} should not depend on zplus sign')
        else:
            fwd_neg = forward(rho_small, hh_small, f0, item, zplus=-0.1, zminus=z0, r=r0, filt=filt)
            if not np.allclose(fwd_neg, -fwd_pos, rtol=1e-12, atol=1e-12):
                raise AssertionError(f'item {item} sign flip mismatch for zplus')

    for bad_item in (0, 6):
        try:
            forward(rho_small, hh_small, f0, bad_item, zplus=0.1, zminus=z0, r=r0, filt=filt)
        except ValueError:
            pass
        else:
            raise AssertionError(f'forward should reject item={bad_item}')

    for bad_f, bad_r, bad_zminus in ((0.0, r0, z0), (f0, 0.0, z0), (f0, r0, -0.1)):
        try:
            forward(rho_small, hh_small, bad_f, 2, zplus=0.1, zminus=bad_zminus, r=bad_r, filt=filt)
        except ValueError:
            pass
        else:
            raise AssertionError('forward should reject invalid scalar inputs')

    x = np.logspace(-2, 2, 6)
    fx = np.array([0.5, 1.0, 1.5, 1.0, 0.2, -0.1], dtype=np.float64)
    c_py = splin1(fx)
    if c_py.shape != (3, fx.size):
        raise AssertionError(f'splin1 returned wrong shape: {c_py.shape}')
    if not np.all(np.isfinite(c_py)):
        raise AssertionError('splin1 returned non-finite coefficients')
    if _splin1_numba is not None:
        c_nb = _splin1_numba(fx)
        if not np.allclose(c_py, c_nb, rtol=1e-12, atol=1e-12):
            raise AssertionError('splin1 Python and numba results differ')

    fx_same = spl(x, fx, x)
    if not np.allclose(fx_same, fx, rtol=1e-10, atol=1e-10):
        raise AssertionError(f'spl failed to reproduce knots: {fx_same} vs {fx}')

    logx = np.log10(x)
    c_for_boundary = splin1(fx)
    left_val = spl(x, fx, np.array([x[0] * 0.1], dtype=np.float64))[0]
    left_expected = splin2(c_for_boundary, math.log10(x[0] * 0.1), logx[0], logx[-1])
    if not np.isclose(left_val, left_expected, rtol=0.0, atol=1e-12):
        raise AssertionError('left boundary behavior mismatch')

    at_left_val = spl(x, fx, np.array([x[0]], dtype=np.float64))[0]
    at_left_expected = splin2(c_for_boundary, logx[0], logx[0], logx[-1])
    if not np.isclose(at_left_val, at_left_expected, rtol=0.0, atol=1e-12):
        raise AssertionError('x == x[0] behavior mismatch')

    at_right_val = spl(x, fx, np.array([x[-1]], dtype=np.float64))[0]
    at_right_expected = splin2(c_for_boundary, logx[-1], logx[0], logx[-1])
    if not np.isclose(at_right_val, at_right_expected, rtol=0.0, atol=1e-12):
        raise AssertionError('x == x[-1] behavior mismatch')

    right_val = spl(x, fx, np.array([x[-1] * 10.0], dtype=np.float64))[0]
    right_expected = splin2(c_for_boundary, math.log10(x[-1] * 10.0), logx[0], logx[-1])
    if not np.isclose(right_val, right_expected, rtol=0.0, atol=1e-12):
        raise AssertionError('right boundary behavior mismatch')

    x_dense = np.logspace(-2, 2, 21)
    fx_dense_py = spl(x, fx, x_dense)
    if _spl_numba is not None:
        fx_dense_nb = _spl_numba(x, fx, x_dense)
        if not np.allclose(fx_dense_py, fx_dense_nb, rtol=1e-12, atol=1e-12):
            raise AssertionError('spl Python and numba results differ on dense grid')

    for bad_x, bad_fx, bad_x2 in [
        (np.array([0.0, 1.0]), np.array([1.0, 2.0]), np.array([1.0])),
        (np.array([1.0, 10.0]), np.array([1.0]), np.array([1.0])),
        (np.array([1.0, 2.0, 10.0]), np.array([1.0, 2.0, 3.0]), np.array([1.0])),
    ]:
        try:
            spl(bad_x, bad_fx, bad_x2)
        except ValueError:
            pass
        else:
            raise AssertionError('spl should reject invalid input')

    try:
        spl(x, fx, np.array([0.0], dtype=np.float64))
    except ValueError:
        pass
    else:
        raise AssertionError('spl should reject non-positive x2')

    print('Hankel filter self-check passed.')
    print(f'nc={filt.nc}, ncnull={filt.ncnull}')
    print(f'h0_len={filt.h0.size}, h1_len={filt.h1.size}')
    print(f'h0[0]={filt.h0[0]:.8e}, h0[-1]={filt.h0[-1]:.8e}')
    print(f'h1[0]={filt.h1[0]:.8e}, h1[-1]={filt.h1[-1]:.8e}')
    print(f'u_grid[0]={u_grid[0]:.8e}, u_grid[-1]={u_grid[-1]:.8e}')
    print(f'b_half_space={b_val.real:.8e}+{b_val.imag:.8e}j')
    print(f't3_small={t3_val.real:.8e}+{t3_val.imag:.8e}j')
    print(f't5_small={t5_val.real:.8e}+{t5_val.imag:.8e}j')
    print(f't6_small={t6_val.real:.8e}+{t6_val.imag:.8e}j')
    print(f't3_real={real_t3.real:.8e}+{real_t3.imag:.8e}j')
    print(f't5_real={real_t5.real:.8e}+{real_t5.imag:.8e}j')
    print(f't6_real={real_t6.real:.8e}+{real_t6.imag:.8e}j')

    frt_filt = get_frt_filter()
    omega_grid, f_grid = compute_frt_frequency_grid(1.0e-3, frt_filt)
    _check_close('frt_omega_ratio', omega_grid[0] / omega_grid[1], FRT_Q, atol=1e-12)
    _check_close('frt_f_match', f_grid[0], omega_grid[0] / TWO_PI, atol=1e-15)

    func_test = np.exp(-0.1 * np.arange(frt_filt.frq.size)) + 1j * np.linspace(0.1, 1.0, frt_filt.frq.size)
    frt_py0 = frt_from_spectrum(1.0e-3, func_test, 0, frt_filt.frq, frt_filt)
    frt_py1 = frt_from_spectrum(1.0e-3, func_test, 1, frt_filt.frq, frt_filt)
    if not np.isfinite(frt_py0) or not np.isfinite(frt_py1):
        raise AssertionError('frt_from_spectrum returned non-finite values')

    funr1 = spl(frt_filt.frq, func_test.real, f_grid)
    funi1 = spl(frt_filt.frq, func_test.imag, f_grid)
    frt_ref0 = float(-np.sum(funi1 * np.sqrt(omega_grid) * frt_filt.h_rev) * math.sqrt(2.0 / (PI * 1.0e-3)))
    frt_ref1 = float(-np.sum((funr1 / omega_grid) * np.sqrt(omega_grid) * frt_filt.h_rev) * math.sqrt(2.0 / (PI * 1.0e-3)))
    if not np.isclose(frt_py0, frt_ref0, rtol=1e-12, atol=1e-12):
        raise AssertionError(f'frt ic=0 mismatch: {{frt_py0}} vs {{frt_ref0}}')
    if not np.isclose(frt_py1, frt_ref1, rtol=1e-12, atol=1e-12):
        raise AssertionError(f'frt ic=1 mismatch: {{frt_py1}} vs {{frt_ref1}}')
    if _frt_accumulate_numba is not None:
        frt_nb0 = _frt_accumulate_numba(omega_grid, funr1, funi1, frt_filt.h_rev, 0, 1.0e-3)
        frt_nb1 = _frt_accumulate_numba(omega_grid, funr1, funi1, frt_filt.h_rev, 1, 1.0e-3)
        if not np.isclose(frt_nb0, frt_py0, rtol=1e-12, atol=1e-12):
            raise AssertionError('numba frt ic=0 mismatch')
        if not np.isclose(frt_nb1, frt_py1, rtol=1e-12, atol=1e-12):
            raise AssertionError('numba frt ic=1 mismatch')

    frt_real0 = frt(rho_small, hh_small, 1.0e-3, 2, 0.1, z0, r0, 0, hankel_filt=filt, frt_filt=frt_filt)
    frt_real1 = frt(rho_small, hh_small, 1.0e-3, 2, 0.1, z0, r0, 1, hankel_filt=filt, frt_filt=frt_filt)
    if not np.isfinite(frt_real0) or not np.isfinite(frt_real1):
        raise AssertionError('frt real-case returned non-finite values')

    for bad_ic in (-1, 2):
        try:
            frt_from_spectrum(1.0e-3, func_test, bad_ic, frt_filt.frq, frt_filt)
        except ValueError:
            pass
        else:
            raise AssertionError('frt_from_spectrum should reject invalid ic')

    with open('/Users/xiechushu/project/EM_app/TEM_field_forward/point11/data_conf.json', encoding='utf-8') as fh:
        data_conf = json.load(fh)
    tlog_a = np.asarray(data_conf['gated_time'], dtype=np.float64)
    wave_t = np.asarray(data_conf['wave_start_time'], dtype=np.float64)
    wave_a = np.asarray(data_conf['wave_amp'], dtype=np.float64)
    p1 = wave_t[:1].repeat(2)
    p1[1] = wave_a[0]
    p2 = wave_t[1:2].repeat(2)
    p2[1] = wave_a[1]
    p3 = wave_t[2:3].repeat(2)
    p3[1] = wave_a[2]
    p4 = wave_t[3:4].repeat(2)
    p4[1] = wave_a[3]
    fp_resp = forwardprocess(
        tlog_a=tlog_a,
        rho=rho_real,
        hh=hh_real,
        ns_id=1,
        p1=p1,
        p2=p2,
        p3=p3,
        p4=p4,
        ht=0.0,
        t_ed=float(tlog_a[-1]),
        xr=0.58,
        hr=0.1,
        rt=0.5,
        rr=0.2,
        nturn=8,
        nturn1=90,
        ic=3,
        hankel_filt=filt,
        frt_filt=frt_filt,
    )
    if fp_resp.shape != tlog_a.shape:
        raise AssertionError(f'forwardprocess output shape mismatch: {fp_resp.shape} vs {tlog_a.shape}')
    if not np.all(np.isfinite(fp_resp)):
        raise AssertionError('forwardprocess returned non-finite values')
    if np.allclose(fp_resp, 0.0):
        raise AssertionError('forwardprocess returned all zeros')
    try:
        forwardprocess(
            tlog_a=tlog_a,
            rho=rho_real,
            hh=hh_real,
            ns_id=1,
            p1=p1,
            p2=p2,
            p3=p3,
            p4=p4,
            ht=0.0,
            t_ed=float(tlog_a[-1]),
            xr=0.58,
            hr=0.1,
            rt=0.5,
            rr=0.2,
            nturn=8,
            nturn1=90,
            ic=1,
            hankel_filt=filt,
            frt_filt=frt_filt,
        )
    except NotImplementedError:
        pass
    else:
        raise AssertionError('forwardprocess should reject ic!=3 in current implementation')

    print(f'frt_ic0={frt_py0:.8e}, frt_ic1={frt_py1:.8e}')
    print(f'frt_real_ic0={frt_real0:.8e}, frt_real_ic1={frt_real1:.8e}')
    print(f'forwardprocess_min={fp_resp.min():.8e}, forwardprocess_max={fp_resp.max():.8e}')
    print(f'numba_available={njit is not None}')


if __name__ == '__main__':
    _run_self_check()
