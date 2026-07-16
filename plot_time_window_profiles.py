"""Plot multi-point TEM profiles for post-turnoff time windows."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL))

from config import CURRENT_SCALE, RESULTS_DIR, SOURCE_DATA_PATH, TIME_GATE_START, USE_DENOISED
from data_loader import select_sy6_record


N_PROFILE_CHANNELS = 128
OUTPUT_DIR = RESULTS_DIR.parent
WINDOWS_MS = [
    (0.5, 1.0, "profile_off_0p1ms_to_2ms.png"),
    (10.0, 15.0, "profile_off_10ms_to_15ms.png"),
    (0.1, 15.0, "profile_off_0p1ms_to_15ms.png"),
]


def configure_font() -> None:
    for name in ["PingFang SC", "Heiti SC", "STHeiti", "SimHei", "Arial Unicode MS"]:
        if any(font.name == name for font in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def sort_key(path: Path) -> int:
    digits = "".join(ch for ch in path.name if ch.isdigit())
    return int(digits) if digits else 0


def format_channel_label(t_rel_ms: float) -> str:
    if t_rel_ms >= 10:
        return f"{t_rel_ms:.0f} ms"
    if t_rel_ms >= 1:
        return f"{t_rel_ms:.1f} ms"
    return f"{t_rel_ms:.2f} ms"


def infer_waveform_time_origin(time_axis: np.ndarray, current_raw: np.ndarray) -> float:
    mask2 = (time_axis >= 0.200) & (time_axis <= 0.202)
    idx2_win = np.where(mask2)[0]
    if idx2_win.size == 0:
        raise ValueError("实际时间轴不包含 200–202 ms 峰值搜索窗口")
    i2 = idx2_win[int(np.argmax(current_raw[idx2_win]))]

    mask1 = (time_axis >= 0.200) & (time_axis <= time_axis[i2])
    idx1_win = np.where(mask1)[0]
    if idx1_win.size == 0:
        raise ValueError("实际时间轴不包含上升沿搜索窗口")
    edge_n = max(1, min(idx1_win.size // 20, 2000))
    baseline = float(np.median(current_raw[idx1_win[:edge_n]]))
    threshold = baseline + 0.05 * (current_raw[i2] - baseline)
    candidates = np.where(current_raw[idx1_win] > threshold)[0]
    i1 = idx1_win[candidates[0]] if candidates.size else idx1_win[0]
    return float(time_axis[i1])


def load_point_series() -> tuple[list[str], list[np.ndarray], list[np.ndarray], list[float], list[str]]:
    point_dirs = sorted(
        [path for path in SOURCE_DATA_PATH.parent.iterdir() if path.is_dir() and path.name.startswith("测点")],
        key=sort_key,
    )
    point_names = []
    time_axes = []
    responses = []
    off_times = []
    shots = []
    for point_dir in point_dirs:
        record = select_sy6_record(point_dir, require_denoised_full=USE_DENOISED)
        loaded = record.load()
        response = loaded.denoised_full if USE_DENOISED else loaded.rx
        if response is None:
            raise FileNotFoundError(f"{record.shot} 缺少完整降噪信号")
        waveform_time_origin = infer_waveform_time_origin(loaded.time, loaded.current * CURRENT_SCALE)
        point_names.append(point_dir.name)
        time_axes.append(np.asarray(loaded.time, dtype=float))
        responses.append(np.asarray(response, dtype=float))
        off_times.append(waveform_time_origin + TIME_GATE_START)
        shots.append(record.shot)
    return point_names, time_axes, responses, off_times, shots


def plot_window(
    point_names: list[str],
    time_axes: list[np.ndarray],
    responses: list[np.ndarray],
    off_times: list[float],
    shots: list[str],
    t_start_ms: float,
    t_end_ms: float,
    file_name: str,
) -> Path:
    profile_times_ms = np.logspace(np.log10(max(t_start_ms, 1e-3)), np.log10(t_end_ms), N_PROFILE_CHANNELS)
    x_index = np.arange(len(point_names))
    colors = plt.cm.plasma(np.linspace(0.0, 0.9, profile_times_ms.size))

    fig, ax = plt.subplots(figsize=(12, 7), dpi=170)
    for color, t_rel_ms in zip(colors, profile_times_ms):
        values = []
        for time_axis, response, off_time in zip(time_axes, responses, off_times):
            target_time = off_time + t_rel_ms * 1e-3
            idx = int(np.searchsorted(time_axis, target_time))
            idx = min(max(idx, 0), time_axis.size - 1)
            if idx > 0 and abs(time_axis[idx - 1] - target_time) <= abs(time_axis[idx] - target_time):
                idx -= 1
            values.append(max(abs(float(response[idx])), 1e-12))
        ax.semilogy(
            x_index,
            values,
            "o-",
            color=color,
            ms=5,
            lw=1.1,
            label=format_channel_label(t_rel_ms),
        )

    ax.set_title(f"多时间通道剖面图 | 关断后 {t_start_ms:g}–{t_end_ms:g} ms")
    ax.set_xlabel("测点")
    ax.set_ylabel("|rx| (V)")
    ax.set_xticks(x_index)
    ax.set_xticklabels(point_names)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper right", frameon=False)
    # fig.text(
    #     0.012,
    #     0.012,
    #     "炮次: " + ", ".join(f"{point}:{shot}" for point, shot in zip(point_names, shots)),
    #     fontsize=7,
    # )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output_path = OUTPUT_DIR / file_name
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    configure_font()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    point_names, time_axes, responses, off_times, shots = load_point_series()
    print(f"Loaded {len(point_names)} points from {SOURCE_DATA_PATH.parent}")
    print(f"关断时间定义: waveform_time_origin + TIME_GATE_START = +{TIME_GATE_START*1e3:.3f} ms")
    for t_start_ms, t_end_ms, file_name in WINDOWS_MS:
        path = plot_window(
            point_names,
            time_axes,
            responses,
            off_times,
            shots,
            t_start_ms,
            t_end_ms,
            file_name,
        )
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
