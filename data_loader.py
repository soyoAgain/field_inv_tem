from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

@dataclass(frozen=True)
class Sy6Record:
    point: str
    shot: str
    directory: Path
    time_path: Path
    current_path: Path
    rx_path: Path
    info_path: Path | None = None
    denoised_full_path: Path | None = None
    denoised_decay_path: Path | None = None

    def load(self) -> "LoadedSy6Record":
        time = np.asarray(np.load(self.time_path), dtype=float)
        current = np.asarray(np.load(self.current_path), dtype=float)
        rx = np.asarray(np.load(self.rx_path), dtype=float)
        if not (time.ndim == current.ndim == rx.ndim == 1):
            raise ValueError(f"{self.shot} 的 t/current/rx 必须是一维数组")
        if not (time.size == current.size == rx.size):
            raise ValueError(
                f"{self.shot} 数据长度不一致: "
                f"time={time.size}, current={current.size}, rx={rx.size}"
            )
        info: dict[str, Any] = {}
        if self.info_path is not None and self.info_path.exists():
            with self.info_path.open("r", encoding="utf-8") as handle:
                info = json.load(handle)
        denoised_full = (
            np.asarray(np.load(self.denoised_full_path), dtype=float)
            if self.denoised_full_path is not None
            else None
        )
        denoised_decay = (
            np.asarray(np.load(self.denoised_decay_path), dtype=float)
            if self.denoised_decay_path is not None
            else None
        )
        if denoised_full is not None and denoised_full.size != time.size:
            raise ValueError(f"{self.shot} 的 denoised_full 与时间轴长度不一致")
        if denoised_decay is not None and denoised_decay.size > time.size:
            raise ValueError(f"{self.shot} 的 denoised_decay 长于完整时间轴")
        return LoadedSy6Record(
            record=self,
            time=time,
            current=current,
            rx=rx,
            info=info,
            denoised_full=denoised_full,
            denoised_decay=denoised_decay,
        )


@dataclass(frozen=True)
class LoadedSy6Record:
    record: Sy6Record
    time: np.ndarray
    current: np.ndarray
    rx: np.ndarray
    info: dict[str, Any]
    denoised_full: np.ndarray | None
    denoised_decay: np.ndarray | None

    @property
    def observed_decay(self) -> np.ndarray:
        if self.denoised_decay is not None:
            return self.denoised_decay
        return self.rx

    def decay_times(self) -> np.ndarray:
        if self.denoised_decay is None:
            return self.time
        return infer_decay_times(self.time, self.rx, self.denoised_decay)

def discover_sy6_records(base_dir: str | Path) -> list[Sy6Record]:
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(base)

    records: list[Sy6Record] = []
    for point_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        grouped: dict[str, dict[str, Path]] = {}
        for path in point_dir.glob("*.npy"):
            suffix = _classify_suffix(path.name)
            if suffix is None:
                continue
            stem = path.name[: -len(f"_{suffix}.npy")]
            grouped.setdefault(stem, {})[suffix] = path
        for info_path in point_dir.glob("*_info.json"):
            stem = info_path.name[: -len("_info.json")]
            grouped.setdefault(stem, {})["info"] = info_path
        for stem, files in grouped.items():
            if not {"t", "current", "rx"}.issubset(files):
                continue
            records.append(
                Sy6Record(
                    point=point_dir.name,
                    shot=stem,
                    directory=point_dir,
                    time_path=files["t"],
                    current_path=files["current"],
                    rx_path=files["rx"],
                    info_path=files.get("info"),
                    denoised_full_path=files.get("denoised_full"),
                    denoised_decay_path=files.get("denoised_decay"),
                )
            )
    return sorted(records, key=lambda item: (item.point, item.shot))


def select_sy6_record(
    source_dir: str | Path,
    *,
    shot: Optional[str] = None,
    require_denoised_full: bool = False,
) -> Sy6Record:
    """Select exactly one complete record for the configured point directory."""
    source = Path(source_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {source}")

    records = [
        record
        for record in discover_sy6_records(source.parent)
        if record.directory.resolve() == source
    ]
    if shot is not None:
        records = [record for record in records if record.shot == shot]
    if require_denoised_full:
        records = [record for record in records if record.denoised_full_path is not None]

    if not records:
        raise FileNotFoundError(
            f"未找到符合条件的数据记录: source={source}, shot={shot!r}, "
            f"require_denoised_full={require_denoised_full}"
        )
    if len(records) > 1:
        shots = ", ".join(record.shot for record in records)
        raise ValueError(f"数据目录存在多条候选记录，请明确指定炮号: {shots}")
    return records[0]
def infer_decay_times(full_time: np.ndarray, full_signal: np.ndarray, decay_signal: np.ndarray) -> np.ndarray:
    time = np.asarray(full_time, dtype=float)
    signal = np.asarray(full_signal, dtype=float)
    decay = np.asarray(decay_signal, dtype=float)
    if decay.size > time.size:
        raise ValueError("decay signal cannot be longer than full time array")
    if decay.size == time.size:
        return time

    first = decay[0]
    candidates = np.flatnonzero(np.isclose(signal, first, rtol=1e-5, atol=max(1e-10, abs(first) * 1e-5)))
    for start in candidates:
        stop = start + decay.size
        if stop <= signal.size and np.allclose(signal[start:stop], decay, rtol=1e-5, atol=1e-8):
            return time[start:stop]
    return time[-decay.size:]


def summarize_records(records: Iterable[Sy6Record]) -> dict[str, Any]:
    records = list(records)
    points = sorted({record.point for record in records})
    return {
        "record_count": len(records),
        "point_count": len(points),
        "points": points,
        "denoised_decay_count": sum(record.denoised_decay_path is not None for record in records),
    }


def _classify_suffix(name: str) -> str | None:
    suffixes = ["denoised_decay", "denoised_full", "current", "rx", "t"]
    for suffix in suffixes:
        if name.endswith(f"_{suffix}.npy"):
            return suffix
    return None
