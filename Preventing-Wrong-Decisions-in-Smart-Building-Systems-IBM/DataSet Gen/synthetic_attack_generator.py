from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NORMAL_LABEL = "normal"
REPLAY_LABEL = "replay_attack"
INJECTION_LABEL = "injection_attack"
DRIFT_LABEL = "drift_attack"
NOISE_LABEL = "noise_attack"
DROP_LABEL = "drop_attack"


@dataclass
class AttackGenerationConfig:
    input_path: str | Path = "../iot_sensor_dataset.csv"
    output_path: str | Path = "../enhanced_iot_dataset.csv"
    example_plot_path: str | Path = "../enhanced_iot_attack_examples.png"
    example_csv_path: str | Path = "../enhanced_iot_attack_examples.csv"
    random_seed: int = 42
    attack_ratio: float = 0.30
    include_drop_attack: bool = True
    replay_window_range: tuple[int, int] = (20, 60)
    injection_window_range: tuple[int, int] = (12, 30)
    drift_window_range: tuple[int, int] = (30, 60)
    noise_window_range: tuple[int, int] = (20, 60)
    drop_window_range: tuple[int, int] = (12, 30)
    temperature_clip: tuple[float, float] = (-20.0, 80.0)
    humidity_clip: tuple[float, float] = (0.0, 100.0)
    allow_nan_drop: bool = False

    @property
    def attack_types(self) -> list[str]:
        attacks = [REPLAY_LABEL, INJECTION_LABEL, DRIFT_LABEL, NOISE_LABEL]
        if self.include_drop_attack:
            attacks.append(DROP_LABEL)
        return attacks


@dataclass
class AttackSegment:
    attack_type: str
    sensor_id: str
    target_start_pos: int
    target_end_pos: int
    target_global_indices: list[int]
    donor_start_pos: int | None = None
    donor_end_pos: int | None = None
    donor_global_indices: list[int] | None = None

    @property
    def length(self) -> int:
        return self.target_end_pos - self.target_start_pos


def load_base_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    lower_map = {column.lower().strip(): column for column in df.columns}

    if "sensor_id" in lower_map:
        sensor_id = df[lower_map["sensor_id"]].astype(str)
    elif "device_id" in lower_map:
        sensor_id = df[lower_map["device_id"]].astype(str)
    else:
        sensor_id = pd.Series(["sensor_1"] * len(df))

    if "temperature_c" in lower_map:
        temperature = pd.to_numeric(df[lower_map["temperature_c"]], errors="coerce")
    elif "temperature" in lower_map:
        temperature = pd.to_numeric(df[lower_map["temperature"]], errors="coerce")
    else:
        raise ValueError("Input dataset must contain temperature or temperature_c.")

    if "humidity_percent" in lower_map:
        humidity = pd.to_numeric(df[lower_map["humidity_percent"]], errors="coerce")
    elif "humidity" in lower_map:
        humidity = pd.to_numeric(df[lower_map["humidity"]], errors="coerce")
    else:
        raise ValueError("Input dataset must contain humidity or humidity_percent.")

    if "timestamp" not in lower_map:
        raise ValueError("Input dataset must contain a timestamp column.")

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df[lower_map["timestamp"]], errors="coerce"),
            "sensor_id": sensor_id,
            "temperature_c": temperature,
            "humidity_percent": humidity,
        }
    )
    out = out.dropna(subset=["timestamp", "sensor_id", "temperature_c", "humidity_percent"]).reset_index(drop=True)
    out = out.sort_values(["timestamp", "sensor_id"]).reset_index(drop=True)
    return out


def _segment_range(config: AttackGenerationConfig, attack_type: str) -> tuple[int, int]:
    if attack_type == REPLAY_LABEL:
        return config.replay_window_range
    if attack_type == INJECTION_LABEL:
        return config.injection_window_range
    if attack_type == DRIFT_LABEL:
        return config.drift_window_range
    if attack_type == NOISE_LABEL:
        return config.noise_window_range
    if attack_type == DROP_LABEL:
        return config.drop_window_range
    raise KeyError(f"Unknown attack type: {attack_type}")


def _prepare_sensor_views(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sensor_views: dict[str, pd.DataFrame] = {}
    for sensor_id, group in df.groupby("sensor_id", sort=False):
        g = group.sort_values("timestamp").copy()
        g["sensor_pos"] = np.arange(len(g))
        sensor_views[str(sensor_id)] = g
    return sensor_views


def _init_status_maps(sensor_views: dict[str, pd.DataFrame]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    attack_available: dict[str, np.ndarray] = {}
    donor_available: dict[str, np.ndarray] = {}
    for sensor_id, group in sensor_views.items():
        size = len(group)
        attack_available[sensor_id] = np.ones(size, dtype=bool)
        donor_available[sensor_id] = np.ones(size, dtype=bool)
    return attack_available, donor_available


def _candidate_starts(mask: np.ndarray, length: int) -> list[int]:
    if len(mask) < length:
        return []
    starts: list[int] = []
    for start in range(0, len(mask) - length + 1):
        if mask[start:start + length].all():
            starts.append(start)
    return starts


def _pick_attack_segment(
    attack_type: str,
    sensor_views: dict[str, pd.DataFrame],
    attack_available: dict[str, np.ndarray],
    donor_available: dict[str, np.ndarray],
    config: AttackGenerationConfig,
    rng: np.random.Generator,
    remaining_rows: int,
) -> AttackSegment | None:
    min_len, max_len = _segment_range(config, attack_type)
    actual_max_len = min(max_len, remaining_rows)
    if actual_max_len <= 0:
        return None
    actual_min_len = min(min_len, actual_max_len)
    sensor_order = list(sensor_views.keys())
    rng.shuffle(sensor_order)

    for sensor_id in sensor_order:
        group = sensor_views[sensor_id]
        if len(group) < actual_min_len:
            continue

        candidate_lengths = list(range(actual_min_len, min(actual_max_len, len(group)) + 1))
        rng.shuffle(candidate_lengths)
        for length in candidate_lengths:
            target_starts = _candidate_starts(attack_available[sensor_id], length)
            if not target_starts:
                continue

            if attack_type != REPLAY_LABEL:
                start = int(rng.choice(target_starts))
                target_indices = group.iloc[start:start + length].index.tolist()
                return AttackSegment(
                    attack_type=attack_type,
                    sensor_id=sensor_id,
                    target_start_pos=start,
                    target_end_pos=start + length,
                    target_global_indices=target_indices,
                )

            donor_starts = _candidate_starts(donor_available[sensor_id], length)
            valid_pairs: list[tuple[int, int]] = []
            for target_start in target_starts:
                earlier_donors = [
                    donor_start
                    for donor_start in donor_starts
                    if donor_start + length <= target_start
                ]
                for donor_start in earlier_donors:
                    valid_pairs.append((target_start, donor_start))

            if not valid_pairs:
                continue

            target_start, donor_start = valid_pairs[int(rng.integers(0, len(valid_pairs)))]
            target_indices = group.iloc[target_start:target_start + length].index.tolist()
            donor_indices = group.iloc[donor_start:donor_start + length].index.tolist()
            return AttackSegment(
                attack_type=attack_type,
                sensor_id=sensor_id,
                target_start_pos=target_start,
                target_end_pos=target_start + length,
                target_global_indices=target_indices,
                donor_start_pos=donor_start,
                donor_end_pos=donor_start + length,
                donor_global_indices=donor_indices,
            )
    return None


def plan_attack_segments(df: pd.DataFrame, config: AttackGenerationConfig) -> list[AttackSegment]:
    rng = np.random.default_rng(config.random_seed)
    sensor_views = _prepare_sensor_views(df)
    attack_available, donor_available = _init_status_maps(sensor_views)
    total_rows = len(df)
    total_attack_rows = int(round(total_rows * config.attack_ratio))
    attack_types = config.attack_types
    base_quota = total_attack_rows // len(attack_types)
    remainder = total_attack_rows % len(attack_types)
    quotas = {
        attack_type: base_quota + (1 if idx < remainder else 0)
        for idx, attack_type in enumerate(attack_types)
    }

    segments: list[AttackSegment] = []
    for attack_type in attack_types:
        rows_allocated = 0
        attempts = 0
        while rows_allocated < quotas[attack_type] and attempts < 5000:
            segment = _pick_attack_segment(
                attack_type=attack_type,
                sensor_views=sensor_views,
                attack_available=attack_available,
                donor_available=donor_available,
                config=config,
                rng=rng,
                remaining_rows=quotas[attack_type] - rows_allocated,
            )
            attempts += 1
            if segment is None:
                break

            sensor_attack_mask = attack_available[segment.sensor_id]
            sensor_attack_mask[segment.target_start_pos:segment.target_end_pos] = False

            if segment.attack_type == REPLAY_LABEL and segment.donor_start_pos is not None and segment.donor_end_pos is not None:
                sensor_donor_mask = donor_available[segment.sensor_id]
                sensor_donor_mask[segment.target_start_pos:segment.target_end_pos] = False
                sensor_donor_mask[segment.donor_start_pos:segment.donor_end_pos] = False
            else:
                donor_available[segment.sensor_id][segment.target_start_pos:segment.target_end_pos] = False

            segments.append(segment)
            rows_allocated += segment.length
    return segments


def _clip_values(df: pd.DataFrame, config: AttackGenerationConfig) -> None:
    df["temperature_c"] = df["temperature_c"].clip(*config.temperature_clip)
    df["humidity_percent"] = df["humidity_percent"].clip(*config.humidity_clip)


def _apply_replay_attack(enhanced_df: pd.DataFrame, original_df: pd.DataFrame, segment: AttackSegment) -> None:
    donor_values = original_df.loc[segment.donor_global_indices, ["temperature_c", "humidity_percent"]].to_numpy()
    enhanced_df.loc[segment.target_global_indices, ["temperature_c", "humidity_percent"]] = donor_values


def _apply_injection_attack(
    enhanced_df: pd.DataFrame,
    segment: AttackSegment,
    rng: np.random.Generator,
    config: AttackGenerationConfig,
) -> None:
    direction = 1 if rng.random() > 0.35 else -1
    temp_shift = rng.uniform(10.0, 20.0) * direction
    humidity_shift = rng.uniform(20.0, 40.0) * (1 if rng.random() > 0.15 else -1)
    enhanced_df.loc[segment.target_global_indices, "temperature_c"] += temp_shift
    enhanced_df.loc[segment.target_global_indices, "humidity_percent"] += humidity_shift
    _clip_values(enhanced_df, config)


def _apply_drift_attack(
    enhanced_df: pd.DataFrame,
    segment: AttackSegment,
    rng: np.random.Generator,
    config: AttackGenerationConfig,
) -> None:
    length = segment.length
    temp_step = rng.uniform(0.05, 0.18) * (1 if rng.random() > 0.5 else -1)
    humidity_step = rng.uniform(0.08, 0.28) * (1 if rng.random() > 0.5 else -1)
    drift = np.arange(length, dtype=np.float32)
    enhanced_df.loc[segment.target_global_indices, "temperature_c"] += drift * temp_step
    enhanced_df.loc[segment.target_global_indices, "humidity_percent"] += drift * humidity_step
    _clip_values(enhanced_df, config)


def _apply_noise_attack(
    enhanced_df: pd.DataFrame,
    segment: AttackSegment,
    rng: np.random.Generator,
    config: AttackGenerationConfig,
) -> None:
    length = segment.length
    temp_noise = rng.uniform(-5.0, 5.0, size=length)
    humidity_noise = rng.uniform(-10.0, 10.0, size=length)
    enhanced_df.loc[segment.target_global_indices, "temperature_c"] += temp_noise
    enhanced_df.loc[segment.target_global_indices, "humidity_percent"] += humidity_noise
    _clip_values(enhanced_df, config)


def _apply_drop_attack(
    enhanced_df: pd.DataFrame,
    segment: AttackSegment,
    rng: np.random.Generator,
    config: AttackGenerationConfig,
) -> None:
    target_indices = np.asarray(segment.target_global_indices)
    drop_count = max(1, int(round(0.35 * len(target_indices))))
    dropped_positions = rng.choice(target_indices, size=drop_count, replace=False)
    if config.allow_nan_drop:
        enhanced_df.loc[dropped_positions, ["temperature_c", "humidity_percent"]] = np.nan
    else:
        enhanced_df.loc[dropped_positions, ["temperature_c", "humidity_percent"]] = 0.0


def apply_attack_segments(
    base_df: pd.DataFrame,
    segments: list[AttackSegment],
    config: AttackGenerationConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_seed)
    enhanced_df = base_df.copy()
    original_df = base_df.copy()
    enhanced_df["attack_type"] = NORMAL_LABEL

    for segment in segments:
        if segment.attack_type == REPLAY_LABEL:
            _apply_replay_attack(enhanced_df, original_df, segment)
        elif segment.attack_type == INJECTION_LABEL:
            _apply_injection_attack(enhanced_df, segment, rng, config)
        elif segment.attack_type == DRIFT_LABEL:
            _apply_drift_attack(enhanced_df, segment, rng, config)
        elif segment.attack_type == NOISE_LABEL:
            _apply_noise_attack(enhanced_df, segment, rng, config)
        elif segment.attack_type == DROP_LABEL:
            _apply_drop_attack(enhanced_df, segment, rng, config)
        else:
            raise KeyError(f"Unsupported attack type: {segment.attack_type}")
        enhanced_df.loc[segment.target_global_indices, "attack_type"] = segment.attack_type

    enhanced_df = enhanced_df.sort_values(["timestamp", "sensor_id"]).reset_index(drop=True)
    return enhanced_df


def build_attack_example_frame(
    enhanced_df: pd.DataFrame,
    segments: list[AttackSegment],
    context_rows: int = 8,
) -> pd.DataFrame:
    example_frames: list[pd.DataFrame] = []
    for attack_type in [REPLAY_LABEL, INJECTION_LABEL, DRIFT_LABEL, NOISE_LABEL, DROP_LABEL]:
        segment = next((item for item in segments if item.attack_type == attack_type), None)
        if segment is None:
            continue
        sensor_window = (
            enhanced_df.loc[enhanced_df["sensor_id"] == segment.sensor_id]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        window_start = max(0, segment.target_start_pos - context_rows)
        window_end = min(len(sensor_window), segment.target_end_pos + context_rows)
        sample = sensor_window.iloc[window_start:window_end].copy()
        sample["example_attack_type"] = attack_type
        example_frames.append(sample)
    if not example_frames:
        return pd.DataFrame(columns=list(enhanced_df.columns) + ["example_attack_type"])
    return pd.concat(example_frames, ignore_index=True)


def save_attack_examples_plot(
    enhanced_df: pd.DataFrame,
    segments: list[AttackSegment],
    output_path: str | Path,
    context_rows: int = 8,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    attack_order = [REPLAY_LABEL, INJECTION_LABEL, DRIFT_LABEL, NOISE_LABEL, DROP_LABEL]
    plotted_segments = [segment for attack_type in attack_order for segment in segments if segment.attack_type == attack_type]
    if not plotted_segments:
        return

    plotted_segments = plotted_segments[: len(attack_order)]
    fig, axes = plt.subplots(len(plotted_segments), 1, figsize=(12, 3.5 * len(plotted_segments)), sharex=False)
    if len(plotted_segments) == 1:
        axes = [axes]

    for ax, segment in zip(axes, plotted_segments):
        sensor_window = (
            enhanced_df.loc[enhanced_df["sensor_id"] == segment.sensor_id]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        start = max(0, segment.target_start_pos - context_rows)
        end = min(len(sensor_window), segment.target_end_pos + context_rows)
        window = sensor_window.iloc[start:end].copy()
        target_pos_set = set(range(segment.target_start_pos, segment.target_end_pos))
        target_mask = window.index.isin(target_pos_set)
        ax.plot(window["timestamp"], window["temperature_c"], label="temperature_c", color="tab:red")
        ax.plot(window["timestamp"], window["humidity_percent"], label="humidity_percent", color="tab:blue")
        ax.scatter(window.loc[target_mask, "timestamp"], window.loc[target_mask, "temperature_c"], color="darkred", s=20)
        ax.scatter(window.loc[target_mask, "timestamp"], window.loc[target_mask, "humidity_percent"], color="navy", s=20)
        ax.set_title(f"{segment.attack_type} | sensor={segment.sensor_id}")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def summarize_attack_generation(enhanced_df: pd.DataFrame, segments: list[AttackSegment]) -> dict[str, Any]:
    return {
        "row_count": int(len(enhanced_df)),
        "attack_type_counts": enhanced_df["attack_type"].value_counts(dropna=False).to_dict(),
        "segment_count": len(segments),
        "segments_by_type": {
            attack_type: sum(1 for segment in segments if segment.attack_type == attack_type)
            for attack_type in enhanced_df["attack_type"].unique()
        },
    }


def generate_enhanced_iot_dataset(
    config: AttackGenerationConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], list[AttackSegment]]:
    config = config or AttackGenerationConfig()
    base_df = load_base_dataset(config.input_path)
    segments = plan_attack_segments(base_df, config)
    enhanced_df = apply_attack_segments(base_df, segments, config)
    summary = summarize_attack_generation(enhanced_df, segments)
    return enhanced_df, summary, segments


def save_enhanced_outputs(
    enhanced_df: pd.DataFrame,
    segments: list[AttackSegment],
    config: AttackGenerationConfig,
) -> None:
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enhanced_df.to_csv(output_path, index=False)

    example_csv = build_attack_example_frame(enhanced_df, segments)
    if not example_csv.empty:
        example_csv.to_csv(Path(config.example_csv_path), index=False)
        save_attack_examples_plot(enhanced_df, segments, config.example_plot_path)
