from __future__ import annotations

import csv
import hashlib
import json
import math
import queue
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import paho.mqtt.client as mqtt
except ImportError:  # pragma: no cover - optional dependency at runtime
    mqtt = None


NORMAL_LABEL = "normal"
UNKNOWN_LABEL = "unknown"
ENVIRONMENTAL_LABEL = "environmental_anomaly"
REPLAY_LABEL = "replay_attack"


class DataConfig:
    def __init__(
        self,
        window_size: int = 60,
        data_path: str = "enhanced_iot_dataset.csv",
        consistency_window: int = 5,
        random_seed: int = 42,
    ) -> None:
        self.window_size = window_size
        self.data_path = data_path
        self.consistency_window = consistency_window
        self.random_seed = random_seed

    def resolve(self) -> Path:
        return Path(self.data_path).expanduser().resolve()


@dataclass
class ModelConfig:
    hidden_size: int = 32
    latent_size: int = 16
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 20
    patience: int = 5
    validation_fraction: float = 0.2
    device: str = "cpu"


@dataclass
class ReplayConfig:
    history_size: int = 100
    compare_last_n_windows: int = 50
    similarity_threshold: float = 0.94
    max_mean_abs_distance: float = 0.14
    min_gap_windows: int = 30
    min_gap_seconds: float = 0.0
    rounding_decimals: int = 2


@dataclass
class MinMaxScalerLite:
    feature_names: list[str]
    data_min_: np.ndarray | None = None
    data_max_: np.ndarray | None = None

    def fit(self, values: pd.DataFrame | np.ndarray) -> "MinMaxScalerLite":
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != len(self.feature_names):
            raise ValueError("Scaler expects a 2D array with one column per feature.")
        self.data_min_ = np.nanmin(array, axis=0)
        self.data_max_ = np.nanmax(array, axis=0)
        return self

    @property
    def scale_(self) -> np.ndarray:
        if self.data_min_ is None or self.data_max_ is None:
            raise ValueError("Scaler has not been fit yet.")
        scale = self.data_max_ - self.data_min_
        scale[scale == 0] = 1.0
        return scale

    def transform(self, values: pd.DataFrame | np.ndarray, clip: bool = True) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        scaled = (array - self.data_min_) / self.scale_
        if clip:
            scaled = np.clip(scaled, 0.0, 1.0)
        return scaled

    def transform_frame(self, frame: pd.DataFrame, clip: bool = True) -> pd.DataFrame:
        transformed = self.transform(frame[self.feature_names].to_numpy(dtype=np.float32), clip=clip)
        out = frame.copy()
        out.loc[:, self.feature_names] = transformed
        return out


@dataclass
class SequenceBundle:
    X: np.ndarray
    labels: np.ndarray
    metadata: pd.DataFrame
    feature_cols: list[str]


@dataclass
class ReplayResult:
    is_replay: bool
    similarity: float | None = None
    matched_history_index: int | None = None
    matched_timestamp: pd.Timestamp | None = None
    reason: str | None = None


@dataclass
class DetectionResult:
    timestamp: pd.Timestamp
    source: str
    predicted_label: str
    reconstruction_error: float | None
    threshold: float | None
    replay_flag: bool
    replay_similarity: float | None = None
    matched_history_index: int | None = None
    replay_reason: str | None = None
    anomaly_flag: bool = False
    status: str = "ok"
    message: str | None = None

    def to_log_record(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat() if pd.notna(self.timestamp) else None,
            "source": self.source,
            "predicted_label": self.predicted_label,
            "reconstruction_error": self.reconstruction_error,
            "threshold": self.threshold,
            "replay_flag": self.replay_flag,
            "replay_similarity": self.replay_similarity,
            "matched_history_index": self.matched_history_index,
            "replay_reason": self.replay_reason,
            "anomaly_flag": self.anomaly_flag,
            "status": self.status,
            "message": self.message,
        }


@dataclass
class HistoryWindow:
    sequence: np.ndarray
    sequence_hash: str
    timestamp: pd.Timestamp
    history_index: int


@dataclass
class StreamingSourceState:
    source: str
    model: Any
    scaler: MinMaxScalerLite
    feature_cols: list[str]
    window_size: int
    threshold: float
    replay_config: ReplayConfig
    consistency_window: int = 5
    raw_buffer: Deque[dict[str, Any]] = field(default_factory=deque)
    feature_window: Deque[np.ndarray] = field(default_factory=deque)
    temp_deltas: Deque[float] = field(default_factory=deque)
    humidity_deltas: Deque[float] = field(default_factory=deque)
    time_diffs: Deque[float] = field(default_factory=deque)
    history_buffer: Deque[HistoryWindow] = field(default_factory=deque)
    total_windows_seen: int = 0

    def __post_init__(self) -> None:
        self.raw_buffer = deque(maxlen=max(self.window_size + self.consistency_window, self.window_size + 1))
        self.feature_window = deque(maxlen=self.window_size)
        self.temp_deltas = deque(maxlen=self.consistency_window)
        self.humidity_deltas = deque(maxlen=self.consistency_window)
        self.time_diffs = deque(maxlen=self.consistency_window)
        self.history_buffer = deque(maxlen=self.replay_config.history_size)


@dataclass
class TrainedHybridIDS:
    model: Any
    feature_cols: list[str]
    scalers: dict[str, MinMaxScalerLite]
    threshold_main: float
    threshold_loose: float
    window_size: int
    replay_config: ReplayConfig
    consistency_window: int
    training_history: dict[str, list[float]]


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        latent_size: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1 or num_layers > 2:
            raise ValueError("num_layers must be 1 or 2 to keep the model lightweight.")
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.decoder = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        latent = torch.tanh(self.to_latent(encoded[:, -1, :]))
        repeated = latent.unsqueeze(1).repeat(1, x.shape[1], 1)
        decoded, _ = self.decoder(repeated)
        return self.output_layer(decoded)


class AuditLogger:
    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = [
            "timestamp",
            "source",
            "predicted_label",
            "reconstruction_error",
            "threshold",
            "replay_flag",
            "replay_similarity",
            "matched_history_index",
            "replay_reason",
            "anomaly_flag",
            "status",
            "message",
        ]

    def write(self, result: DetectionResult) -> None:
        record = result.to_log_record()
        file_exists = self.output_path.exists()
        with self.output_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)


def _clean_sensor_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out = out.dropna(subset=["timestamp", "temperature_c", "humidity_percent", "source", "attack_type"])
    out = out[(out["temperature_c"].between(-50, 100)) & (out["humidity_percent"].between(0, 100))]
    out = out.sort_values(["source", "timestamp"]).reset_index(drop=True)
    return out


def load_data(config: DataConfig | None = None) -> pd.DataFrame:
    config = config or DataConfig()
    df = pd.read_csv(config.resolve())
    required_columns = {"timestamp", "sensor_id", "temperature_c", "humidity_percent", "attack_type"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["sensor_id"] = df["sensor_id"].astype(str)
    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    df["humidity_percent"] = pd.to_numeric(df["humidity_percent"], errors="coerce")
    df["attack_type"] = df["attack_type"].fillna(NORMAL_LABEL).astype(str)
    df["label"] = df["attack_type"].apply(lambda value: 0 if value == NORMAL_LABEL else 1)
    df["class_label"] = df["attack_type"]
    df["source"] = df["sensor_id"]
    return _clean_sensor_frame(df)


def engineer_features(df: pd.DataFrame, consistency_window: int = 5) -> pd.DataFrame:
    feature_frames: list[pd.DataFrame] = []
    for source, group in df.groupby("source", sort=False):
        g = group.sort_values("timestamp").copy()
        g["time_diff"] = g["timestamp"].diff().dt.total_seconds().fillna(0.0).clip(lower=0.0)
        g["temperature_delta"] = g["temperature_c"].diff().fillna(0.0)
        g["humidity_delta"] = g["humidity_percent"].diff().fillna(0.0)
        safe_time_diff = g["time_diff"].replace(0.0, np.nan)
        g["temperature_rate"] = (g["temperature_delta"] / safe_time_diff).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        g["humidity_rate"] = (g["humidity_delta"] / safe_time_diff).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        g["temperature_consistency"] = (
            g["temperature_delta"].abs().rolling(consistency_window, min_periods=1).mean().fillna(0.0)
        )
        g["humidity_consistency"] = (
            g["humidity_delta"].abs().rolling(consistency_window, min_periods=1).mean().fillna(0.0)
        )
        g["interval_consistency"] = (
            g["time_diff"].rolling(consistency_window, min_periods=1).std(ddof=0).fillna(0.0)
        )
        g["source"] = source
        feature_frames.append(g)
    return pd.concat(feature_frames, ignore_index=True)


def get_feature_columns() -> list[str]:
    return [
        "temperature_c",
        "humidity_percent",
    ]


def normalize_per_source(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, MinMaxScalerLite]]:
    normalized_frames: list[pd.DataFrame] = []
    scalers: dict[str, MinMaxScalerLite] = {}
    for source, group in df.groupby("source", sort=False):
        scaler = MinMaxScalerLite(feature_names=list(feature_cols))
        normal_mask = group["attack_type"].eq(NORMAL_LABEL)
        fit_frame = group.loc[normal_mask, feature_cols] if normal_mask.any() else group.loc[:, feature_cols]
        scaler.fit(fit_frame.to_numpy(dtype=np.float32))
        scaled = scaler.transform_frame(group, clip=True)
        normalized_frames.append(scaled)
        scalers[source] = scaler
    normalized = pd.concat(normalized_frames, ignore_index=True)
    normalized = normalized.sort_values(["source", "timestamp"]).reset_index(drop=True)
    return normalized, scalers


def _window_label(class_labels: pd.Series) -> str:
    if class_labels.empty:
        return UNKNOWN_LABEL
    return str(class_labels.iloc[-1])


def make_sequences(
    df: pd.DataFrame,
    window_size: int = 60,
    feature_cols: Sequence[str] | None = None,
) -> SequenceBundle:
    feature_cols = list(feature_cols or get_feature_columns())
    sequences: list[np.ndarray] = []
    labels: list[str] = []
    metadata_rows: list[dict[str, Any]] = []

    for source, group in df.groupby("source", sort=False):
        g = group.sort_values("timestamp").reset_index(drop=True)
        if len(g) < window_size:
            continue
        for start in range(0, len(g) - window_size + 1):
            end = start + window_size
            window = g.iloc[start:end]
            sequences.append(window[feature_cols].to_numpy(dtype=np.float32))
            labels.append(_window_label(window["class_label"]))
            metadata_rows.append(
                {
                    "source": source,
                    "start_timestamp": window["timestamp"].iloc[0],
                    "end_timestamp": window["timestamp"].iloc[-1],
                    "window_index": len(metadata_rows),
                    "attack_type": window["attack_type"].iloc[-1],
                    "class_label": window["class_label"].iloc[-1],
                    "label": int(window["label"].iloc[-1]),
                    "effective_span_seconds": (
                        window["timestamp"].iloc[-1] - window["timestamp"].iloc[0]
                    ).total_seconds(),
                }
            )

    if not sequences:
        raise ValueError("No sequences were generated. Check the window size and source lengths.")

    metadata = pd.DataFrame(metadata_rows)
    return SequenceBundle(
        X=np.asarray(sequences, dtype=np.float32),
        labels=np.asarray(labels, dtype=object),
        metadata=metadata,
        feature_cols=feature_cols,
    )


def _subset_bundle(bundle: SequenceBundle, mask: np.ndarray) -> SequenceBundle:
    return SequenceBundle(
        X=bundle.X[mask],
        labels=bundle.labels[mask],
        metadata=bundle.metadata.loc[mask].reset_index(drop=True),
        feature_cols=bundle.feature_cols,
    )


def _split_normal_sequences(
    bundle: SequenceBundle,
    validation_fraction: float,
) -> tuple[SequenceBundle, SequenceBundle]:
    train_masks: list[np.ndarray] = []
    val_masks: list[np.ndarray] = []
    for source in bundle.metadata["source"].unique():
        source_mask = bundle.metadata["source"].eq(source).to_numpy()
        indices = np.where(source_mask)[0]
        if len(indices) < 2:
            train_masks.append(indices)
            continue
        split_idx = max(1, int(math.floor(len(indices) * (1.0 - validation_fraction))))
        split_idx = min(split_idx, len(indices) - 1)
        train_masks.append(indices[:split_idx])
        val_masks.append(indices[split_idx:])

    train_idx = np.concatenate(train_masks) if train_masks else np.array([], dtype=int)
    val_idx = np.concatenate(val_masks) if val_masks else np.array([], dtype=int)
    if len(val_idx) == 0:
        val_idx = train_idx[-max(1, min(10, len(train_idx))):]
        train_idx = train_idx[: max(1, len(train_idx) - len(val_idx))]
    return _subset_bundle(bundle, train_idx), _subset_bundle(bundle, val_idx)


def build_lstm_autoencoder(
    timesteps: int,
    n_features: int,
    model_config: ModelConfig | None = None,
) -> LSTMAutoencoder:
    model_config = model_config or ModelConfig()
    return LSTMAutoencoder(
        input_size=n_features,
        hidden_size=model_config.hidden_size,
        latent_size=model_config.latent_size,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
    )


def train_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    model_config: ModelConfig | None = None,
) -> tuple[LSTMAutoencoder, dict[str, list[float]]]:
    model_config = model_config or ModelConfig()
    device = torch.device(model_config.device)
    model = build_lstm_autoencoder(X_train.shape[1], X_train.shape[2], model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config.learning_rate)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=model_config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
        batch_size=model_config.batch_size,
        shuffle=False,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for _epoch in range(model_config.epochs):
        model.train()
        train_losses: list[float] = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                reconstruction = model(batch)
                loss = loss_fn(reconstruction, batch)
                val_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= model_config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def reconstruct_sequences(model: Any, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X), dtype=np.float32)

    if isinstance(model, nn.Module):
        model.eval()
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                batch = torch.tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
                outputs.append(model(batch).cpu().numpy())
        return np.concatenate(outputs, axis=0) if outputs else np.empty_like(X)

    if callable(model):
        return np.asarray(model(X), dtype=np.float32)

    raise TypeError("Model must be a torch.nn.Module, have a predict method, or be callable.")


def compute_reconstruction_error(model: Any, X: np.ndarray) -> np.ndarray:
    reconstructed = reconstruct_sequences(model, X)
    return np.mean(np.square(X - reconstructed), axis=(1, 2))


def estimate_threshold(errors: np.ndarray, sigma: int = 3) -> float:
    errors = np.asarray(errors, dtype=np.float32)
    return float(np.mean(errors) + sigma * np.std(errors))


def _sequence_hash(sequence: np.ndarray, decimals: int) -> str:
    rounded = np.round(np.asarray(sequence, dtype=np.float32), decimals=decimals)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def _window_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    return float(max(0.0, 1.0 - (np.linalg.norm(a - b) / a.size)))


def detect_replay(
    sequence: np.ndarray,
    history_buffer: Sequence[HistoryWindow],
    replay_config: ReplayConfig,
    current_timestamp: pd.Timestamp | None = None,
) -> ReplayResult:
    if not history_buffer:
        return ReplayResult(is_replay=False)

    rounded_hash = _sequence_hash(sequence, replay_config.rounding_decimals)
    history_list = list(history_buffer)[-replay_config.compare_last_n_windows:]
    if replay_config.min_gap_windows <= 0:
        eligible = history_list
    elif len(history_list) > replay_config.min_gap_windows:
        eligible = history_list[:-replay_config.min_gap_windows]
    else:
        eligible = []

    if current_timestamp is not None and replay_config.min_gap_seconds > 0:
        eligible = [
            item
            for item in eligible
            if abs((current_timestamp - item.timestamp).total_seconds()) >= replay_config.min_gap_seconds
        ]

    for item in eligible:
        if item.sequence_hash == rounded_hash:
            return ReplayResult(
                is_replay=True,
                similarity=1.0,
                matched_history_index=item.history_index,
                matched_timestamp=item.timestamp,
                reason="exact_hash_match",
            )

    best_similarity = -1.0
    best_item: HistoryWindow | None = None
    for item in eligible:
        similarity = _window_similarity(sequence, item.sequence)
        mean_abs_distance = float(np.mean(np.abs(sequence - item.sequence)))
        if (
            similarity > (replay_config.similarity_threshold - 0.01)
            and mean_abs_distance <= replay_config.max_mean_abs_distance
        ):
            return ReplayResult(
                is_replay=True,
                similarity=float(similarity),
                matched_history_index=item.history_index,
                matched_timestamp=item.timestamp,
                reason="similarity_match",
            )
        if similarity > best_similarity:
            best_similarity = similarity
            best_item = item

    return ReplayResult(is_replay=False, similarity=float(best_similarity) if best_similarity >= 0 else None)


def classify_window(
    sequence: np.ndarray,
    model: Any,
    threshold: float,
    history_buffer: Sequence[HistoryWindow],
    replay_config: ReplayConfig | None = None,
    timestamp: pd.Timestamp | None = None,
    source: str = "unknown",
) -> DetectionResult:
    replay_config = replay_config or ReplayConfig()
    replay_result = detect_replay(sequence, history_buffer, replay_config, current_timestamp=timestamp)
    reconstruction_error = float(compute_reconstruction_error(model, sequence[None, ...])[0])
    anomaly_flag = reconstruction_error > threshold

    if replay_result.is_replay:
        if reconstruction_error < threshold:
            predicted_label = "Replay Attack"
        else:
            predicted_label = "Environmental Anomaly"
    elif anomaly_flag:
        predicted_label = "Environmental Anomaly"
    else:
        predicted_label = "Normal"

    return DetectionResult(
        timestamp=timestamp or pd.Timestamp.utcnow(),
        source=source,
        predicted_label=predicted_label,
        reconstruction_error=reconstruction_error,
        threshold=threshold,
        replay_flag=replay_result.is_replay,
        replay_similarity=replay_result.similarity,
        matched_history_index=replay_result.matched_history_index,
        replay_reason=replay_result.reason,
        anomaly_flag=anomaly_flag,
    )


def _append_history_window(
    history_buffer: Deque[HistoryWindow],
    sequence: np.ndarray,
    timestamp: pd.Timestamp,
    replay_config: ReplayConfig,
    history_index: int,
) -> None:
    history_buffer.append(
        HistoryWindow(
            sequence=np.asarray(sequence, dtype=np.float32).copy(),
            sequence_hash=_sequence_hash(sequence, replay_config.rounding_decimals),
            timestamp=timestamp,
            history_index=history_index,
        )
    )


def create_stream_state(
    source: str,
    detector: TrainedHybridIDS,
) -> StreamingSourceState:
    if source not in detector.scalers:
        raise KeyError(f"No scaler available for source '{source}'.")
    return StreamingSourceState(
        source=source,
        model=detector.model,
        scaler=detector.scalers[source],
        feature_cols=detector.feature_cols,
        window_size=detector.window_size,
        threshold=detector.threshold_main,
        replay_config=detector.replay_config,
        consistency_window=detector.consistency_window,
    )


def _compute_stream_feature_row(reading: dict[str, Any], state: StreamingSourceState) -> np.ndarray:
    timestamp = pd.to_datetime(reading["timestamp"])
    temperature = float(reading.get("temperature_c", reading.get("temperature")))
    humidity = float(reading.get("humidity_percent", reading.get("humidity")))
    previous = state.raw_buffer[-1] if state.raw_buffer else None
    if previous is None:
        time_diff = 0.0
        temperature_delta = 0.0
        humidity_delta = 0.0
    else:
        time_diff = max(0.0, float((timestamp - previous["timestamp"]).total_seconds()))
        temperature_delta = temperature - previous["temperature"]
        humidity_delta = humidity - previous["humidity"]

    safe_time_diff = time_diff if time_diff > 0 else np.nan
    temperature_rate = 0.0 if np.isnan(safe_time_diff) else temperature_delta / safe_time_diff
    humidity_rate = 0.0 if np.isnan(safe_time_diff) else humidity_delta / safe_time_diff

    state.temp_deltas.append(float(abs(temperature_delta)))
    state.humidity_deltas.append(float(abs(humidity_delta)))
    state.time_diffs.append(float(time_diff))

    feature_row = {
        "temperature_c": temperature,
        "humidity_percent": humidity,
        "temperature_delta": temperature_delta,
        "humidity_delta": humidity_delta,
        "temperature_rate": temperature_rate,
        "humidity_rate": humidity_rate,
        "time_diff": time_diff,
        "temperature_consistency": float(np.mean(state.temp_deltas)) if state.temp_deltas else 0.0,
        "humidity_consistency": float(np.mean(state.humidity_deltas)) if state.humidity_deltas else 0.0,
        "interval_consistency": float(np.std(state.time_diffs)) if state.time_diffs else 0.0,
    }
    frame = pd.DataFrame([feature_row], columns=state.feature_cols)
    scaled = state.scaler.transform(frame.to_numpy(dtype=np.float32))[0]
    state.raw_buffer.append({"timestamp": timestamp, "temperature": temperature, "humidity": humidity})
    return scaled


def stream_infer(
    reading: dict[str, Any],
    source_state: StreamingSourceState,
    audit_logger: AuditLogger | None = None,
) -> DetectionResult | None:
    try:
        scaled_feature_row = _compute_stream_feature_row(reading, source_state)
        source_state.feature_window.append(scaled_feature_row)
        timestamp = pd.to_datetime(reading["timestamp"])

        if len(source_state.feature_window) < source_state.window_size:
            return None

        sequence = np.asarray(source_state.feature_window, dtype=np.float32)
        result = classify_window(
            sequence=sequence,
            model=source_state.model,
            threshold=source_state.threshold,
            history_buffer=list(source_state.history_buffer),
            replay_config=source_state.replay_config,
            timestamp=timestamp,
            source=source_state.source,
        )
        source_state.total_windows_seen += 1

        if result.predicted_label == "Normal":
            _append_history_window(
                source_state.history_buffer,
                sequence,
                timestamp,
                source_state.replay_config,
                source_state.total_windows_seen,
            )

        if audit_logger is not None:
            audit_logger.write(result)
        return result
    except Exception as exc:  # pragma: no cover - fail-safe branch
        fallback = DetectionResult(
            timestamp=pd.to_datetime(reading.get("timestamp", pd.Timestamp.utcnow())),
            source=source_state.source,
            predicted_label="Normal",
            reconstruction_error=None,
            threshold=source_state.threshold,
            replay_flag=False,
            anomaly_flag=False,
            status="fail_safe",
            message=str(exc),
        )
        if audit_logger is not None:
            audit_logger.write(fallback)
        return fallback


def inject_synthetic_replay(
    eval_bundle: SequenceBundle,
    donor_bundle: SequenceBundle,
    replay_fraction: float,
    random_seed: int = 42,
) -> SequenceBundle:
    rng = np.random.default_rng(random_seed)
    X = eval_bundle.X.copy()
    labels = eval_bundle.labels.copy()
    metadata = eval_bundle.metadata.copy()
    metadata["synthetic_replay_from"] = pd.Series([None] * len(metadata), dtype=object)

    for source in metadata["source"].unique():
        target_idx = metadata.index[(metadata["source"] == source) & (metadata["label"] == NORMAL_LABEL)].to_numpy()
        donor_idx = donor_bundle.metadata.index[
            (donor_bundle.metadata["source"] == source) & (donor_bundle.metadata["label"] == NORMAL_LABEL)
        ].to_numpy()
        if len(target_idx) == 0 or len(donor_idx) == 0:
            continue
        sample_count = max(1, int(len(target_idx) * replay_fraction))
        chosen_targets = rng.choice(target_idx, size=min(sample_count, len(target_idx)), replace=False)
        chosen_donors = rng.choice(donor_idx, size=len(chosen_targets), replace=True)
        for target, donor in zip(chosen_targets, chosen_donors):
            X[target] = donor_bundle.X[donor].copy()
            labels[target] = REPLAY_LABEL
            metadata.loc[target, "label"] = REPLAY_LABEL
            metadata.loc[target, "synthetic_replay_from"] = int(donor)

    return SequenceBundle(X=X, labels=labels, metadata=metadata, feature_cols=eval_bundle.feature_cols)


def summarize_sampling_intervals(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, group in df.groupby("source", sort=False):
        diffs = group["timestamp"].diff().dt.total_seconds().dropna()
        median_interval = float(diffs.median()) if not diffs.empty else 0.0
        rows.append(
            {
                "source": source,
                "median_interval_seconds": median_interval,
            }
        )
    return pd.DataFrame(rows)


def iter_csv_stream(path: str | Path, sleep_seconds: float = 0.0) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    required = {"timestamp", "sensor_id", "temperature_c", "humidity_percent"}
    if not required.issubset(frame.columns):
        raise ValueError(f"CSV stream requires columns {sorted(required)}.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["sensor_id"] = frame["sensor_id"].astype(str)

    for row in frame.itertuples(index=False):
        reading = {
            "timestamp": getattr(row, "timestamp"),
            "temperature_c": getattr(row, "temperature_c"),
            "humidity_percent": getattr(row, "humidity_percent"),
            "source": getattr(row, "sensor_id"),
        }
        yield reading
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def mqtt_stream(
    host: str,
    port: int,
    topic: str,
    parser: Callable[[bytes], dict[str, Any]] | None = None,
    keepalive: int = 60,
    timeout_seconds: float = 1.0,
) -> Iterator[dict[str, Any]]:
    if mqtt is None:  # pragma: no cover - depends on optional import
        raise ImportError("paho-mqtt is required for MQTT streaming support.")

    parser = parser or (lambda payload: json.loads(payload.decode("utf-8")))
    q: queue.Queue[dict[str, Any]] = queue.Queue()
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    def on_message(_client: Any, _userdata: Any, msg: Any) -> None:
        q.put(parser(msg.payload))

    client.on_message = on_message
    client.connect(host, port, keepalive=keepalive)
    client.subscribe(topic)
    client.loop_start()
    try:
        while True:
            yield q.get(timeout=timeout_seconds)
    finally:  # pragma: no cover - external resource cleanup
        client.loop_stop()
        client.disconnect()


def _plot_loss(history: dict[str, list[float]], output_path: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="validation")
    ax.set_title("LSTM Autoencoder Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    fig.tight_layout()
    if output_path is None:
        plt.close(fig)
        return
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_error_distribution(
    val_errors: np.ndarray,
    eval_errors: np.ndarray,
    threshold_loose: float,
    threshold_main: float,
    output_path: Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(val_errors, bins=40, alpha=0.6, label="normal validation", density=True)
    ax.hist(eval_errors, bins=40, alpha=0.4, label="evaluation", density=True)
    ax.axvline(threshold_loose, color="orange", linestyle="--", label="mean + 2*std")
    ax.axvline(threshold_main, color="red", linestyle="--", label="mean + 3*std")
    ax.set_title("Reconstruction Error Distribution")
    ax.set_xlabel("MSE")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    if output_path is None:
        plt.close(fig)
        return
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _seed_history_buffers(
    bundle: SequenceBundle,
    replay_config: ReplayConfig,
) -> dict[str, Deque[HistoryWindow]]:
    buffers: dict[str, Deque[HistoryWindow]] = defaultdict(lambda: deque(maxlen=replay_config.history_size))
    for idx, row in bundle.metadata.iterrows():
        _append_history_window(
            buffers[row["source"]],
            bundle.X[idx],
            row["end_timestamp"],
            replay_config,
            history_index=int(row["window_index"]),
        )
    return buffers


def run_demo_pipeline(
    data_config: DataConfig | None = None,
    model_config: ModelConfig | None = None,
    replay_config: ReplayConfig | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    data_config = data_config or DataConfig()
    model_config = model_config or ModelConfig()
    replay_config = replay_config or ReplayConfig(min_gap_windows=max(5, data_config.window_size // 3))
    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    merged = load_data(data_config)
    interval_summary = summarize_sampling_intervals(merged)
    engineered = engineer_features(merged, consistency_window=data_config.consistency_window)
    feature_cols = get_feature_columns()
    normalized, scalers = normalize_per_source(engineered, feature_cols)
    bundle = make_sequences(normalized, window_size=data_config.window_size, feature_cols=feature_cols)

    labeled_bundle = bundle
    normal_mask = labeled_bundle.labels == NORMAL_LABEL
    if int(np.sum(normal_mask)) < 10:
        raise ValueError("Not enough normal sequences to train the LSTM autoencoder.")

    normal_bundle = _subset_bundle(labeled_bundle, normal_mask)
    train_bundle, val_bundle = _split_normal_sequences(normal_bundle, model_config.validation_fraction)
    if len(train_bundle.X) == 0 or len(val_bundle.X) == 0:
        raise ValueError("Training or validation split is empty. Reduce validation_fraction or window size.")

    model, history = train_autoencoder(train_bundle.X, val_bundle.X, model_config)
    val_errors = compute_reconstruction_error(model, val_bundle.X)
    threshold_loose = estimate_threshold(val_errors, sigma=2)
    threshold_main = estimate_threshold(val_errors, sigma=3)

    eval_bundle = labeled_bundle
    eval_errors = compute_reconstruction_error(model, eval_bundle.X)

    history_buffers = _seed_history_buffers(train_bundle, replay_config)
    predictions: list[str] = []
    audit_rows: list[dict[str, Any]] = []
    for idx, row in eval_bundle.metadata.iterrows():
        source = row["source"]
        result = classify_window(
            sequence=eval_bundle.X[idx],
            model=model,
            threshold=threshold_main,
            history_buffer=list(history_buffers[source]),
            replay_config=replay_config,
            timestamp=row["end_timestamp"],
            source=source,
        )
        predictions.append(result.predicted_label)
        audit_rows.append(result.to_log_record())
        if result.predicted_label == "Normal":
            _append_history_window(
                history_buffers[source],
                eval_bundle.X[idx],
                row["end_timestamp"],
                replay_config,
                history_index=int(row["window_index"]),
            )

    detector = TrainedHybridIDS(
        model=model,
        feature_cols=feature_cols,
        scalers=scalers,
        threshold_main=threshold_main,
        threshold_loose=threshold_loose,
        window_size=data_config.window_size,
        replay_config=replay_config,
        consistency_window=data_config.consistency_window,
        training_history=history,
    )

    if output_path is not None:
        _plot_loss(history, output_path / "loss_curve.png")
        _plot_error_distribution(
            val_errors,
            eval_errors,
            threshold_loose,
            threshold_main,
            output_path / "reconstruction_error_distribution.png",
        )
        audit_logger = AuditLogger(output_path / "audit_log.csv")
        for record in audit_rows:
            audit_logger.write(
                DetectionResult(
                    timestamp=pd.to_datetime(record["timestamp"]),
                    source=record["source"],
                    predicted_label=record["predicted_label"],
                    reconstruction_error=record["reconstruction_error"],
                    threshold=record["threshold"],
                    replay_flag=record["replay_flag"],
                    replay_similarity=record["replay_similarity"],
                    matched_history_index=record["matched_history_index"],
                    replay_reason=record["replay_reason"],
                    anomaly_flag=record["anomaly_flag"],
                    status=record["status"],
                    message=record["message"],
                )
            )

    demo_source = merged["source"].iloc[0]
    source_state = create_stream_state(demo_source, detector)
    demo_rows = merged.loc[merged["source"] == demo_source, ["timestamp", "temperature_c", "humidity_percent", "source"]].head(
        data_config.window_size + 10
    )
    streaming_results = []
    for row in demo_rows.to_dict("records"):
        stream_row = {
            "timestamp": row["timestamp"],
            "temperature_c": row["temperature_c"],
            "humidity_percent": row["humidity_percent"],
            "source": row["source"],
        }
        result = stream_infer(stream_row, source_state)
        if result is not None:
            streaming_results.append(result.to_log_record())

    label_counts = eval_bundle.metadata["attack_type"].value_counts().to_dict()
    prediction_counts = pd.Series(predictions).value_counts().to_dict()

    return {
        "detector": detector,
        "interval_summary": interval_summary,
        "label_counts": label_counts,
        "prediction_counts": prediction_counts,
        "thresholds": {
            "mean_plus_2_std": threshold_loose,
            "mean_plus_3_std": threshold_main,
        },
        "validation_errors": val_errors,
        "evaluation_errors": eval_errors,
        "streaming_results": streaming_results,
        "output_dir": str(output_path) if output_path is not None else None,
    }
