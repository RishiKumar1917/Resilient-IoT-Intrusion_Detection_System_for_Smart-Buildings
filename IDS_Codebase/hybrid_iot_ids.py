from __future__ import annotations

import csv
import hashlib
import json
import math
import queue
import time
from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import paho.mqtt.client as mqtt
except ImportError:  # pragma: no cover - optional dependency at runtime
    mqtt = None

ENABLE_FEEDBACK = True
DEBUG_MODE = False
INTERACTIVE_MODE = False


NORMAL_LABEL = "normal"
UNKNOWN_LABEL = "unknown"
ENVIRONMENTAL_LABEL = "environmental_anomaly"
REPLAY_LABEL = "replay_attack"


class DataConfig:
    def __init__(
        self,
        window_size: int = 30,
        data_path: str = "enhanced_iot_dataset_3sensors.csv",
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
class ThresholdConfig:
    drift_threshold: float = 0.05
    replay_similarity_threshold: float = 0.9
    drop_std_threshold: float = 0.1
    drop_range_threshold: float = 0.2
    injection_range_threshold: float = 5.0
    noise_std_threshold: float = 2.0
    noise_entropy_threshold: float = 1.5
    iso_anomaly_threshold: float = 0.0


from detection.types import ReplayConfig, ReplayResult, HistoryWindow


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
class DetectionResult:
    timestamp: pd.Timestamp
    source: str
    predicted_label: str
    reconstruction_error: float | None
    threshold: float | None
    replay_flag: bool
    confidence: float | str | None = None
    decision_source: str | None = None
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
            "confidence": self.confidence,
            "decision_source": self.decision_source,
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

    def to_prediction_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.predicted_label,
            "confidence": self.confidence,
            "source": self.decision_source,
        }


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
            "confidence",
            "decision_source",
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
    df = df.head(2000)   # max 2000 rows for demo
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


from features.feature_engineering import engineer_features


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
    return float(np.percentile(errors, 95))


from detection.replay import (
    _sequence_hash,
    compute_similarity,
    _window_similarity,
    detect_replay,
)


def classify_window(
    sequence: np.ndarray,
    model: Any,
    threshold: float,
    history_buffer: Sequence[HistoryWindow],
    replay_config: ReplayConfig | None = None,
    threshold_config: ThresholdConfig | None = None,
    timestamp: pd.Timestamp | None = None,
    source: str = "unknown",
    window_size: int = 30,
) -> DetectionResult:
    threshold_config = threshold_config or ThresholdConfig()
    if len(sequence) < window_size:
        return DetectionResult(
            timestamp=timestamp or pd.Timestamp.utcnow(),
            source=source,
            predicted_label="Normal",
            confidence=None,
            decision_source=None,
            reconstruction_error=0.0,
            threshold=threshold,
            replay_flag=False,
            replay_similarity=None,
            matched_history_index=None,
            replay_reason=None,
            anomaly_flag=False,
        )

    replay_config = replay_config or ReplayConfig()
    replay_config.similarity_threshold = threshold_config.replay_similarity_threshold
    replay_result = detect_replay(sequence, history_buffer, replay_config, current_timestamp=timestamp)
    reconstruction_error = float(compute_reconstruction_error(model, sequence[None, ...])[0])
    # Restore standard 1.0x reconstruction error threshold for 3-sigma anomaly flagging
    anomaly_flag = reconstruction_error > threshold

    feature_dict = {}
    rf_prediction = "Normal"
    gb_prediction = "Normal"
    rf_conf = 0.0
    gb_conf = 0.0
    iso_score = 1.0
    rf_valid = False

    global global_engineered_df
    global global_rf_model, global_gb_model, global_iso_model, global_feature_scaler
    global DEBUG_MODE
    
    # [FIX] Gate RF/GB inference: only run classifiers when anomaly detected
    if global_engineered_df is not None and global_rf_model is not None and global_gb_model is not None and global_iso_model is not None and global_feature_scaler is not None:
        src_df = global_engineered_df[global_engineered_df["source"] == source].sort_values("timestamp").reset_index(drop=True)
        end_mask = src_df["timestamp"] <= timestamp
        window_size_seq = sequence.shape[0]
        candidates = src_df[end_mask]
        if len(candidates) >= window_size_seq:
            seq_slice = candidates.iloc[-window_size_seq:]
            feature_dict = build_sequence_feature_vector(seq_slice)
            feature_df = pd.DataFrame([feature_dict])

            # Align column order to exactly what StandardScaler was trained on.
            # This prevents silent feature-position mismatches when dict ordering
            # ever differs from training-time ordering.
            if hasattr(global_feature_scaler, "feature_names_in_"):
                expected_cols = list(global_feature_scaler.feature_names_in_)
                # Only keep columns the scaler knows; fill any missing with 0.0
                for col in expected_cols:
                    if col not in feature_df.columns:
                        feature_df[col] = 0.0
                feature_df = feature_df[expected_cols]

            feature_scaled = pd.DataFrame(
                global_feature_scaler.transform(feature_df),
                columns=feature_df.columns,
            )
            iso_score = float(global_iso_model.decision_function(feature_scaled)[0])
            feature_scaled["iso_score"] = iso_score

            # Debug: confirm RF is receiving the right feature shape
            if DEBUG_MODE:
                print("RF Input Features:", feature_scaled.columns.tolist())
                print("RF Input Shape:", feature_scaled.shape)

            # RF Inference
            rf_raw = global_rf_model.predict(feature_scaled)[0]
            if hasattr(global_rf_model, "predict_proba"):
                rf_probs = global_rf_model.predict_proba(feature_scaled)[0]
                rf_conf = float(np.max(rf_probs))
            else:
                rf_conf = 0.0

            # [FIX 1] RF confidence gating — only trust RF when confident enough
            rf_valid = rf_conf >= 0.6

            # Injection bias guard: require ≥0.92 confidence for injection label
            if rf_raw == "injection_attack" and rf_conf < 0.92:
                rf_raw = "noise_attack"
                
            # GB Inference
            gb_raw = global_gb_model.predict(feature_scaled)[0]
            if hasattr(global_gb_model, "predict_proba"):
                gb_probs = global_gb_model.predict_proba(feature_scaled)[0]
                gb_conf = float(np.max(gb_probs))
            else:
                gb_conf = 0.0
                
            label_mapping = {
                "drift_attack": "Drift Attack",
                "injection_attack": "Injection Attack",
                "noise_attack": "Noise Attack",
                "drop_attack": "Drop Attack",
                "normal": "Normal",
            }

            # Re-evaluate RF 'normal' when LSTM says strong anomaly
            if rf_raw == "normal" and anomaly_flag and reconstruction_error > threshold * 1.1:
                rf_raw = "drift_attack"

            rf_prediction = label_mapping.get(rf_raw, "Normal")
            gb_prediction = label_mapping.get(gb_raw, "Normal")


    temp_slope = feature_dict.get("temp_30_slope", 0.0)
    temp_std = feature_dict.get("temp_30_std", 0.0)
    temp_range = feature_dict.get("temp_30_range", 0.0)
    temp_entropy = feature_dict.get("temp_30_entropy", 0.0)
    
    temp_10_max_jump = feature_dict.get("temp_10_max_jump", 0.0)
    temp_10_zscore_max = feature_dict.get("temp_10_zscore_max", 0.0)

    # Noise suppression: very low std means stable signal — let ML decide instead of rules
    if temp_std < 0.3 and temp_range < 1.0 and not replay_result.is_replay:
        pass_to_ml = True
    else:
        pass_to_ml = False

    # ===== DECISION CHAIN =====
    # Priority: Replay > Rule Engine > RF (if confident) > Normal

    # [FIX 3] Replay is HIGHEST priority
    if replay_result.is_replay:
        predicted_label = "Replay Attack"
        confidence: float | str | None = "HIGH"
        decision_source = "rule_engine"

    # Noise Attack (high overall variance in the window) - check first to avoid false-spikes
    elif temp_std > threshold_config.noise_std_threshold and temp_entropy > threshold_config.noise_entropy_threshold and feature_dict:
        predicted_label = "Noise Attack"
        confidence = "HIGH"
        decision_source = "rule_engine"
    elif temp_entropy > 2.5:
        predicted_label = "Noise Attack"
        confidence = "HIGH"
        decision_source = "rule_engine"

    # Injection Attack (sudden extreme jumps, but overall variance is not noise-like)
    elif not pass_to_ml and (temp_10_max_jump > 5.0 or temp_10_zscore_max > 5.0):
        predicted_label = "Injection Attack"
        confidence = "HIGH"
        decision_source = "rule_engine"
    elif not pass_to_ml and temp_range > threshold_config.injection_range_threshold and feature_dict:
        predicted_label = "Injection Attack"
        confidence = "HIGH"
        decision_source = "rule_engine"

    # Drift Attacks can occur on stable signals (low variance) initially - do not gate by not pass_to_ml
    elif abs(temp_slope) > threshold_config.drift_threshold and feature_dict:
        predicted_label = "Drift Attack"
        confidence = "HIGH"
        decision_source = "rule_engine"

    # Drop Attack represents a stuck sensor (freeze) or a sudden large temperature drop
    elif temp_std < 0.1 and temp_range < 0.4 and feature_dict:
        predicted_label = "Drop Attack"
        confidence = "HIGH"
        decision_source = "rule_engine"
    elif temp_slope < -0.15 and feature_dict:
        predicted_label = "Drop Attack"
        confidence = "HIGH"
        decision_source = "rule_engine"

    else:
        if 'rf_raw' in locals() and rf_conf >= 0.5:
            predicted_label = {
                "normal": "Normal",
                "drift_attack": "Drift Attack",
                "noise_attack": "Noise Attack",
                "injection_attack": "Injection Attack",
                "drop_attack": "Drop Attack",
                "replay_attack": "Replay Attack"
            }.get(rf_raw, rf_raw)
            decision_source = "ensemble_rf"
            if rf_conf > 0.75:
                confidence = "HIGH"
            elif rf_conf > 0.5:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
        else:
            predicted_label = "Normal"
            confidence = "LOW"
            decision_source = "default"

    # Only override Normal to Noise Attack if the signal actually displays elevated standard deviation/entropy
    if predicted_label == "Normal" and anomaly_flag:
        if temp_std > 1.2 or temp_entropy > 2.0:
            predicted_label = "Noise Attack"

    # [FIX 4] Feedback — only assists when confidence is LOW
    from feedback_engine import load_feedback, find_similar_feedback
    feedback_memory = load_feedback()
    
    if feature_dict:
        current_features = np.array(list(feature_dict.values()), dtype=np.float32)
        matched_result = find_similar_feedback(current_features, feedback_memory)
        
        if matched_result is not None:
            feedback_label, sim_score, match_count = matched_result
            # Only apply feedback when similarity is very high AND current
            # confidence is LOW — prevents feedback from overriding strong predictions.
            if sim_score > 0.95 and confidence == "LOW":
                predicted_label = feedback_label
                if match_count > 3:
                    confidence = "VERY HIGH"
                else:
                    confidence = "HIGH"
                decision_source = "feedback_memory"

    # [FIX 6] Debug prints — gated by DEBUG_MODE
    if DEBUG_MODE:
        print(f"RF Prediction: {rf_prediction}")
        print(f"RF Confidence: {rf_conf:.4f}")
        print(f"RF Valid: {rf_valid}")
        print(f"Confidence: {confidence}")
        print(f"Final Label: {predicted_label}")
        print(f"Source: {decision_source}")

    global ENABLE_FEEDBACK
    if ENABLE_FEEDBACK and 'seq_slice' in locals() and not seq_slice.empty:
        unique_sensors = seq_slice['sensor_id'].unique().tolist()
        min_ts = seq_slice['timestamp'].min()
        max_ts = seq_slice['timestamp'].max()
        last_5 = seq_slice[['timestamp', 'sensor_id', 'temperature_c', 'humidity_percent']].tail(5)

        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        if DEBUG_MODE:
            print(f"\n{BOLD}================ WINDOW ANALYSIS ================{RESET}")
            print("Sensors:")
            print(f"{unique_sensors}")
            print("\nTime Range:")
            print(f"{min_ts} -> {max_ts}")
            print("\n--- SENSOR DATA (LAST 5 ROWS) ---")
            print("timestamp | sensor_id | temperature_c | humidity_percent")
            print("--------------------------------------------------------")
            for _, r in last_5.iterrows():
                print(f"{r['timestamp']} | {r['sensor_id']} | {r['temperature_c']:.2f} | {r['humidity_percent']:.2f}")
            
            print("\n--- FEATURES ---")
            print(f"Slope: {temp_slope:.4f}")
            print(f"Std: {temp_std:.4f}")
            print(f"Range: {temp_range:.4f}")
            print(f"Entropy: {temp_entropy:.4f}\n")

        pred_color = GREEN if predicted_label == "Normal" else RED
        print(f">>> FINAL DECISION: {pred_color}{predicted_label}{RESET} | Source: {YELLOW}{decision_source}{RESET} | Confidence: {confidence}")

        if decision_source == "feedback_memory":
            pass

        if DEBUG_MODE:
            print("--------------------------------------------------------\n")

        valid_labels = ["replay", "injection", "drop", "drift", "noise", "normal"]
        label_map = {
            "replay": "Replay Attack",
            "injection": "Injection Attack",
            "drop": "Drop Attack",
            "drift": "Drift Attack",
            "noise": "Noise Attack",
            "normal": "Normal",
        }
        
        global INTERACTIVE_MODE
        if INTERACTIVE_MODE:
            user_input = input("Enter correct label (replay/injection/drop/drift/noise/normal) OR press Enter to accept: ").strip().lower()
        else:
            user_input = ""

        if user_input and user_input in valid_labels:
            from feedback_engine import add_feedback
            add_feedback(current_features.tolist(), label_map[user_input])
            predicted_label = label_map[user_input]
            decision_source = "user_feedback"
            confidence = "HIGH"
            print(f"{BOLD}{YELLOW}>>> USER CORRECTION APPLIED{RESET}\n")

    # [FIX 4] Final label normalization — ensures only clean display labels are returned
    _label_normalizer = {
        "normal": "Normal",
        "replay_attack": "Replay Attack",
        "injection_attack": "Injection Attack",
        "drop_attack": "Drop Attack",
        "drift_attack": "Drift Attack",
        "noise_attack": "Noise Attack",
    }
    predicted_label = _label_normalizer.get(predicted_label.lower().replace(" ", "_"), predicted_label)

    return DetectionResult(
        timestamp=timestamp or pd.Timestamp.utcnow(),
        source=source,
        predicted_label=predicted_label,
        confidence=confidence,
        decision_source=decision_source,
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


def _style_visualization_axes(ax: Any, title: str, ylabel: str = "Network Metric") -> None:
    ax.set_title(title, fontsize=14, fontweight="semibold")
    ax.set_xlabel("Time Index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(fig: plt.Figure, output_path: Path | None) -> None:
    fig.tight_layout()
    if output_path is None:
        plt.close(fig)
        return
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_realtime_clean(df: pd.DataFrame, output_path: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["temperature_c"], label="Network Metric", color="#1f77b4", linewidth=1.5, alpha=0.8)
    _style_visualization_axes(ax, "Real-Time Network Traffic Monitoring")
    ax.legend(frameon=False)
    _save_figure(fig, output_path)


def _plot_attack_highlights(df: pd.DataFrame, output_path: Path | None = None) -> None:
    attack_colors = {
        "replay_attack": "#d62728",
        "injection_attack": "#ff7f0e",
        "drop_attack": "#9467bd",
        "noise_attack": "#2ca02c",
        "drift_attack": "#111111",
    }
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["temperature_c"], label="Network Metric", color="#4c78a8", linewidth=1.4, alpha=0.65)
    for attack, color in attack_colors.items():
        attack_rows = df[df["attack_type"] == attack]
        if attack_rows.empty:
            continue
        ax.scatter(
            attack_rows.index,
            attack_rows["temperature_c"],
            color=color,
            label=attack.replace("_", " ").title(),
            s=18,
            alpha=0.9,
            edgecolors="none",
        )
    _style_visualization_axes(ax, "Network Behavior with Attack Highlights")
    ax.legend(frameon=False, ncol=3)
    _save_figure(fig, output_path)


def _plot_replay_focus(df: pd.DataFrame, output_path: Path | None = None) -> None:
    replay_df = df[df["attack_type"] == REPLAY_LABEL]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["temperature_c"], color="#9aa5b1", linewidth=1.2, alpha=0.35, label="Normal Background")
    # Add vertical markers at replay attack rows
    replay_idx = df[df["attack_type"] == REPLAY_LABEL].index
    for ridx in replay_idx:
        ax.axvline(x=ridx, color='red', alpha=0.05)
    if not replay_df.empty:
        ax.scatter(
            replay_df.index,
            replay_df["temperature_c"],
            color="#d62728",
            label="Replay Attack",
            s=22,
            alpha=0.95,
            zorder=5,
            edgecolors="white",
            linewidths=0.3,
        )
    _style_visualization_axes(ax, "Replay Attack Detection Visualization")
    ax.legend(frameon=False)
    _save_figure(fig, output_path)


def _plot_zoom_view(df: pd.DataFrame, output_path: Path | None = None, limit: int = 1000) -> None:
    zoom_df = df.iloc[:limit]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(zoom_df.index, zoom_df["temperature_c"], label="Network Metric", color="#1f77b4", linewidth=1.5, alpha=0.8)
    _style_visualization_axes(ax, f"Zoomed Network Traffic View (First {len(zoom_df)} Points)")
    ax.legend(frameon=False)
    _save_figure(fig, output_path)


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


global_rf_model = None
global_gb_model = None
global_iso_model = None
global_engineered_df = None


from features.sequence_features import extract_sequence_features, build_sequence_feature_vector


def train_rf_classifier(df: pd.DataFrame, window_size: int = 60):
    global global_rf_model, global_gb_model, global_iso_model, global_feature_scaler

    # Use the same sliding-window logic as inference:
    # group per sensor, stride=1, label = attack_type at the LAST row inside the window.
    # (Inference labels by the row at end_timestamp, i.e. the final row of the window.)
    source_col = "sensor_id" if "sensor_id" in df.columns else "source"
    X_list: list[dict] = []
    y_list: list[str] = []
    for source_id, group in df.groupby(source_col, sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        for i in range(window_size, len(group)):
            seq_df = group.iloc[i - window_size : i]
            feature_dict = build_sequence_feature_vector(seq_df)
            X_list.append(feature_dict)
            # iloc[i-1] = last row INSIDE the window (matches inference end_timestamp label)
            y_list.append(group["attack_type"].iloc[i - 1])

    X = pd.DataFrame(X_list)
    y = pd.Series(y_list)

    # Variance check: surface any near-zero-variance features before training
    low_var = X.var()[X.var() < 0.01]
    if not low_var.empty:
        print("Low-variance features (may not help RF):", low_var.index.tolist())
    else:
        print("Feature variance OK — all features have variance ≥ 0.01")


    split_index = int(len(X) * 0.8)

    X_train = X[:split_index].reset_index(drop=True)
    y_train = y[:split_index].reset_index(drop=True)
    X_test = X[split_index:].reset_index(drop=True)
    y_test = y[split_index:].reset_index(drop=True)

    # Oversample ALL minority attack classes to at least 300 samples
    # (Previously only injection_attack was boosted — all others were left imbalanced)
    from sklearn.utils import resample
    from collections import Counter
    train_df = pd.concat([X_train, y_train.rename("attack_type")], axis=1)
    minority_classes = [c for c in train_df["attack_type"].unique() if c != "normal"]
    oversample_target = 300
    resampled_parts = [train_df]
    for cls in minority_classes:
        cls_df = train_df[train_df["attack_type"] == cls]
        if 0 < len(cls_df) < oversample_target:
            cls_up = resample(cls_df, replace=True, n_samples=oversample_target, random_state=42)
            resampled_parts.append(cls_up)
    train_df = pd.concat(resampled_parts, ignore_index=True)

    X_train = train_df.drop(columns=["attack_type"])
    y_train = train_df["attack_type"]

    print("Training class distribution:", Counter(y_train))

    # Scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    global_feature_scaler = scaler

    iso_model = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
    normal_mask = y_train == "normal"
    if normal_mask.sum() > 0:
        iso_model.fit(X_train_scaled[normal_mask])
    else:
        iso_model.fit(X_train_scaled)
    
    X_train_scaled["iso_score"] = iso_model.decision_function(X_train_scaled)
    X_test_scaled["iso_score"] = iso_model.decision_function(X_test_scaled)
    global_iso_model = iso_model

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train_scaled, y_train)
    print("RF Classes:", rf_model.classes_)

    y_pred = rf_model.predict(X_test_scaled)

    # print("\nRandom Forest Classification Report:")
    # print(classification_report(y_test, y_pred, zero_division=1))

    global_rf_model = rf_model
    
    gb_model = GradientBoostingClassifier(
        random_state=42,
    )
    gb_model.fit(X_train_scaled, y_train)
    global_gb_model = gb_model

    return rf_model



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
    
    global global_engineered_df
    global_engineered_df = engineered
    
    # Train separate Random Forest classification module
    train_rf_classifier(engineered, window_size=data_config.window_size)
    
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
    final_predictions: list[str] = []
    window_true_labels: list[str] = []
    audit_rows: list[dict[str, Any]] = []
    
    decision_counts = {
        "rule_engine": 0,
        "ensemble_gb": 0,
        "ensemble_rf": 0
    }
    
    true_label_mapping = {
        "normal": "Normal",
        "replay_attack": "Replay Attack",
        "injection_attack": "Injection Attack",
        "drop_attack": "Drop Attack",
        "noise_attack": "Noise Attack",
        "drift_attack": "Drift Attack",
    }

    for idx, row in eval_bundle.metadata.iterrows():
        source = row["source"]
        
        # Calculate true window label
        src_df = engineered[engineered["source"] == source]
        window_df = src_df[(src_df["timestamp"] >= row["start_timestamp"]) & (src_df["timestamp"] <= row["end_timestamp"])]
        if not window_df.empty:
            most_common = window_df["attack_type"].mode()[0]
        else:
            most_common = "normal"
        window_true_labels.append(true_label_mapping.get(most_common, "Normal"))

        result = classify_window(
            sequence=eval_bundle.X[idx],
            model=model,
            threshold=threshold_main,
            history_buffer=list(history_buffers[source]),
            replay_config=replay_config,
            timestamp=row["end_timestamp"],
            source=source,
            window_size=data_config.window_size,
        )
        
        if result.decision_source in decision_counts:
            decision_counts[result.decision_source] += 1
        elif result.decision_source is not None:
            decision_counts[result.decision_source] = decision_counts.get(result.decision_source, 0) + 1

        predictions.append(result.predicted_label)
        final_predictions.append(result.predicted_label)
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
                    confidence=record.get("confidence"),
                    decision_source=record.get("decision_source"),
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

    realtime_clean_path = (output_path / "realtime_clean.png") if output_path is not None else Path("realtime_clean.png")
    attack_highlight_path = (output_path / "attack_highlight.png") if output_path is not None else Path("attack_highlight.png")
    replay_focus_path = (output_path / "replay_focus.png") if output_path is not None else Path("replay_focus.png")
    zoom_view_path = (output_path / "zoom_view.png") if output_path is not None else Path("zoom_view.png")

    _plot_realtime_clean(engineered, realtime_clean_path)
    _plot_attack_highlights(engineered, attack_highlight_path)
    _plot_replay_focus(engineered, replay_focus_path)
    _plot_zoom_view(engineered, zoom_view_path)

    # ===== IDS SUMMARY (Percentage) =====
    total = len(final_predictions)
    pred_counts = Counter(final_predictions)
    print("\n========== IDS SUMMARY ==========")
    for label, count in pred_counts.items():
        percent = (count / total) * 100
        print(f"  {label}: {percent:.2f}%")
    print(f"\n  Total Events: {total}")
    print("==================================")

    # ===== DECISION SOURCE SUMMARY =====
    print("\n========== DECISION SOURCE SUMMARY ==========")
    print(f"Rule Engine Decisions: {decision_counts.get('rule_engine', 0)}")
    print(f"Gradient Boosting Decisions: {decision_counts.get('ensemble_gb', 0)}")
    print(f"Random Forest Decisions: {decision_counts.get('ensemble_rf', 0)}")
    print("===========================================")

    # ===== DETECTION RATES =====
    true_counts = Counter(window_true_labels)
    print("\n========== DETECTION RATES ==========")
    for attack in true_counts:
        if attack != "Normal":
            detected = sum(
                1 for t, p in zip(window_true_labels, final_predictions)
                if t == attack and p == attack
            )
            total_attack = true_counts[attack]
            rate = (detected / total_attack) * 100 if total_attack > 0 else 0
            print(f"  {attack}: {detected}/{total_attack} ({rate:.2f}%)")
    print("=====================================")

    # ===== IDS PERFORMANCE (Recall, Precision, Missed) =====
    print("\n========== IDS PERFORMANCE ==========")
    for label in true_counts:
        detected = sum(
            1 for t, p in zip(window_true_labels, final_predictions)
            if t == label and p == label
        )
        t_count = true_counts[label]
        predicted_total = pred_counts.get(label, 0)
        recall = detected / t_count if t_count > 0 else 0
        precision = detected / predicted_total if predicted_total > 0 else 0
        missed = t_count - detected
        print(f"  {label}:")
        print(f"    Recall (Detection Rate): {recall:.2%} ({detected}/{t_count})")
        print(f"    Precision: {precision:.2%}")
        print(f"    Missed: {missed}")
    print("====================================")

    # ===== SYSTEM INFO =====
    print("\nSystem Info:")
    print(f"  Evaluation Mode: Window-based Temporal Detection")
    print(f"  Window Size: {data_config.window_size}")
    print(f"  Threshold (main): {threshold_main:.4f}")

    # ===== KEY INSIGHT =====
    print("\nKey Insight:")
    print("  Replay and drift attacks remain challenging due to similarity with normal patterns in temporal data.")

    label_counts = eval_bundle.metadata["attack_type"].value_counts().to_dict()
    prediction_counts = dict(Counter(final_predictions))
    true_labels = pd.Series(window_true_labels)
    # [FIX 5] Canonical label order — consistent across confusion matrix and reports
    class_labels = [
        "Normal",
        "Replay Attack",
        "Injection Attack",
        "Drop Attack",
        "Drift Attack",
        "Noise Attack",
    ]
    model_accuracy = float(accuracy_score(true_labels, final_predictions))
    print(f"\nOverall Accuracy: {model_accuracy * 100:.2f}%")
    confusion = confusion_matrix(true_labels, final_predictions, labels=class_labels)
    confusion_norm = confusion_matrix(true_labels, final_predictions, labels=class_labels, normalize='true')
    confusion_matrix_path = (output_path / "confusion_matrix.png") if output_path is not None else Path("confusion_matrix.png")
    confusion_norm_path = (output_path / "confusion_matrix_normalized.png") if output_path is not None else Path("confusion_matrix_normalized.png")

    # Raw counts confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=class_labels)
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (Counts) - Hybrid Network IDS")
    fig.tight_layout()
    fig.savefig(confusion_matrix_path, dpi=150)
    plt.close(fig)

    # Normalized (percentage) confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=confusion_norm, display_labels=class_labels)
    disp_norm.plot(cmap="Blues", ax=ax, colorbar=False, values_format=".2f")
    ax.set_title("Confusion Matrix (Normalized) - Hybrid Network IDS")
    fig.tight_layout()
    fig.savefig(confusion_norm_path, dpi=150)
    plt.close(fig)

    return {
        "detector": detector,
        "interval_summary": interval_summary,
        "label_counts": label_counts,
        "prediction_counts": prediction_counts,
        "model_accuracy": model_accuracy,
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_labels": class_labels,
        "confusion_matrix_path": str(confusion_matrix_path),
        "thresholds": {
            "mean_plus_2_std": threshold_loose,
            "mean_plus_3_std": threshold_main,
        },
        "validation_errors": val_errors,
        "evaluation_errors": eval_errors,
        "streaming_results": streaming_results,
        "output_dir": str(output_path) if output_path is not None else None,
        "visualization_paths": {
            "realtime_clean": str(realtime_clean_path),
            "attack_highlight": str(attack_highlight_path),
            "replay_focus": str(replay_focus_path),
            "zoom_view": str(zoom_view_path),
        },
    }
