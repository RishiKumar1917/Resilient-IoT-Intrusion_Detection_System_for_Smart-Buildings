from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import hybrid_iot_ids as ids
from hybrid_iot_ids import (
    DetectionResult,
    HistoryWindow,
    MinMaxScalerLite,
    ReplayConfig,
    TrainedHybridIDS,
    classify_window,
    create_stream_state,
    detect_replay,
    engineer_features,
    estimate_threshold,
    get_feature_columns,
    make_sequences,
    stream_infer,
)


class IdentityModel:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float32)


class ShiftModel:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float32) + 0.5


class PassthroughScaler:
    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float32)


class ConstantClassifier:
    def __init__(self, label: str = "normal") -> None:
        self.label = label

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.label], dtype=object)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.array([[1.0]], dtype=np.float32)


class ConstantIsoForest:
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.array([0.0], dtype=np.float32)


class HybridIoTIDSTest(unittest.TestCase):
    def _sample_frame(self) -> pd.DataFrame:
        rows = []
        for source in ["src_a", "src_b"]:
            for idx in range(8):
                attack_type = "normal" if idx < 7 else "injection_attack"
                rows.append(
                    {
                        "timestamp": pd.Timestamp("2026-01-01 00:00:00") + pd.Timedelta(seconds=idx + (100 if source == "src_b" else 0)),
                        "temperature_c": 20.0 + idx,
                        "humidity_percent": 40.0 + idx,
                        "source": source,
                        "sensor_id": source,
                        "attack_type": attack_type,
                        "label": 0 if attack_type == "normal" else 1,
                        "class_label": attack_type,
                    }
                )
        return pd.DataFrame(rows)

    def test_estimate_threshold_matches_mean_plus_sigma(self) -> None:
        errors = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        expected = float(np.percentile(errors, 95))
        self.assertAlmostEqual(estimate_threshold(errors, sigma=3), expected)

    def test_make_sequences_respects_source_boundaries(self) -> None:
        featured = engineer_features(self._sample_frame(), consistency_window=3)
        bundle = make_sequences(featured, window_size=4, feature_cols=get_feature_columns())
        self.assertTrue((bundle.metadata["source"].value_counts() == 5).all())
        self.assertEqual(set(bundle.metadata["source"]), {"src_a", "src_b"})

    def test_detect_replay_uses_exact_hash(self) -> None:
        replay_config = ReplayConfig(history_size=10, similarity_threshold=0.99, min_gap_windows=0)
        sequence = np.array([[0.1, 0.2], [0.2, 0.3]], dtype=np.float32)
        history = []
        for idx in range(6):
            seq = sequence.copy() if idx == 0 else sequence + idx
            history.append(
                HistoryWindow(
                    sequence=seq,
                    sequence_hash=ids._sequence_hash(seq, 2),
                    timestamp=pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=idx),
                    history_index=idx,
                )
            )
        result = detect_replay(sequence, history, replay_config)
        self.assertTrue(result.is_replay)
        self.assertEqual(result.matched_history_index, 0)

    def test_stream_infer_returns_none_until_window_full(self) -> None:
        feature_cols = get_feature_columns()
        scaler = MinMaxScalerLite(feature_names=feature_cols).fit(np.zeros((5, len(feature_cols)), dtype=np.float32))
        detector = TrainedHybridIDS(
            model=IdentityModel(),
            feature_cols=feature_cols,
            scalers={"src_a": scaler},
            threshold_main=0.01,
            threshold_loose=0.005,
            window_size=3,
            replay_config=ReplayConfig(history_size=5, min_gap_windows=1),
            consistency_window=3,
            training_history={"train_loss": [], "val_loss": []},
        )
        state = create_stream_state("src_a", detector)
        readings = [
            {"timestamp": "2026-01-01T00:00:00", "temperature_c": 20.0, "humidity_percent": 40.0, "source": "src_a"},
            {"timestamp": "2026-01-01T00:00:01", "temperature_c": 20.2, "humidity_percent": 40.2, "source": "src_a"},
            {"timestamp": "2026-01-01T00:00:02", "temperature_c": 20.4, "humidity_percent": 40.4, "source": "src_a"},
        ]
        self.assertIsNone(stream_infer(readings[0], state))
        self.assertIsNone(stream_infer(readings[1], state))
        result = stream_infer(readings[2], state)
        self.assertIsInstance(result, DetectionResult)
        self.assertEqual(result.predicted_label, "Normal")

    def test_classify_window_replay_requires_normal_reconstruction(self) -> None:
        replay_config = ReplayConfig(history_size=10, similarity_threshold=0.8, min_gap_windows=0)
        sequence = np.ones((3, 2), dtype=np.float32)
        history = []
        for idx in range(6):
            seq = sequence.copy() if idx == 0 else sequence + (idx * 0.1)
            history.append(
                HistoryWindow(
                    sequence=seq,
                    sequence_hash=ids._sequence_hash(seq, 2),
                    timestamp=pd.Timestamp("2026-01-01T00:00:00") + pd.Timedelta(minutes=idx),
                    history_index=idx,
                )
            )
        result = classify_window(
            sequence,
            model=ShiftModel(),
            threshold=0.01,
            history_buffer=history,
            replay_config=replay_config,
            timestamp=pd.Timestamp("2026-01-01T00:01:00"),
            source="src_a",
            window_size=3,
        )
        self.assertEqual(result.predicted_label, "Replay Attack")
        self.assertTrue(result.anomaly_flag)
        self.assertTrue(result.replay_flag)
        self.assertEqual(result.confidence, "HIGH")
        self.assertEqual(result.decision_source, "rule_engine")

    def test_classify_window_returns_replay_for_replay_like_normal_window(self) -> None:
        replay_config = ReplayConfig(history_size=10, similarity_threshold=0.8, min_gap_windows=0)
        sequence = np.ones((3, 2), dtype=np.float32)
        history = []
        for idx in range(6):
            seq = sequence.copy() if idx == 0 else sequence + (idx * 0.1)
            history.append(
                HistoryWindow(
                    sequence=seq,
                    sequence_hash=ids._sequence_hash(seq, 2),
                    timestamp=pd.Timestamp("2026-01-01T00:00:00") + pd.Timedelta(minutes=idx),
                    history_index=idx,
                )
            )
        result = classify_window(
            sequence,
            model=IdentityModel(),
            threshold=0.01,
            history_buffer=history,
            replay_config=replay_config,
            timestamp=pd.Timestamp("2026-01-01T00:01:00"),
            source="src_a",
            window_size=3,
        )
        self.assertEqual(result.predicted_label, "Replay Attack")
        self.assertFalse(result.anomaly_flag)
        self.assertTrue(result.replay_flag)

    def test_similarity_returns_zero_for_shape_mismatch(self) -> None:
        self.assertEqual(ids._window_similarity(np.ones((2, 2), dtype=np.float32), np.ones((3, 2), dtype=np.float32)), 0.0)

    def _classify_with_stubbed_features(
        self,
        feature_overrides: dict[str, float],
        *,
        control_columns: dict[str, float] | None = None,
    ) -> DetectionResult:
        base_features = {
            "temp_30_slope": 0.0,
            "temp_30_std": 0.0,
            "temp_30_range": 0.0,
            "temp_30_entropy": 0.0,
            "temp_10_max_jump": 0.0,
            "temp_10_zscore_max": 0.0,
        }
        base_features.update(feature_overrides)

        window_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-01-01T00:00:00", "2026-01-01T00:00:01", "2026-01-01T00:00:02"]
                ),
                "source": ["src_a", "src_a", "src_a"],
                "sensor_id": ["src_a", "src_a", "src_a"],
                "temperature_c": [20.0, 20.1, 20.2],
                "humidity_percent": [40.0, 40.1, 40.2],
            }
        )
        for col, val in (control_columns or {}).items():
            window_df[col] = [val, val, val]

        sequence = np.ones((3, 2), dtype=np.float32)
        timestamp = pd.Timestamp("2026-01-01T00:00:02")

        old_globals = {
            "global_engineered_df": getattr(ids, "global_engineered_df", None),
            "global_rf_model": getattr(ids, "global_rf_model", None),
            "global_gb_model": getattr(ids, "global_gb_model", None),
            "global_iso_model": getattr(ids, "global_iso_model", None),
            "global_feature_scaler": getattr(ids, "global_feature_scaler", None),
            "ENABLE_FEEDBACK": getattr(ids, "ENABLE_FEEDBACK", True),
        }
        setattr(ids, "global_engineered_df", window_df)
        setattr(ids, "global_rf_model", ConstantClassifier("normal"))
        setattr(ids, "global_gb_model", ConstantClassifier("normal"))
        setattr(ids, "global_iso_model", ConstantIsoForest())
        setattr(ids, "global_feature_scaler", PassthroughScaler())
        setattr(ids, "ENABLE_FEEDBACK", False)

        try:
            with patch.object(ids, "build_sequence_feature_vector", return_value=base_features):
                return classify_window(
                    sequence=sequence,
                    model=IdentityModel(),
                    threshold=1.0,
                    history_buffer=[],
                    replay_config=ReplayConfig(history_size=10, min_gap_windows=0),
                    threshold_config=ids.ThresholdConfig(),
                    timestamp=timestamp,
                    source="src_a",
                    window_size=3,
                )
        finally:
            for name, value in old_globals.items():
                setattr(ids, name, value)

    def test_drop_rule_accepts_expected_sensor_noise(self) -> None:
        result = self._classify_with_stubbed_features({"temp_30_std": 0.25, "temp_30_range": 0.35})
        self.assertEqual(result.predicted_label, "Drop Attack")
        self.assertEqual(result.decision_source, "rule_engine")

    def test_injection_rule_suppressed_when_hvac_active(self) -> None:
        result = self._classify_with_stubbed_features(
            {"temp_30_std": 1.0, "temp_30_range": 7.0, "temp_10_max_jump": 7.0, "temp_10_zscore_max": 7.0},
            control_columns={"hvac_on": 1.0},
        )
        self.assertNotEqual(result.predicted_label, "Injection Attack")

    def test_injection_rule_triggers_when_control_state_inactive(self) -> None:
        result = self._classify_with_stubbed_features(
            {"temp_30_std": 1.0, "temp_30_range": 7.0, "temp_10_max_jump": 7.0, "temp_10_zscore_max": 7.0},
            control_columns={"hvac_on": 0.0},
        )
        self.assertEqual(result.predicted_label, "Injection Attack")
        self.assertEqual(result.decision_source, "rule_engine")


if __name__ == "__main__":
    unittest.main()
