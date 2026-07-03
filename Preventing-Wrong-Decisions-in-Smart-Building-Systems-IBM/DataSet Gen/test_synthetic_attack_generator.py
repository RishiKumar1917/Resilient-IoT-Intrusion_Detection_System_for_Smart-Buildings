from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from synthetic_attack_generator import (
    AttackGenerationConfig,
    DROP_LABEL,
    DRIFT_LABEL,
    INJECTION_LABEL,
    NOISE_LABEL,
    NORMAL_LABEL,
    REPLAY_LABEL,
    generate_enhanced_iot_dataset,
    load_base_dataset,
)


class SyntheticAttackGeneratorTest(unittest.TestCase):
    def _make_input_csv(self, path: Path) -> None:
        rows: list[dict[str, object]] = []
        base_ts = pd.Timestamp("2026-01-01 00:00:00")
        for sensor_id in ["sensor_1", "sensor_2"]:
            for step in range(180):
                rows.append(
                    {
                        "timestamp": base_ts + pd.Timedelta(seconds=step * 10 + (0 if sensor_id == "sensor_1" else 5)),
                        "sensor_id": sensor_id,
                        "temperature_c": 25.0 + (step * 0.02),
                        "humidity_percent": 50.0 + (step * 0.03),
                    }
                )
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_generated_dataset_preserves_timestamps_and_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.csv"
            self._make_input_csv(input_path)
            original_df = load_base_dataset(input_path)
            enhanced_df, summary, _segments = generate_enhanced_iot_dataset(
                AttackGenerationConfig(input_path=input_path)
            )

            self.assertEqual(
                enhanced_df.columns.tolist(),
                ["timestamp", "sensor_id", "temperature_c", "humidity_percent", "attack_type"],
            )
            pd.testing.assert_series_equal(original_df["timestamp"], enhanced_df["timestamp"], check_names=False)
            pd.testing.assert_series_equal(original_df["sensor_id"], enhanced_df["sensor_id"], check_names=False)
            self.assertEqual(summary["row_count"], len(original_df))

    def test_attack_ratio_and_labels_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.csv"
            self._make_input_csv(input_path)
            enhanced_df, _summary, _segments = generate_enhanced_iot_dataset(
                AttackGenerationConfig(input_path=input_path)
            )
            attack_ratio = 1.0 - (enhanced_df["attack_type"] == NORMAL_LABEL).mean()
            self.assertGreaterEqual(attack_ratio, 0.26)
            self.assertLessEqual(attack_ratio, 0.34)
            labels = set(enhanced_df["attack_type"])
            self.assertTrue({NORMAL_LABEL, REPLAY_LABEL, INJECTION_LABEL, DRIFT_LABEL, NOISE_LABEL, DROP_LABEL}.issubset(labels))

    def test_replay_segments_copy_past_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.csv"
            self._make_input_csv(input_path)
            enhanced_df, _summary, segments = generate_enhanced_iot_dataset(
                AttackGenerationConfig(input_path=input_path, random_seed=7)
            )
            original_df = load_base_dataset(input_path)
            replay_segment = next(segment for segment in segments if segment.attack_type == REPLAY_LABEL)
            donor = original_df.loc[replay_segment.donor_global_indices, ["temperature_c", "humidity_percent"]].reset_index(drop=True)
            target = enhanced_df.loc[replay_segment.target_global_indices, ["temperature_c", "humidity_percent"]].reset_index(drop=True)
            pd.testing.assert_frame_equal(donor, target)


if __name__ == "__main__":
    unittest.main()
