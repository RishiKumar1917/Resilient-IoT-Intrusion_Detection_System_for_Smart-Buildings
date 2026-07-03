from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from hybrid_iot_ids import DataConfig, ModelConfig, ReplayConfig, run_demo_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hybrid IoT IDS demo pipeline.")
    parser.add_argument("--output-dir", default="demo_outputs", help="Directory for plots and audit logs.")
    parser.add_argument("--window-size", type=int, default=30, help="Sliding window length.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs for the LSTM autoencoder.")
    parser.add_argument("--history-size", type=int, default=100, help="Replay history buffer size.")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.94,
        help="Replay similarity threshold.",
    )
    args = parser.parse_args()

    summary = run_demo_pipeline(
        data_config=DataConfig(window_size=args.window_size, data_path="enhanced_iot_dataset_3sensors.csv"),
        model_config=ModelConfig(epochs=args.epochs),
        replay_config=ReplayConfig(
            history_size=args.history_size,
            similarity_threshold=args.similarity_threshold,
            min_gap_windows=max(5, args.window_size // 3),
        ),
        output_dir=Path(args.output_dir),
    )

    print("Hybrid IoT IDS demo completed.")
    print(f"Model Accuracy: {summary['model_accuracy'] * 100:.2f} %")
    print("Confusion matrix saved to:", summary["confusion_matrix_path"])
    print("Visualization files:", summary["visualization_paths"])
    print("Thresholds:", summary["thresholds"])
    print("Ground-truth label counts:", summary["label_counts"])
    print("Predicted label counts:", summary["prediction_counts"])
    print("Output directory:", summary["output_dir"])


if __name__ == "__main__":
    main()
