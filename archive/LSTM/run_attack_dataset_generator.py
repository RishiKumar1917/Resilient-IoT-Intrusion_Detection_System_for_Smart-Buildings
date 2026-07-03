from __future__ import annotations

import argparse
from pathlib import Path

from synthetic_attack_generator import AttackGenerationConfig, generate_enhanced_iot_dataset, save_enhanced_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic cyber-physical IoT attack dataset.")
    parser.add_argument("--input", default="iot_sensor_dataset.csv", help="Input normal IoT dataset path.")
    parser.add_argument("--output", default="enhanced_iot_dataset.csv", help="Output CSV path.")
    parser.add_argument(
        "--plot",
        default="enhanced_iot_attack_examples.png",
        help="Output path for the attack example visualization.",
    )
    parser.add_argument(
        "--examples-csv",
        default="enhanced_iot_attack_examples.csv",
        help="Output path for attack example rows.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible attack generation.")
    args = parser.parse_args()

    config = AttackGenerationConfig(
        input_path=Path(args.input),
        output_path=Path(args.output),
        example_plot_path=Path(args.plot),
        example_csv_path=Path(args.examples_csv),
        random_seed=args.seed,
    )
    enhanced_df, summary, segments = generate_enhanced_iot_dataset(config)
    save_enhanced_outputs(enhanced_df, segments, config)

    print("Synthetic cyber-physical attack dataset generated.")
    print("Rows:", summary["row_count"])
    print("Attack type counts:", summary["attack_type_counts"])
    print("Segments by type:", summary["segments_by_type"])
    print("Saved CSV:", Path(config.output_path).resolve())
    print("Saved example plot:", Path(config.example_plot_path).resolve())
    print("Saved example CSV:", Path(config.example_csv_path).resolve())


if __name__ == "__main__":
    main()
