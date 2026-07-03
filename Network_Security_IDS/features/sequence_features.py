import numpy as np
import pandas as pd

def extract_sequence_features(seq: np.ndarray) -> dict:
    """Compute summary statistics over a 1-D numeric sequence.

    Features extracted:
      - mean, std, min, max, range  (basic statistics)
      - slope          : linear trend via np.polyfit degree-1
      - acceleration   : change-in-slope (second derivative proxy)
      - cumulative_change : last value minus window mean
      - diff_max       : max - min (peak-to-peak swing)
      - spike_strength : max(|diff(seq)|) — largest single-step jump
      - var_ratio      : std / (mean + 1e-6) — coefficient of variation
      - trend_change   : slope_second_half - slope_first_half
      - zero_cross_rate: fraction of sign changes in diff(seq)
    All outputs are NaN-safe floats.
    """
    seq = np.asarray(seq, dtype=np.float32)
    n = len(seq)

    # --- basic statistics ---
    mean_val = float(np.mean(seq))
    std_val = float(np.std(seq))
    min_val = float(np.min(seq))
    max_val = float(np.max(seq))
    range_val = float(np.ptp(seq))
    spike_val = float(np.max(np.abs(seq - mean_val))) if n >= 1 else 0.0
    max_jump_val = float(np.max(np.abs(np.diff(seq)))) if n >= 2 else 0.0

    temp_zscore_max = float(np.max(np.abs((seq - mean_val) / (std_val + 0.1)))) if n >= 1 else 0.0
    temp_outlier_count = float(np.sum(np.abs(seq - mean_val) > 2 * (std_val + 0.1))) if n >= 1 else 0.0

    # --- slope: linear trend over the window ---
    if n >= 2:
        slope = float(np.polyfit(np.arange(n), seq, 1)[0])
    else:
        slope = 0.0
    if np.isnan(slope):
        slope = 0.0

    # --- acceleration: change in slope (split window into two halves) ---
    if n >= 4:
        mid = n // 2
        slope_first = float(np.polyfit(np.arange(mid), seq[:mid], 1)[0])
        slope_second = float(np.polyfit(np.arange(n - mid), seq[mid:], 1)[0])
        if np.isnan(slope_first):
            slope_first = 0.0
        if np.isnan(slope_second):
            slope_second = 0.0
        acceleration = slope_second - slope_first
    else:
        acceleration = 0.0

    # --- cumulative change: last value minus window mean ---
    cumulative_change = float(seq[-1]) - mean_val if n >= 1 else 0.0

    # --- entropy (histogram-based) ---
    counts, _ = np.histogram(seq, bins=10)
    probs = counts / (np.sum(counts) + 1e-9)
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log(probs)))

    # --- NEW: discriminative features for better class separability ---

    # diff_max: full peak-to-peak swing (stronger signal than range for attacks)
    diff_max = max_val - min_val

    # spike_strength: largest single-step absolute jump (catches injection/noise spikes)
    diffs = np.diff(seq)
    spike_strength = float(np.max(np.abs(diffs))) if n >= 2 else 0.0

    # var_ratio: coefficient of variation — relative variability independent of scale
    var_ratio = std_val / (abs(mean_val) + 1e-6)

    # trend_change: slope of second half minus slope of first half
    # (reuses acceleration already computed above — just expose it under a clearer name)
    trend_change = acceleration

    # zero_cross_rate: fraction of diff-sign changes — high for noise, low for drift/drop
    if n >= 2 and len(diffs) > 0:
        sign_changes = float(np.sum(np.diff(np.sign(diffs)) != 0))
        zero_cross_rate = sign_changes / max(len(diffs) - 1, 1)
    else:
        zero_cross_rate = 0.0

    return {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "range": range_val,
        "spike": spike_val,
        "max_jump": max_jump_val,
        "zscore_max": temp_zscore_max,
        "outlier_count": temp_outlier_count,
        "slope": slope,
        "acceleration": acceleration,
        "cumulative_change": cumulative_change,
        "entropy": entropy,
        # --- new separability features ---
        "diff_max": diff_max,
        "spike_strength": spike_strength,
        "var_ratio": var_ratio,
        "trend_change": trend_change,
        "zero_cross_rate": zero_cross_rate,
    }


def build_sequence_feature_vector(sequence_df: pd.DataFrame) -> dict:
    seq_len = len(sequence_df)
    win_30 = sequence_df.iloc[-30:] if seq_len >= 30 else sequence_df
    win_10 = sequence_df.iloc[-10:] if seq_len >= 10 else sequence_df

    t_30 = extract_sequence_features(win_30["temperature_c"].values)
    h_30 = extract_sequence_features(win_30["humidity_percent"].values)
    t_10 = extract_sequence_features(win_10["temperature_c"].values)
    h_10 = extract_sequence_features(win_10["humidity_percent"].values)

    combined: dict = {}
    for k, v in t_30.items(): combined[f"temp_30_{k}"] = v
    for k, v in h_30.items(): combined[f"hum_30_{k}"] = v
    for k, v in t_10.items(): combined[f"temp_10_{k}"] = v
    for k, v in h_10.items(): combined[f"hum_10_{k}"] = v
    return combined
