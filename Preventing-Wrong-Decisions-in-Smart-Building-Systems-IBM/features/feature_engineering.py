import numpy as np
import pandas as pd

def engineer_features(df: pd.DataFrame, consistency_window: int = 5) -> pd.DataFrame:
    feature_frames: list[pd.DataFrame] = []
    for source, group in df.groupby("source", sort=False):
        g = group.sort_values("timestamp").copy()
        g["time_diff"] = g["timestamp"].diff().dt.total_seconds().fillna(0.0).clip(lower=0.0)
        g["temperature_delta"] = g["temperature_c"].diff().fillna(0.0)
        g["humidity_delta"] = g["humidity_percent"].diff().fillna(0.0)
        g["temp_diff"] = g["temperature_c"].diff().fillna(0.0)
        g["humidity_diff"] = g["humidity_percent"].diff().fillna(0.0)
        g["temp_rolling_mean"] = g["temperature_c"].rolling(window=5).mean().bfill()
        g["humidity_rolling_mean"] = g["humidity_percent"].rolling(window=5).mean().bfill()
        g["temp_std"] = g["temperature_c"].rolling(window=5).std().fillna(0.0)
        g["humidity_std"] = g["humidity_percent"].rolling(window=5).std().fillna(0.0)
        g["temp_rate_change"] = g["temperature_c"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        g["humidity_rate_change"] = g["humidity_percent"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
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
        # EMA features
        g["temp_ema"] = g["temperature_c"].ewm(span=5).mean()
        g["humidity_ema"] = g["humidity_percent"].ewm(span=5).mean()
        # Trend slope features
        g["temp_slope"] = g["temperature_c"].diff(5).fillna(0.0)
        g["humidity_slope"] = g["humidity_percent"].diff(5).fillna(0.0)
        # Signal entropy features
        def _calculate_entropy(series):
            counts, _ = np.histogram(series, bins=10)
            probs = counts / (np.sum(counts) + 1e-9)
            probs = probs[probs > 0]
            return -np.sum(probs * np.log(probs))
        g["temp_entropy"] = g["temperature_c"].rolling(window=10).apply(_calculate_entropy, raw=True).fillna(0.0)
        g["humidity_entropy"] = g["humidity_percent"].rolling(window=10).apply(_calculate_entropy, raw=True).fillna(0.0)
        g = g.bfill()
        g["source"] = source
        feature_frames.append(g)
    featured = pd.concat(feature_frames, ignore_index=True)
    print("\nFeature Engineering Completed. Columns:")
    print(featured.columns)
    return featured
