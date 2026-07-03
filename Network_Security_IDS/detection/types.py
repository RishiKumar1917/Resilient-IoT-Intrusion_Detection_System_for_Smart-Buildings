from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


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
class ReplayResult:
    is_replay: bool
    similarity: float | None = None
    matched_history_index: int | None = None
    matched_timestamp: pd.Timestamp | None = None
    reason: str | None = None


@dataclass
class HistoryWindow:
    sequence: np.ndarray
    sequence_hash: str
    timestamp: pd.Timestamp
    history_index: int

