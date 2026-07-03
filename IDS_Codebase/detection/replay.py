from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import pandas as pd

from detection.types import HistoryWindow, ReplayConfig, ReplayResult


def _sequence_hash(sequence: np.ndarray, decimals: int) -> str:
    rounded = np.round(np.asarray(sequence, dtype=np.float32), decimals=decimals)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def compute_similarity(seq1: np.ndarray, seq2: np.ndarray) -> float:
    s1 = np.asarray(seq1, dtype=np.float32).reshape(-1)
    s2 = np.asarray(seq2, dtype=np.float32).reshape(-1)
    if s1.shape != s2.shape:
        return 0.0
    s1_std = np.std(s1)
    s2_std = np.std(s2)
    if s1_std == 0 or s2_std == 0:
        return 0.0
    s1_norm = (s1 - np.mean(s1)) / s1_std
    s2_norm = (s2 - np.mean(s2)) / s2_std
    return float(np.corrcoef(s1_norm, s2_norm)[0, 1])


def _window_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return compute_similarity(a, b)


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

    # Improve replay detection stability:
    # When comparing current window with history, skip the most recent 5 windows
    # to avoid self-matching false positives.
    # Only compare against older windows in history buffer.
    gap_windows = max(5, replay_config.min_gap_windows)
    eligible = history_list[:-gap_windows] if len(history_list) > gap_windows else []

    if current_timestamp is not None and replay_config.min_gap_seconds > 0:
        eligible = [
            item for item in eligible
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
        similarity = compute_similarity(sequence, item.sequence)
        
        if similarity > replay_config.similarity_threshold:
            return ReplayResult(
                is_replay=True,
                similarity=float(similarity),
                matched_history_index=item.history_index,
                matched_timestamp=item.timestamp,
                reason="correlation_match",
            )
        if similarity > best_similarity:
            best_similarity = similarity
            best_item = item

    return ReplayResult(is_replay=False, similarity=float(best_similarity) if best_similarity >= 0 else None)
