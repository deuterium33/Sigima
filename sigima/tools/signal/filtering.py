# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Filtering functions (see parent package :mod:`sigima.tools.signal`).

This module provides denoising and filtering tools, such as Savitzky-Golay.

"""

from __future__ import annotations

import dataclasses

import numpy as np
import scipy.signal  # type: ignore[import]


@dataclasses.dataclass
class SimilarityResult:
    """Result of signal similarity validation."""

    ok: bool
    rel_dc_diff: float
    corr: float


def signal_similarity(
    y: np.ndarray,
    y_filtered: np.ndarray,
    max_dc_diff: float = 1e-2,
    min_corr: float = 0.99,
) -> SimilarityResult:
    """Check global similarity between two signals.

    Criteria:
        - DC level (mean value) must not drift more than ``max_dc_diff`` (relative).
        - Correlation (cosine similarity) must stay above ``min_corr``.

    Args:
        y: Original 1D signal.
        y_filtered: Filtered 1D signal (same length as ``y``).
        max_dc_diff: Maximum allowed relative change in mean value.
        min_corr: Minimum allowed correlation between signals.

    Returns:
        A result object containing the similarity metrics.
    """
    if y.size != y_filtered.size:
        raise ValueError("Signals must have the same length.")

    # DC level
    dc_orig = float(np.mean(y))
    dc_filt = float(np.mean(y_filtered))
    rel_diff = abs(dc_filt - dc_orig) / (abs(dc_orig) + 1e-12)

    # Correlation (cosine similarity)
    num = float(np.dot(y, y_filtered))
    denom = float(np.linalg.norm(y) * np.linalg.norm(y_filtered) + 1e-12)
    corr = num / denom

    ok = (rel_diff <= max_dc_diff) and (corr >= min_corr)

    return SimilarityResult(ok=ok, rel_dc_diff=rel_diff, corr=corr)


def savgol_filter(
    y: np.ndarray, window_length: int = 11, polyorder: int = 3, mode: str = "interp"
) -> np.ndarray:
    """Smooth a 1D signal using the Savitzky-Golay filter.

    Args:
        y: Input signal values.
        window_length: Length of the filter window (must be odd and > polyorder).
        polyorder: Order of the polynomial used to fit the samples.
        mode: Padding mode passed to ``scipy.signal.savgol_filter``.

    Returns:
        Smoothed signal values.
    """
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd.")
    if window_length <= polyorder:
        raise ValueError("window_length must be greater than polyorder.")

    y_smooth = scipy.signal.savgol_filter(y, window_length, polyorder, mode=mode)
    return y_smooth


def choose_savgol_window_auto(
    y: np.ndarray,
    target_reduction: float = 0.3,
    polyorder: int = 3,
    min_len: int = 5,
    max_len: int = 101,
) -> int:
    """Choose the smallest Savitzky-Golay window that sufficiently reduces noise.

    Strategy: measure noise on first differences of y, then
    increase the window until noise is reduced by ``target_reduction``.

    Args:
        y: 1D signal values.
        target_reduction: Desired reduction factor in diff-std (e.g. 0.3 → ÷3).
        polyorder: Polynomial order.
        min_len: Minimum allowed window length.
        max_len: Maximum allowed window length.

    Returns:
        Odd integer window length.
    """
    # Constrain max_len to be strictly less than the length of y
    # (required for mode='interp' in scipy.signal.savgol_filter)
    max_len = min(max_len, len(y) - 1)

    diffs = np.diff(y)
    sigma0 = np.median(np.abs(diffs - np.median(diffs))) / 0.6745

    for win in range(min_len | 1, max_len + 1, 2):  # odd lengths
        if win <= polyorder:
            continue
        if win >= len(y):  # Additional safety check
            break
        y_smooth = scipy.signal.savgol_filter(y, win, polyorder)
        sigma = (
            np.median(np.abs(np.diff(y_smooth) - np.median(np.diff(y_smooth)))) / 0.6745
        )
        if sigma <= target_reduction * sigma0:
            return win

    # Fallback: return largest valid odd window
    fallback = max_len | 1  # Make it odd
    if fallback >= len(y):
        # Need an odd number < len(y)
        fallback = (len(y) - 1) if (len(y) - 1) % 2 == 1 else (len(y) - 2)
    return fallback


def denoise_preserve_shape(
    y: np.ndarray,
    polyorder: int = 3,
    target_reduction: float = 0.3,
    max_dc_diff: float = 1e-2,
    min_corr: float = 0.99,
    min_len: int = 5,
    max_len: int = 101,
) -> tuple[np.ndarray, SimilarityResult]:
    """Denoise a signal while preserving slow variations.

    Strategy:
        1. Estimate noise on first differences.
        2. Choose the smallest Savitzky-Golay window that reduces noise
           by at least ``target_reduction``.
        3. Apply the filter.
        4. Check similarity with the original signal (DC and correlation).
        5. Return filtered signal if ok, otherwise return original.

    Args:
        y: Input signal values.
        polyorder: Polynomial order of Savitzky-Golay filter.
        target_reduction: Desired noise reduction factor (0.3 → ÷3).
        max_dc_diff: Maximum allowed relative change in mean value.
        min_corr: Minimum allowed correlation between signals.
        min_len: Minimum window length.
        max_len: Maximum window length.

    Returns:
        A tuple ``(y_denoised, result)`` where ``y_denoised`` is either the
        filtered signal or the original if similarity criteria are not met, and
        ``result`` contains the details of the similarity check.
    """
    win = choose_savgol_window_auto(
        y,
        target_reduction=target_reduction,
        polyorder=polyorder,
        min_len=min_len,
        max_len=max_len,
    )
    y_smooth = savgol_filter(y, window_length=win, polyorder=polyorder, mode="interp")
    result = signal_similarity(y, y_smooth, max_dc_diff=max_dc_diff, min_corr=min_corr)
    if not result.ok:
        y_smooth = y
    return y_smooth, result
