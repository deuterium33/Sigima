# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Features (see parent package :mod:`sigima.algorithms.signal`)
"""

from __future__ import annotations

import numpy as np

from sigima.tools.checks import check_1d_array, check_1d_arrays


@check_1d_array(min_size=2, finite_only=True)
def find_zero_crossings(y: np.ndarray) -> np.ndarray:
    """Find the left indices of the zero-crossing intervals in the given array.

    Args:
        y: Input array.

    Returns:
        An array of indices where zero-crossings occur.
    """
    zero_crossing_indices = np.nonzero(np.diff(np.sign(y)))[0]
    return zero_crossing_indices


@check_1d_arrays(x_sorted=True)
def find_first_x_at_y_value(x: np.ndarray, y: np.ndarray, y_value: float) -> float:
    """Find the first x value where the signal reaches a given y value (interpolated).

    Args:
        x: x signal data
        y: y signal data (possibly non-monotonic)
        y_value: the y value to find the corresponding x value for

    Returns:
        The first interpolated x value at the given y, or `nan` if not found
    """
    if y_value < np.nanmin(y) or y_value > np.nanmax(y):
        return np.nan  # out of bounds

    for i in range(len(y) - 1):
        y1, y2 = y[i], y[i + 1]
        if np.isnan(y1) or np.isnan(y2):
            continue  # skip bad segments

        if (y1 <= y_value <= y2) or (y2 <= y_value <= y1):
            x1, x2 = x[i], x[i + 1]
            if y1 == y2:
                return x1  # flat segment, arbitrary choice
            # Linear interpolation
            return x1 + (y_value - y1) * (x2 - x1) / (y2 - y1)

    return np.nan  # not found


@check_1d_arrays(x_sorted=True)
def find_y_at_x_value(x: np.ndarray, y: np.ndarray, x_value: float) -> float:
    """Find the y value at a given x value using linear interpolation.

    Args:
        x: Monotonic X data
        y: Y data (may contain NaNs)
        x_value: The x value to find the corresponding y value for

    Returns:
        The interpolated y value at the given x, or `nan` if not computable
    """
    if np.isnan(x_value):
        return np.nan

    # Filter out NaNs
    valid = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid]
    y_valid = y[valid]

    if len(x_valid) == 0 or x_value < x_valid[0] or x_value > x_valid[-1]:
        return np.nan

    return float(np.interp(x_value, x_valid, y_valid))


def find_x_at_value(x: np.ndarray, y: np.ndarray, y0: float) -> np.ndarray:
    """Find the :math:`x_n` values where :math:`y(x_n)` intercepts :math:`y_0`

    This function use a linear interpolation to deduce the precise x values.

    Args:
        x: X data
        y: Y data
        y0: intercept value to find the corresponding x values for

    Returns:
        An array of x values where the y value is the closest to the given value
        (empty array if no zero crossing is found)
    """

    leveled_y = y - y0
    xi_before = find_zero_crossings(leveled_y)

    if len(xi_before) == 0:
        # Return an empty array if no zero crossing is found
        return np.array([])

    # linear interpolation
    xi_after = xi_before + 1
    slope = (y[xi_after] - y[xi_before]) / (x[xi_after] - x[xi_before])
    x0 = (-y[xi_before] + y0) / slope + x[xi_before]
    return x0


@check_1d_arrays(x_evenly_spaced=True)
def bandwidth(
    x: np.ndarray, y: np.ndarray, level: float = 3.0
) -> tuple[float, float, float, float]:
    """Compute the bandwidth of the signal at a given level.

    Args:
        x: X data
        y: Y data
        level: Level in dB at which the bandwidth is computed. Defaults to 3.0.

    Returns:
        Bandwidth of the signal at the given level: segment coordinates
    """
    half_max: float = np.max(y) - level
    bw = find_x_at_value(x, y, half_max)[0]
    coords = (x[0], half_max, bw, half_max)
    return coords


def contrast(y: np.ndarray) -> float:
    """Compute contrast

    Args:
        y: Input array

    Returns:
        Contrast
    """
    max_, min_ = np.max(y), np.min(y)
    return (max_ - min_) / (max_ + min_)
