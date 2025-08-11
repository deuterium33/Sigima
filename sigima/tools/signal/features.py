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
def find_x_intercepts(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Find the :math:`x_n` values where :math:`y = f(x)` intercepts the x-axis.

    This function uses zero-crossing detection and interpolation to find the x values
    where :math:`y = 0`.

    Args:
        x: X data.
        y: Y data.

    Returns:
        Array of x-intercepts. The array is empty if no intercept is found.
    """
    # Find zero crossings.
    xi_before = find_zero_crossings(y)
    if len(xi_before) == 0:
        return np.array([])
    # Interpolate to find x values at zero crossings.
    xi_after = xi_before + 1
    slope = (y[xi_after] - y[xi_before]) / (x[xi_after] - x[xi_before])
    with np.errstate(divide="ignore"):
        x0 = -y[xi_before] / slope + x[xi_before]
        mask = ~np.isfinite(x0)
        x0[mask] = xi_before[mask]
    return x0


@check_1d_arrays(x_sorted=True)
def find_first_x_at_given_y_value(x: np.ndarray, y: np.ndarray, y0: float) -> float:
    """Find the first x value where :math:`y = f(x)` equals the value :math:`y_0`.

    Args:
        x: X data.
        y: Y data.
        y0: Target y value.

    Returns:
        The first interpolated x value at the given :math:`y_0`, or `nan` if none found.
    """
    x_values = find_all_x_at_given_y_value(x, y, y0)
    return x_values[0] if len(x_values) > 0 else np.nan


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


@check_1d_arrays
def find_all_x_at_given_y_value(x: np.ndarray, y: np.ndarray, y0: float) -> np.ndarray:
    """Find the first x value where :math:`y = f(x)` equals the value :math:`y_0`.

    Args:
        x: X data.
        y: Y data.
        y0: Target value.

    Returns:
        Array of values where :math:`y = f(x)` equals :math:`y_0`.
    """
    return find_x_intercepts(x, y - y0)


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
    bw = find_all_x_at_given_y_value(x, y, half_max)[0]
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
