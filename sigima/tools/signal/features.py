# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Features (see parent package :mod:`sigima.algorithms.signal`)
"""

from __future__ import annotations

import warnings

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


@check_1d_arrays(x_min_size=2, x_finite_only=True, x_sorted=True)
def find_y_at_given_x_value(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    """Return the y value at a specified x value using linear interpolation.

    Args:
        x: X data.
        y: Y data.
        x0: Input x value.

    Returns:
        Interpolated y value at x0, or `nan` if input value is not within the
        interpolation range.
    """
    if np.isnan(x0):
        return np.nan
    return float(np.interp(x0, x, y, left=np.nan, right=np.nan))


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


def get_crossing_time(
    x: np.ndarray,
    y: np.ndarray,
    level: float,
    warn_if_multiple: bool = True,
) -> float:
    """
    Finds the x-value at which the signal, supposed monotonic crosses a given threshold
    level. y can be noisy, an estimation of crossing time is returned in such case.

    This function detects zero-crossings in the shifted signal `y - level`,
    and estimates the crossing time as the average of the x-values around
    the first and last crossing points. It assumes a monotonic transition
    (e.g., a single step or slope).

    Args:
        x: 1D array of x values (e.g., time).
        y: 1D array of y values corresponding to `x`.
        level: Threshold level at which to detect the crossing.

    Returns:
        Estimated x-value where the signal crosses the specified level.
    """
    y_shift = y - level
    start_intersections_index = find_zero_crossings(y_shift)
    if len(start_intersections_index) == 0:
        warnings.warn(
            "No zero-crossing found: signal does not cross the specified level.",
            RuntimeWarning,
        )
        return np.nan
    if len(start_intersections_index) == 1:
        # If only one crossing is found, return the corresponding x value
        return x[start_intersections_index[0]]
    # If multiple crossings are found, average the x values around the first and last
    # crossing
    if warn_if_multiple:
        warnings.warn(
            "Multiple zero-crossings found: averaging the x values"
            " around the first and last crossing.",
            UserWarning,
        )
    # Average the x values at the first and last crossing points
    start_time = np.mean(
        [x[start_intersections_index[0]], x[start_intersections_index[-1] + 1]]
    )
    return start_time


@check_1d_arrays(x_evenly_spaced=True)
def bandwidth(
    x: np.ndarray, y: np.ndarray, threshold: float = 3.0
) -> tuple[float, float, float, float]:
    """Compute the bandwidth of the signal at a given threshold relative to the maximum.

    Args:
        x: X data.
        y: Y data.
        threshold: Threshold in decibel (relative to the maximum) at which the bandwidth
        is computed. Defaults to 3.0 dB.

    Returns:
        Segment coordinates of the bandwidth of the signal at the given threshold.
    """
    one_crossing = 1
    two_crossings = 2
    level: float = np.max(y) + threshold
    crossings = find_all_x_at_given_y_value(x, y, level)
    if len(crossings) == one_crossing:
        if x[np.argmax(y)] < crossings[0]:  # Baseband bandwidth
            coords = (0.0, level, crossings[0], level)
        else:
            coords = (crossings[0], level, x[-1], level)
    elif len(crossings) == two_crossings:  # Passband bandwidth
        coords = (crossings[0], level, crossings[1], level)
    else:
        coords = (np.nan, np.nan, np.nan, np.nan)
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
