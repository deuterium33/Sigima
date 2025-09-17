# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Pulse analysis (see parent package :mod:`sigima.tools.signal`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import abc
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.optimize  # type: ignore
import scipy.special

from sigima.enums import SignalShape
from sigima.tools.checks import check_1d_arrays
from sigima.tools.signal import features, peakdetection


class PulseFitModel(abc.ABC):
    """Base class for 1D pulse fit models"""

    @classmethod
    @abc.abstractmethod
    def func(cls, x, amp, sigma, x0, y0):
        """Return fitting function"""

    # pylint: disable=unused-argument
    @classmethod
    def get_amp_from_amplitude(cls, amplitude, sigma):
        """Return amp from function amplitude and sigma"""
        return amplitude

    @classmethod
    def amplitude(cls, amp, sigma):
        """Return function amplitude"""
        return cls.func(0, amp, sigma, 0, 0)

    @classmethod
    @abc.abstractmethod
    def fwhm(cls, amp, sigma):
        """Return function FWHM"""

    @classmethod
    def half_max_segment(cls, amp, sigma, x0, y0):
        """Return segment coordinates for y=half-maximum intersection"""
        hwhm = 0.5 * cls.fwhm(amp, sigma)
        yhm = 0.5 * cls.amplitude(amp, sigma) + y0
        return x0 - hwhm, yhm, x0 + hwhm, yhm


class GaussianModel(PulseFitModel):
    """1-dimensional Gaussian fit model"""

    @classmethod
    def func(cls, x, amp, sigma, x0, y0):
        """Return fitting function"""
        return (
            amp / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - x0) / sigma) ** 2)
            + y0
        )

    @classmethod
    def get_amp_from_amplitude(cls, amplitude, sigma):
        """Return amp from function amplitude and sigma"""
        return amplitude * (sigma * np.sqrt(2 * np.pi))

    @classmethod
    def amplitude(cls, amp, sigma):
        """Return function amplitude"""
        return amp / (sigma * np.sqrt(2 * np.pi))

    @classmethod
    def fwhm(cls, amp, sigma):
        """Return function FWHM"""
        return 2 * sigma * np.sqrt(2 * np.log(2))


class LorentzianModel(PulseFitModel):
    """1-dimensional Lorentzian fit model"""

    @classmethod
    def func(cls, x, amp, sigma, x0, y0):
        """Return fitting function"""
        return (amp / (sigma * np.pi)) / (1 + ((x - x0) / sigma) ** 2) + y0

    @classmethod
    def get_amp_from_amplitude(cls, amplitude, sigma):
        """Return amp from function amplitude and sigma"""
        return amplitude * (sigma * np.pi)

    @classmethod
    def amplitude(cls, amp, sigma):
        """Return function amplitude"""
        return amp / (sigma * np.pi)

    @classmethod
    def fwhm(cls, amp, sigma):
        """Return function FWHM"""
        return 2 * sigma


class VoigtModel(PulseFitModel):
    """1-dimensional Voigt fit model"""

    @classmethod
    def func(cls, x, amp, sigma, x0, y0):
        """Return fitting function"""
        # pylint: disable=no-member
        z = (x - x0 + 1j * sigma) / (sigma * np.sqrt(2.0))
        return y0 + amp * scipy.special.wofz(z).real / (sigma * np.sqrt(2 * np.pi))

    @classmethod
    def fwhm(cls, amp, sigma):
        """Return function FWHM"""
        wg = GaussianModel.fwhm(amp, sigma)
        wl = LorentzianModel.fwhm(amp, sigma)
        return 0.5346 * wl + np.sqrt(0.2166 * wl**2 + wg**2)


# MARK: Pulse analysis -----------------------------------------------------------------


class PulseAnalysisError(Exception):
    """Base exception for pulse analysis errors."""


class InvalidSignalError(PulseAnalysisError):
    """Raised when signal data is invalid or insufficient."""


class PolarityDetectionError(PulseAnalysisError):
    """Raised when polarity cannot be determined."""


# @dataclass
# class PulseAnalysisConfig:
#     """Configuration for pulse analysis parameters."""

#     start_range: tuple[float, float] | None = None
#     end_range: tuple[float, float] | None = None
#     plateau_range: tuple[float, float] | None = None
#     signal_shape: SignalShape | str | None = None
#     start_rise_ratio: float = 0.1
#     stop_rise_ratio: float = 0.9
#     z_score_threshold: float = 5.0

#     def __post_init__(self):
#         """Validate configuration parameters."""
#         if self.start_rise_ratio < 0 or self.start_rise_ratio > 1:
#             raise ValueError("start_rise_ratio must be between 0 and 1")
#         if self.stop_rise_ratio < 0 or self.stop_rise_ratio > 1:
#             raise ValueError("stop_rise_ratio must be between 0 and 1")
#         if self.z_score_threshold <= 0:
#             raise ValueError("z_score_threshold must be positive")


@dataclass
class FootInfo:
    """Information about the foot (flat region before rise) of a signal."""

    index: int
    threshold: float
    foot_duration: float
    x_end: float


@dataclass
class PulseFeatures:
    """Extracted features from a pulse signal."""

    signal_shape: SignalShape
    polarity: int
    amplitude: float
    rise_time: float | None
    fall_time: float | None
    fwhm: float | None
    offset: float
    t50: float | None
    tmax: float
    foot_duration: float


def heuristically_recognize_shape(
    x: np.ndarray,
    y: np.ndarray,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
    step_threshold_ratio: float = 0.5,
) -> SignalShape:
    """
    Heuristically determine the shape of the signal: 'step' or 'square'.

    Args:
        x: 1D array of x values (e.g., time or position).
        y: 1D array of y values corresponding to x.
        start_range: Range for the start baseline.
        end_range: Range for the end baseline.
        step_threshold_ratio: Threshold ratio to distinguish step from square pulse.
         If step amplitude > threshold_ratio * total_amplitude, classify as step.

    Returns:
        Signal shape, either SignalShape.STEP or SignalShape.SQUARE.

    Raises:
        InvalidSignalError: If signal data is invalid.
    """
    if x.size != y.size:
        raise InvalidSignalError("x and y arrays must have the same length")
    if x.size < 3:
        raise InvalidSignalError("Signal must have at least 3 data points")

    # if ranges are None, use the first and last points
    if start_range is None:
        start_range = (x[0], x[0])
    if end_range is None:
        end_range = (x[-1], x[-1])

    step_amplitude = get_amplitude(
        x, y, start_range, end_range, signal_shape=SignalShape.STEP
    )
    total_amplitude = np.max(y) - np.min(y)

    if total_amplitude == 0:
        raise InvalidSignalError("Signal has zero amplitude")

    if np.abs(step_amplitude) > np.abs(step_threshold_ratio * total_amplitude):
        signal_shape = SignalShape.STEP
    else:
        signal_shape = SignalShape.SQUARE

    return signal_shape


def _detect_square_polarity(
    x: np.ndarray,
    y: np.ndarray,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
    plateau_range: tuple[float, float] | None = None,
    y_start: float | None = None,
    y_end: float | None = None,
) -> int:
    """Detect the polarity of a square pulse in a signal based on baseline regions.

    Args:
        x: The array of x-values (typically time or sample indices).
        y: The array of y-values (signal amplitudes) corresponding to `x`.
        start_range: The x range for the initial baseline (before the pulse).
        end_range: The x range for the final baseline (after the pulse).
        plateau_range: The x range for the plateau region, if applicable.
         If None, uses the reduced y-values.
        y_start: The y value of the baseline at the start of the pulse.
        y_end: The y value of the baseline at the end of the pulse.

    Returns:
        1 if the pulse is positive, -1 if negative, or 0 if indeterminate.
    """
    if start_range is None:
        start_range = (x[0], x[0])
    if end_range is None:
        end_range = (x[-1], x[-1])

    # reduce x and y outside the base level
    y_red = y[np.logical_and(x >= start_range[1], x <= end_range[0])]

    if plateau_range is None:
        max_y = np.max(y_red)
        min_y = np.min(y_red)
    else:
        max_y = min_y = get_range_mean_y(x, y, plateau_range)
    positive_score = negative_score = 0

    y_start = get_range_mean_y(x, y, start_range) if y_start is None else y_start
    y_end = get_range_mean_y(x, y, end_range) if y_end is None else y_end

    if max_y > y_start and max_y > y_end:
        positive_score = (max_y - y_start) ** 2 + (max_y - y_end) ** 2
    if min_y < y_start and min_y < y_end:
        negative_score = (min_y - y_start) ** 2 + (min_y - y_end) ** 2
    return int(np.sign(positive_score - negative_score))


def detect_polarity(
    x: np.ndarray,
    y: np.ndarray,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
    plateau_range: tuple[float, float] | None = None,
    signal_shape: SignalShape | None = None,
    fraction: float = 0.05,
) -> int:
    """Get step curve polarity.

    Args:
        x: Array of x-values (abscisse).
        y: Array of y-values (ordinate).
        start_range: Range for the start baseline.
        end_range: Range for the end baseline.
        plateau_range: Range for the plateau.
        signal_shape: Shape of the signal.
        fraction: Fraction of the x-range to use for baseline and plateau calculations.

    Returns:
        Polarity of the step (1 for positive, -1 for negative).

    Raises:
        PolarityDetectionError: If polarity cannot be determined.
        ValueError: If signal shape is unknown.
    """
    if start_range is None:
        start_range = get_start_range(x, fraction)
    if end_range is None:
        end_range = get_end_range(x, fraction)

    if signal_shape is None:
        signal_shape = heuristically_recognize_shape(x, y, start_range, end_range)

    y_start = get_range_mean_y(x, y, start_range)
    y_end = get_range_mean_y(x, y, end_range)

    if signal_shape == SignalShape.STEP:
        if y_start < y_end:
            return 1
        if y_start > y_end:
            return -1

        raise PolarityDetectionError(
            "Polarity could not be determined. Check signal data and baseline ranges."
        )
    if signal_shape == SignalShape.SQUARE:
        return _detect_square_polarity(
            x,
            y,
            start_range,
            end_range,
            plateau_range,
            y_start,
            y_end,
        )
    raise ValueError(
        f"\nUnknown signal shape '{signal_shape}'. Use 'step' or 'square'."
    )


def get_start_range(x: np.ndarray, fraction: float = 0.05) -> tuple[float, float]:
    """Get start range based on fraction of x-range.

    Args:
        x: 1D array of x values.
        fraction: Fraction of the x-range to use for the start range.

    Returns:
        Tuple representing the start range (min, max).
    """
    x_fraction = fraction * (x[-1] - x[0])
    return (x[0], x[0] + x_fraction)


def get_end_range(x: np.ndarray, fraction: float = 0.05) -> tuple[float, float]:
    """Get end range based on fraction of x-range.

    Args:
        x: 1D array of x values.
        fraction: Fraction of the x-range to use for the end range.

    Returns:
        Tuple representing the end range (min, max).
    """
    x_fraction = fraction * (x[-1] - x[0])
    return (x[-1] - x_fraction, x[-1])


def get_plateau_range(
    x: np.ndarray, y: np.ndarray, polarity: int, fraction: float = 0.05
) -> tuple[float, float]:
    """Get plateau range around the max y-value based on fraction of x-range.

    Args:
        x: 1D array of x values.
        y: 1D array of y values.
        polarity: Polarity of the signal (1 for positive, -1 for negative).
        fraction: Fraction of the x-range to use for the plateau range.

    Returns:
        Tuple representing the plateau range (min, max).
    """
    x_fraction = fraction * (x[-1] - x[0])
    max_index = np.argmax(y * polarity)
    return (x[max_index] - 0.5 * x_fraction, x[max_index] + 0.5 * x_fraction)


def get_range_mean_y(
    x: np.ndarray,
    y: np.ndarray,
    value_range: tuple[float, float],
) -> float:
    """Get mean y-value in a given x-range.

    Args:
        x: 1D array of x values.
        y: 1D array of y values.
        value_range: Tuple representing the x-range (min, max).

    Returns:
        Mean y-value in the specified x-range.
    """
    return float(np.mean(y[np.logical_and(x >= value_range[0], x <= value_range[1])]))


def get_amplitude(
    x: np.ndarray,
    y: np.ndarray,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
    plateau_range: tuple[float, float] | None = None,
    signal_shape: SignalShape | str | None = None,
    fraction: float = 0.05,
) -> float:
    """Get curve amplitude.

    Args:
        x: 1D array of x values.
        y: 1D array of y values.
        start_range: Range for the start baseline.
        end_range: Range for the end baseline.
        plateau_range: Range for the plateau.
        signal_shape: Shape of the signal.
        fraction: Fraction of the x-range to use for baseline and plateau calculations
         if start, end, or plateau are None.

    Returns:
        Amplitude of the step.
    """
    if signal_shape is None:
        signal_shape = heuristically_recognize_shape(x, y, start_range, end_range)

    if start_range is None:
        start_range = get_start_range(x, fraction)
    if end_range is None:
        end_range = get_end_range(x, fraction)

    if signal_shape == SignalShape.STEP:
        min_level = get_range_mean_y(x, y, start_range)
        max_level = get_range_mean_y(x, y, end_range)
    elif signal_shape == SignalShape.SQUARE:
        try:
            polarity = detect_polarity(
                x, y, start_range, end_range, signal_shape=signal_shape
            )
        except PolarityDetectionError:
            # If polarity cannot be determined, use total amplitude
            return np.max(y) - np.min(y)

        if plateau_range is None:
            plateau_range = get_plateau_range(x, y, polarity, fraction)

        # reverse y if polarity is negative
        y_positive = y * polarity
        # compute base level
        min_level = get_range_mean_y(x, y_positive, start_range)
        max_level = get_range_mean_y(x, y_positive, plateau_range)
    else:
        raise ValueError("Unknown signal type. Use 'step' or 'square'.")

    return np.abs(min_level - max_level)


def get_crossing_ratio_time(
    x: np.ndarray,
    y: np.ndarray,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
    decimal_ratio: float = 0.1,
    fraction: float = 0.05,
) -> float | None:
    """
    Calculates the x-value at which a normalized step signal crosses a specified
    fractional amplitude.

    This function normalizes the input signal `y` relative to the baseline level defined
    by `start` and the amplitude between `start` and `end`. It accounts for the
    polarity of the step (rising or falling) and then finds the x-position where the
    normalized signal crosses the specified `decimalRatio` fraction of the step height.

    Args:
        x: 1D array of x values (e.g., time).
        y: 1D array of y values corresponding to `x`.
        start_range: Tuple defining the start baseline region (initial plateau).
        end_range: Tuple defining the end baseline region (final plateau).
        decimal_ratio: The fractional amplitude (between 0 and 1) at which to find the
         crossing time. For example, 0.5 corresponds to the half-maximum crossing.
        fraction: Fraction of the x-range to use for baseline calculations if
         start_range or end_range are None.

    Returns:
        The x-value where the normalized signal crosses the specified fractional
        amplitude.
    """
    try:
        polarity = detect_polarity(
            x,
            y,
            start_range,
            end_range,
            signal_shape=SignalShape.STEP,
        )
    except PolarityDetectionError as e:
        raise InvalidSignalError(f"Cannot determine crossing time: {e}") from e
    amplitude = get_amplitude(x, y, start_range, end_range)
    y_positive = y * polarity
    if start_range is None:
        start_range = get_start_range(x, fraction)
    y_start = get_range_mean_y(x, y_positive, start_range)
    y_norm = (y_positive - y_start) / amplitude
    roots = features.find_all_x_at_given_y_value(x, y_norm, decimal_ratio)
    if len(roots) == 0:
        return None
    if len(roots) > 1:
        warnings.warn("Multiple crossing points found. Returning first.")
    return roots[0]


def get_step_rise_time(
    x: np.ndarray,
    y: np.ndarray,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
    start_rise_ratio: float = 0.1,
    stop_rise_ratio: float = 0.9,
) -> float | None:
    """Calculates the rise time of a step-like signal between two defined plateaus.

    The rise time is defined as the time it takes for the signal to increase from
    a specified lower fraction (`start_rise_ratio`) to a higher fraction
    (`1 - stop_rise_ratio`) of the total amplitude change between two reference
    regions (e.g., before and after the step transition).

    This function uses `get_crossing_ratio_time` to find both the start and stop
    times of the rise and returns their difference.

    Args:
        x: 1D array of x values (e.g., time).
        y: 1D array of y values corresponding to `x`.
        start_range: Tuple defining the start plateau region (before the rise).
        end_range: Tuple defining the end plateau region (after the rise).
        start_rise_ratio: Fraction of the step height at which the rise starts.
         Default is 0.1 (i.e., 10% of the step height).
        stop_rise_ratio: Fraction from the top of the step to define the end
         of the rise. Default is 0.9 (i.e., 90% of the step height).

    Returns:
        The rise time (difference between the stop and start of the step).
    """
    # start rise
    start_time = get_crossing_ratio_time(x, y, start_range, end_range, start_rise_ratio)

    # stop rise
    stop_time = get_crossing_ratio_time(
        x, y, start_range, end_range, 1 - stop_rise_ratio
    )
    if start_time is None or stop_time is None:
        warnings.warn(
            "Could not determine start or stop time for the step rise. Returning None."
        )
        return None

    return stop_time - start_time


def heuristically_find_foot_end_time(
    x: np.ndarray,
    y: np.ndarray,
    start_range: tuple[float, float],
    z_score_threshold: float = 5,
) -> float | None:
    """
    Finds the first index in the input array where the value deviates significantly
    from the running average of all previous values.

    Uses an optimized approach with running statistics to avoid O(n²) complexity.

    Args:
        x: 1D array of x values (e.g., time).
        y: 1D array of y values corresponding to `x`.
        start_range: Tuple defining the lower plateau region (start of the step).
        z_score_threshold: Number of standard deviations to use as the outlier
         threshold.

    Returns:
        The x-value of the first outlier, or None if no such value is found.

    Raises:
        InvalidSignalError: If insufficient data is provided.
    """
    if y.size < 10:
        raise InvalidSignalError(
            "Insufficient data for statistical analysis (need ≥10 points)"
        )

    start_indices = np.nonzero(x >= start_range[1])[0]
    if len(start_indices) == 0:
        raise InvalidSignalError("No data points found after start_baseline_range")

    start_idx = start_indices[0]
    if start_idx < 2:  # Need at least 2 points for statistics
        start_idx = 2

    # Initialize running statistics
    running_sum = np.sum(y[:start_idx])
    running_sum_sq = np.sum(y[:start_idx] ** 2)
    n = start_idx

    for i in range(start_idx, y.size):
        # Update running statistics with previous value
        if i > start_idx:
            prev_val = y[i - 1]
            running_sum += prev_val
            running_sum_sq += prev_val**2
            n += 1

        # Calculate mean and std efficiently
        mean_prev = running_sum / n
        variance = (running_sum_sq / n) - (mean_prev**2)

        if variance <= 0:
            continue  # Skip if no variability

        std_prev = np.sqrt(variance)

        # Check if current value is an outlier
        if abs(y[i] - mean_prev) >= z_score_threshold * std_prev:
            return x[i]

    return None  # No outlier found


def get_foot_info(
    x: np.ndarray,
    y: np.ndarray,
    start_range: tuple[float, float],
    end_range: tuple[float, float],
    start_rise_ratio: float | None = None,
    end_time: float | None = None,
) -> FootInfo:
    """
    Detects the 'foot' of a rising signal: the region before the rise starts.

    Args:
        x: 1D array of x values (e.g., time or position).
        y: 1D array of y values (same size as x).
        start_range: A range (min, max) representing the initial flat region ("foot").
        end_range: A range (min, max) representing the final high region after the rise.
        start_rise_ratio: Fraction of the rise height to detect the start of the rise.
        end_time: If provided, only consider data up to this x-value.

    Returns:
        FootInfo object containing foot analysis results.

    Raises:
        InvalidSignalError: If foot end time cannot be determined.
    """
    if end_time is None:
        if start_rise_ratio is not None:
            try:
                end_time = get_crossing_ratio_time(
                    x, y, start_range, end_range, start_rise_ratio
                )
            except Exception:
                end_time = None

            if end_time is None:
                try:
                    end_time = heuristically_find_foot_end_time(x, y, start_range)
                except InvalidSignalError:
                    raise InvalidSignalError("Could not determine foot end time")
        else:
            end_time = heuristically_find_foot_end_time(x, y, start_range)

    if end_time is None:
        raise InvalidSignalError("Could not determine foot end time")

    indices = np.nonzero(x >= end_time)[0]
    if len(indices) == 0:
        raise InvalidSignalError("End time is beyond signal range")

    idx = int(indices[0])
    return FootInfo(
        index=idx,
        threshold=float(y[idx]),
        foot_duration=float(end_time - x[0]),
        x_end=float(end_time),
    )


@check_1d_arrays(x_sorted=True)
def full_width_at_y(
    x: np.ndarray,
    y: np.ndarray,
    level: float,
) -> tuple[float, float, float, float]:
    """Compute the full width at a given y level of a square shaped signal using
    zero-crossing method.

    Args:
        x: 1D array of x values.
        y: 1D array of y values.
        level: The Y level at which to compute the width

    Returns:
        Full width segment coordinates
    """
    tmax_idx = np.argmax(y)

    roots1 = features.find_all_x_at_given_y_value(
        x[0 : tmax_idx + 1],
        y[0 : tmax_idx + 1],
        level,
    )
    if len(roots1) > 1:
        warnings.warn("Multiple crossing points found. Returning first.")
    roots2 = features.find_all_x_at_given_y_value(
        x[tmax_idx:],
        y[tmax_idx:],
        level,
    )
    if len(roots2) > 1:
        warnings.warn("Multiple crossing points found. Returning last.")
    t1 = roots1[0] if len(roots1) > 0 else np.nan
    t2 = roots2[-1] if len(roots2) > 0 else np.nan
    return t1, y.dtype.type(level), t2, y.dtype.type(level)


def full_width_at_ratio(
    x: np.ndarray,
    y: np.ndarray,
    ratio: float,
    start_range: tuple[float, float] | None = None,
    end_range: tuple[float, float] | None = None,
    fraction: float = 0.05,
) -> tuple[float, float, float, float]:
    """
    Calculate the full width at a specified ratio of the amplitude for a pulse signal.

    This function determines the two crossing points (x1, x2) where the normalized
    signal crosses a given ratio of its amplitude, and returns these points along with
    the corresponding y-level.

    Args:
        x: 1D array of x-values.
        y: 1D array of y-values.
        ratio: Ratio (between 0 and 1) of the amplitude at which to measure the width.
        start_range: Range of x-values to estimate the start baseline.
        end_range: Range of x-values to estimate the end baseline.
        fraction: Fraction of the x-range to use for baseline calculations if
         start_range or end_range are None.

    Returns:
        (x1, level, x2, level), where x1 and x2 are the crossing points at the specified
        ratio, and level is the y-value at that ratio.

    Raises:
        ValueError: If the amplitude of the signal is zero.
        RuntimeWarning: If the polarity cannot be determined, returns NaN for crossing
         times.

    Notes:
        - The function normalizes the signal based on the detected amplitude and
        polarity.
        - The crossing times are computed using `features.find_first_x_at_given_y_value`
        function.
    """
    amplitude = get_amplitude(x, y, start_range, end_range)

    try:
        polarity = detect_polarity(x, y, start_range, end_range)
    except PolarityDetectionError as e:
        raise InvalidSignalError(f"Cannot determine width at ratio: {e}") from e

    if start_range is None:
        start_range = get_start_range(x, fraction)
    if end_range is None:
        end_range = get_end_range(x, fraction)
    start_baseline = get_range_mean_y(x, y * polarity, start_range)

    if amplitude == 0:
        raise InvalidSignalError(
            "Amplitude of your square signal is zero. Check your data."
        )

    y_norm = np.asarray(polarity * (y - start_baseline) / amplitude, dtype=y.dtype.type)

    level = y.dtype.type(ratio * polarity * amplitude + start_baseline)

    tmax_idx = np.argmax(y_norm)

    roots1 = features.find_all_x_at_given_y_value(
        x[0 : tmax_idx + 1],
        y_norm[0 : tmax_idx + 1],
        ratio,
    )
    if len(roots1) > 1:
        warnings.warn("Multiple crossing points found. Returning first.")
    x1 = roots1[0] if len(roots1) > 0 else np.nan
    roots2 = features.find_all_x_at_given_y_value(
        x[tmax_idx:],
        y_norm[tmax_idx:],
        ratio,
    )
    if len(roots2) > 1:
        warnings.warn("Multiple crossing points found. Returning last.")
    x2 = roots2[-1] if len(roots2) > 0 else np.nan
    return x1, level, x2, level


def fwhm(
    x: np.ndarray,
    y: np.ndarray,
    method: Literal["zero-crossing", "gauss", "lorentz", "voigt"] = "zero-crossing",
    xmin: float | None = None,
    xmax: float | None = None,
) -> tuple[float, float, float, float]:
    """Compute Full Width at Half Maximum (FWHM) of the input data

    Args:
        x: 1D array of x-values.
        y: 1D array of y-values.
        method: Calculation method. Two types of methods are supported: a zero-crossing
         method and fitting methods (based on various models: Gauss, Lorentz, Voigt).
         Defaults to "zero-crossing".
        xmin: Lower X bound for the fitting. Defaults to None (no lower bound,
         i.e. the fitting starts from the first point).
        xmax: Upper X bound for the fitting. Defaults to None (no upper bound,
         i.e. the fitting ends at the last point)

    Returns:
        FWHM segment coordinates
    """
    dx, dy, base = np.max(x) - np.min(x), np.max(y) - np.min(y), np.min(y)
    sigma, mu = dx * 0.1, peakdetection.xpeak(x, y)
    if isinstance(xmin, float):
        indices = np.where(x >= xmin)[0]
        x = x[indices]
        y = y[indices]
    if isinstance(xmax, float):
        indices = np.where(x <= xmax)[0]
        x = x[indices]
        y = y[indices]

    if method == "zero-crossing":
        x1, y1, x2, y2 = full_width_at_ratio(x, y, 0.5)
        return x1, y1, x2, y2

    try:
        fit_model_class: type[PulseFitModel] = {
            "gauss": GaussianModel,
            "lorentz": LorentzianModel,
            "voigt": VoigtModel,
        }[method]
    except KeyError as exc:
        raise ValueError(f"Invalid method {method}") from exc

    def func(params) -> np.ndarray:
        """Fitting model function"""
        # pylint: disable=cell-var-from-loop
        return y - fit_model_class.func(x, *params)

    amp = fit_model_class.get_amp_from_amplitude(dy, sigma)
    (amp, sigma, mu, base), _ier = scipy.optimize.leastsq(
        func, np.array([amp, sigma, mu, base])
    )
    return fit_model_class.half_max_segment(amp, sigma, mu, base)


@check_1d_arrays(x_sorted=True)
def fw1e2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Compute Full Width at 1/e² of the input data (using a Gaussian model fitting).

    Args:
        x: 1D array of x-values.
        y: 1D array of y-values.

    Returns:
        FW at 1/e² segment coordinates
    """
    dx, dy, base = np.max(x) - np.min(x), np.max(y) - np.min(y), np.min(y)
    sigma, mu = dx * 0.1, peakdetection.xpeak(x, y)
    amp = GaussianModel.get_amp_from_amplitude(dy, sigma)
    p_in = np.array([amp, sigma, mu, base])

    def func(params):
        """Fitting model function"""
        # pylint: disable=cell-var-from-loop
        return y - GaussianModel.func(x, *params)

    p_out, _ier = scipy.optimize.leastsq(func, p_in)
    amp, sigma, mu, base = p_out
    hw = 2 * sigma
    yhm = GaussianModel.amplitude(amp, sigma) / np.e**2 + base
    return mu - hw, yhm, mu + hw, yhm


def extract_pulse_features(
    x: np.ndarray,
    y: np.ndarray,
    start_range: tuple[float, float],
    end_range: tuple[float, float],
    start_rise_ratio: float = 0.1,
    stop_rise_ratio: float = 0.9,
    signal_shape: SignalShape | str | None = None,
) -> PulseFeatures:
    """Extract various pulse features from the input signal.

    Args:
        x: 1D array of x values (e.g., time).
        y: 1D array of y values (signal).
        start_range: Interval for the first plateau (baseline).
        end_range: Interval for the second plateau (peak).
        signal_shape: Signal type ('step' or 'square').
        start_rise_ratio: Fraction for rise start.
        stop_rise_ratio: Fraction for rise end.

    Returns:
        Pulse features.
    """
    if signal_shape is None or signal_shape == "auto":
        signal_shape = heuristically_recognize_shape(x, y, start_range, end_range)
    if signal_shape not in (SignalShape.STEP, SignalShape.SQUARE):
        raise ValueError(
            f"\nUnknown signal shape '{signal_shape}'. Use 'step' or 'square'."
        )

    polarity = detect_polarity(x, y, start_range, end_range, signal_shape=signal_shape)
    amplitude = get_amplitude(x, y, start_range, end_range, signal_shape=signal_shape)
    tmax_idx = np.argmax(y)
    tmax = x[tmax_idx]

    if signal_shape == SignalShape.STEP:
        t_rise = get_step_rise_time(
            x,
            y,
            start_range,
            end_range,
            start_rise_ratio=start_rise_ratio,
            stop_rise_ratio=stop_rise_ratio,
        )
        t50 = get_crossing_ratio_time(x, y, start_range, end_range, 0.5)
        foot_info = get_foot_info(x, y, start_range, end_range, start_rise_ratio)
        t_fall = None
        fwhm_value = None
    else:  # is square
        t_rise = get_step_rise_time(
            x[0 : tmax_idx + 1],
            y[0 : tmax_idx + 1],
            start_range=start_range,
            end_range=(x[tmax_idx], x[tmax_idx]),
            start_rise_ratio=start_rise_ratio,
            stop_rise_ratio=stop_rise_ratio,
        )
        t50 = get_crossing_ratio_time(
            x[0 : tmax_idx + 1],
            y[0 : tmax_idx + 1],
            start_range,
            (x[tmax_idx], x[tmax_idx]),
            0.5,
        )
        t_fall = get_step_rise_time(
            x[tmax_idx:],
            y[tmax_idx:],
            start_range=(x[tmax_idx], x[tmax_idx]),
            end_range=end_range,
            start_rise_ratio=start_rise_ratio,
            stop_rise_ratio=stop_rise_ratio,
        )

        foot_info = get_foot_info(
            x[0 : tmax_idx + 1],
            y[0 : tmax_idx + 1],
            start_range=start_range,
            end_range=(x[tmax_idx], x[tmax_idx]),
            start_rise_ratio=start_rise_ratio,
        )
        # fwhm = t50 - t50fall  # half maximum value
        x1, _, x2, _ = fwhm(x, y, "zero-crossing")
        fwhm_value = x2 - x1  # full width at half maximum
        mean_x_sampling_time = float(np.mean(np.diff(x)))
        if fwhm_value <= 10 * mean_x_sampling_time:
            # if the fwhm is smaller than 10 times the mean sampling time, we cannot
            # rely on rising and falling times, as the pulse is too narrow
            t_fall = None
            t_rise = None

    offset = get_range_mean_y(x, y * polarity, start_range)  # baseline

    foot_duration = foot_info.foot_duration

    return PulseFeatures(
        signal_shape=signal_shape,
        polarity=polarity,
        amplitude=amplitude,
        rise_time=t_rise,
        fall_time=t_fall,
        fwhm=fwhm_value,
        offset=offset,
        t50=t50,
        tmax=tmax,
        foot_duration=foot_duration,
    )
