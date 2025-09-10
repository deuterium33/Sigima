# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Pulse analysis (see parent package :mod:`sigima.tools.signal`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import abc
import enum
import warnings
from typing import Literal

import numpy as np
import scipy.optimize  # type: ignore
import scipy.special

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


class SignalShape(str, enum.Enum):
    """Enum for signal shapes."""

    def __str__(self) -> str:
        return self.value

    STEP = "step"
    SQUARE = "square"


def heuristically_recognize_shape(
    x: np.ndarray,
    y: np.ndarray,
    start_basement_range: tuple[float, float] | None = None,
    end_basement_range: tuple[float, float] | None = None,
) -> str:
    """
    Heuristically determine the shape of the signal: 'step' or 'square'.

    Args:
        x: 1D array of x values (e.g., time or position).
        y: 1D array of y values corresponding to x.
        start_basement_range: Range for the first baseline (lower plateau).
        end_basement_range: Range for the second baseline (upper plateau).

    Returns:
        Detected signal shape, either 'step' or 'square'.
    """
    # if ranges are None, use the first and last points
    if start_basement_range is None:
        start_basement_range = (x[0], x[0])
    if end_basement_range is None:
        end_basement_range = (x[-1], x[-1])

    step_amplitude = get_amplitude(
        x, y, start_basement_range, end_basement_range, signal_shape=SignalShape.STEP
    )

    if step_amplitude > 0.5 * (np.max(y) - np.min(y)):
        signal_shape = SignalShape.STEP
    else:
        signal_shape = SignalShape.SQUARE

    return str(signal_shape)


def _detect_square_polarity(
    x: np.ndarray,
    y: np.ndarray,
    start_basement_range: tuple[float, float] | None = None,
    end_basement_range: tuple[float, float] | None = None,
    high_baseline_range: tuple[float, float] | None = None,
    start_basement: float | None = None,
    end_basement: float | None = None,
) -> int:
    """
    Detects the polarity of a square pulse in a signal based on baseline and pulse
    regions.

    Parameters
    ----------
    x : The array of x-values (typically time or sample indices).
    y : The array of y-values (signal amplitudes) corresponding to `x`.
    start_basement_range :
        The (start, end) range in `x` for the initial baseline (before the pulse).
    end_basement_range :
        The (start, end) range in `x` for the final baseline (after the pulse).
    high_baseline_range :
        The (start, end) range in `x` for a high baseline region, if applicable.
        If None, uses the reduced y-values.
    start_basement :
        The baseline value at the start of the pulse.
    end_basement :
        The baseline value at the end of the pulse.

    Returns
    -------
        Returns 1 if the pulse is positive, -1 if negative, or 0 if indeterminate.
    """
    if start_basement_range is None:
        start_basement_range = (x[0], x[0])
    if end_basement_range is None:
        end_basement_range = (x[-1], x[-1])

    # reduce x and y outside the base level
    y_red = y[np.logical_and(x >= start_basement_range[1], x <= end_basement_range[0])]

    if high_baseline_range is None:
        max_y = np.max(y_red)
        min_y = np.min(y_red)
    else:
        max_y = min_y = np.mean(
            y[np.logical_and(x >= high_baseline_range[0], x <= high_baseline_range[1])]
        )
    positive_score = negative_score = 0

    start_basement = (
        np.mean(
            y[
                np.logical_and(
                    x >= start_basement_range[0], x <= start_basement_range[1]
                )
            ]
        )
        if start_basement is None
        else start_basement
    )
    end_basement = (
        np.mean(
            y[np.logical_and(x >= end_basement_range[0], x <= end_basement_range[1])]
        )
        if end_basement is None
        else end_basement
    )

    if max_y > start_basement and max_y > end_basement:
        positive_score = (max_y - start_basement) ** 2 + (max_y - end_basement) ** 2
    if min_y < start_basement and min_y < end_basement:
        negative_score = (min_y - start_basement) ** 2 + (min_y - end_basement) ** 2
    return int(np.sign(positive_score - negative_score))


def detect_polarity(
    x: np.ndarray,
    y: np.ndarray,
    start_basement_range: tuple[float, float] | None = None,
    end_basement_range: tuple[float, float] | None = None,
    high_baseline_range: tuple[float, float] | None = None,
    signal_shape: str | None = None,
):
    """get step curve polarity

    Args:
        x: abscisse
        y: ordinate
        start_basement_range: range first basement
        end_basement_range: range second basement

    Returns:
        amplitude of the step
    """
    if start_basement_range is None:
        start_basement_range = (x[0], x[0])
    if end_basement_range is None:
        end_basement_range = (x[-1], x[-1])

    if signal_shape is None:
        signal_shape = heuristically_recognize_shape(
            x, y, start_basement_range, end_basement_range
        )

    start_basement = np.mean(
        y[np.logical_and(x >= start_basement_range[0], x <= start_basement_range[1])]
    )
    end_basement = np.mean(
        y[np.logical_and(x >= end_basement_range[0], x <= end_basement_range[1])]
    )

    if signal_shape == SignalShape.STEP:
        if start_basement < end_basement:
            return 1
        if start_basement > end_basement:
            return -1

        warnings.warn(
            "Polarity could not be determined. Please check your data.",
            RuntimeWarning,
        )
        return None
    if signal_shape == SignalShape.SQUARE:
        return _detect_square_polarity(
            x,
            y,
            start_basement_range,
            end_basement_range,
            high_baseline_range,
            start_basement,
            end_basement,
        )
    raise ValueError(
        f"\nUnknown signal shape '{signal_shape}'. Use 'step' or 'square'."
    )


def get_amplitude(
    x: np.ndarray,
    y: np.ndarray,
    start_basement_range: tuple[float, float] | None = None,
    end_basement_range: tuple[float, float] | None = None,
    high_baseline_range: tuple[float, float] | None = None,
    signal_shape: str | None = None,
) -> float:
    """get curve amplitude

    Args:
        x: abscisse
        y: ordinate
        start_basement_range: range first basement
        end_basement_range: range second basement

    Returns:
        amplitude of the step
    """
    if signal_shape is None:
        signal_shape = heuristically_recognize_shape(
            x, y, start_basement_range, end_basement_range
        )

    if signal_shape == SignalShape.STEP:
        # compute base level
        min_level = (
            np.mean(
                y[
                    np.logical_and(
                        x >= start_basement_range[0], x <= start_basement_range[1]
                    )
                ]
            )
            if start_basement_range is not None
            else np.min(y)
        )
        max_level = (
            np.mean(
                y[
                    np.logical_and(
                        x >= end_basement_range[0], x <= end_basement_range[1]
                    )
                ]
            )
            if end_basement_range is not None
            else np.max(y)
        )
    elif signal_shape == SignalShape.SQUARE:
        polarity = detect_polarity(
            x, y, start_basement_range, end_basement_range, signal_shape=signal_shape
        )
        if polarity is None:
            return np.max(y) - np.min(y)

        # reverse y if polarity is negative
        y_positive = y * polarity
        # compute base level
        min_level = (
            np.mean(
                y_positive[
                    np.logical_and(
                        x >= start_basement_range[0], x <= start_basement_range[1]
                    )
                ]
            )
            if start_basement_range is not None
            else np.min(y_positive)
        )
        max_level = (
            np.mean(
                y_positive[
                    np.logical_and(
                        x >= high_baseline_range[0], x <= high_baseline_range[1]
                    )
                ]
            )
            if high_baseline_range is not None
            else np.max(y_positive)
        )
    else:
        raise ValueError("Unknown signal type. Use 'step' or 'square'.")

    return np.abs(min_level - max_level)


def get_crossing_ratio_time(
    x: np.ndarray,
    y: np.ndarray,
    start_basement_range: tuple[float, float] | None = None,
    end_basement_range: tuple[float, float] | None = None,
    decimal_ratio: float = 0.1,
) -> float | None:
    """
    Calculates the x-value at which a normalized step signal crosses a specified
    fractional amplitude.

    This function normalizes the input signal `y` relative to the baseline level defined
    by `start_basement_range` and the amplitude between `start_basement_range` and
    `end_basement_range`. It accounts for the
    polarity of the step (rising or falling) and then finds the x-position where the
    normalized signal crosses the specified `decimalRatio` fraction of the step height.

    Args:
        x: 1D array of x values (e.g., time).
        y: 1D array of y values corresponding to `x`.
        start_basement_range: Tuple defining the lower plateau region (baseline).
        end_basement_range: Tuple defining the upper plateau region (peak).
        decimalRatio: The fractional amplitude (between 0 and 1) at which to find the
            crossing time.
            For example, 0.5 corresponds to the half-maximum crossing.

    Returns:
        The x-value where the normalized signal crosses the specified fractional
        amplitude.
    """
    polarity = detect_polarity(
        x, y, start_basement_range, end_basement_range, signal_shape=SignalShape.STEP
    )
    if polarity is None:
        warnings.warn(
            "Polarity could not be determined. Returning None for crossing time."
        )
        return None
    amplitude = get_amplitude(x, y, start_basement_range, end_basement_range)
    y_positive = y * polarity
    start_basement = (
        np.mean(
            y_positive[
                np.logical_and(
                    x >= start_basement_range[0], x <= start_basement_range[1]
                )
            ]
        )
        if start_basement_range is not None
        else np.min(y_positive)
    )
    y_norm = (y_positive - start_basement) / amplitude
    return features.get_crossing_time(x, y_norm, decimal_ratio)


def get_step_rise_time(
    x: np.ndarray,
    y: np.ndarray,
    start_basement_range: tuple[float, float] | None = None,
    end_basement_range: tuple[float, float] | None = None,
    start_rise_ratio: float = 0.1,
    stop_rise_ratio: float = 0.1,
) -> float | None:
    """
    Calculates the rise time of a step-like signal between two defined plateaus.

    The rise time is defined as the time it takes for the signal to increase from
    a specified lower fraction (`start_rise_ratio`) to a higher fraction
    (`1 - stop_rise_ratio`) of the total amplitude change between two reference
    regions (e.g., before and after the step transition).

    This function uses `get_crossing_ratio_time` to find both the start and stop
    times of the rise and returns their difference.

    Args:
        x: 1D array of x values (e.g., time).
        y: 1D array of y values corresponding to `x`.
        start_basement_range: Tuple defining the lower plateau region (before the rise).
        end_basement_range: Tuple defining the upper plateau region (after the rise).
        start_rise_ratio: Fraction of the step height at which the rise starts.
            Default is 0.1 (i.e., 10% of the step height).
        stop_rise_ratio: Fraction from the top of the step to define the end
            of the rise. Default is 0.1 (i.e., 90% of the step height).

    Returns:
        The rise time (difference between the stop and start of the step).
    """
    # start rise
    start_time = get_crossing_ratio_time(
        x, y, start_basement_range, end_basement_range, start_rise_ratio
    )

    # stop rise
    stop_time = get_crossing_ratio_time(
        x, y, start_basement_range, end_basement_range, 1 - stop_rise_ratio
    )
    if start_time is None or stop_time is None:
        warnings.warn(
            "Could not determine start or stop time for the step rise. Returning None."
        )
        return None

    return stop_time - start_time


def get_step_time_at_half_maximum(
    x: np.ndarray,
    y: np.ndarray,
    start_basement_range: tuple[float, float],
    end_basement_range: tuple[float, float],
) -> float | None:
    """
    Estimates the time at which a rising signal reaches half of its total step height.

    This function computes the time at which the signal crosses 50% of the amplitude
    difference between two reference regions (typically representing the initial and
    final plateaus of a step-like transition). It uses `get_crossing_ratio_time` with
    a fixed ratio of 0.5 to identify the midpoint of the rise.

    Args:
        x: 1D array of x values (e.g., time).
        y: 1D array of y values corresponding to `x`.
        start_basement_range: Tuple defining the lower plateau region (start of
            the step).
        end_basement_range: Tuple defining the upper plateau region (end of the step).

    Returns:
        The x-value (e.g., time) at which the signal reaches 50% of the step height.
    """
    # start rise
    time_at_half_maximum = get_crossing_ratio_time(
        x, y, start_basement_range, end_basement_range, 0.5
    )
    return time_at_half_maximum


def get_step_start_time(
    x: np.ndarray,
    y: np.ndarray,
    start_basement_range: tuple[float, float],
    end_basement_range: tuple[float, float],
    start_rise_ratio: float = 0.1,
) -> float | None:
    """
    Estimates the time at which a rising signal begins its step transition.

    This function computes the time at which the signal reaches a specified fraction
    of the rise between two reference regions.

    Args:
        x: 1D array of x values (e.g., time).
        y: 1D array of y values corresponding to `x`.
        start_basement_range: Tuple defining the lower plateau region (before the rise).
        end_basement_range: Tuple defining the upper plateau region (after the rise).
        start_rise_ratio: Ratio (between 0 and 1) that determines when the
            rise is considered to have started. Default is 0.1 (i.e., 10% of the full
            step height).

    Returns:
        The x-value (e.g., time) at which the signal starts rising significantly.
    """
    stop_time = get_crossing_ratio_time(
        x, y, start_basement_range, end_basement_range, start_rise_ratio
    )

    return stop_time


def get_step_end_time(
    x: np.ndarray,
    y: np.ndarray,
    start_basement_range: tuple[float, float],
    end_basement_range: tuple[float, float],
    stop_rise_ratio: float = 0.1,
) -> float | None:
    """
    Estimates the time at which a rising signal reaches the end of its step (plateau).

    This function computes the time at which the signal reaches a specified percentage
    (typically close to 1) of the rise between two reference ranges.

    Args:
        x: 1D array of x values (e.g., time).
        y: 1D array of y values corresponding to `x`.
        start_basement_range: Tuple defining the lower plateau region (start of the
            step).
        end_basement_range: Tuple defining the upper plateau region (end of the step).
        stop_rise_ratio: Ratio (between 0 and 1) used to determine the end
            of the rise. For example, 0.1 corresponds to detecting when the signal has
            reached 90% of its step (1 - 0.1). Default is 0.1.

    Returns:
        The x-value (e.g., time) at which the signal reaches the end of the step.
    """
    stop_time = get_crossing_ratio_time(
        x, y, start_basement_range, end_basement_range, 1 - stop_rise_ratio
    )

    return stop_time


def heuristically_find_foot_end_time(
    x: np.ndarray,
    y: np.ndarray,
    start_basement_range: tuple[float, float],
    z_score_threshold: float = 5,
) -> float | None:
    """
    Finds the first index in the input array where the value deviates significantly
    from the running average of all previous values.

    A value is considered an outlier if it differs from the mean of all preceding values
    by more than `z_score_threshold` times the standard deviation of those values.

    Args:
        x: 1D array of x values (e.g., time).
        y: 1D array of y values corresponding to `x`.
        start_basement_range: Tuple defining the lower plateau region (start of the
            step).
        z_score_threshold: Number of standard deviations to use as the outlier
            threshold.

    Returns:
        The index of the first outlier, or None if no such value is found.
    """
    data_size = y.size

    if data_size < 10:
        return None  # Not enough data to compute statistics

    for i in range(np.nonzero(x >= start_basement_range[1])[0][0], data_size):
        # Compute the mean and standard deviation of all previous values
        mean_prev = np.mean(y[:i])
        std_prev = np.std(y[:i])

        if std_prev == 0:
            continue  # Skip this index if standard deviation is zero (no variability)

        # Check if the current value deviates more than 3 standard deviations from the
        # previous mean
        if abs(y[i] - mean_prev) >= z_score_threshold * std_prev:
            return x[i]  # Return the first index that violates the condition

    return None  # No outlier found


def get_foot_info(
    x: np.ndarray,
    y: np.ndarray,
    start_basement_range: tuple[float, float],
    end_basement_range: tuple[float, float],
    start_rise_ratio: float | None = None,
    end_time: float | None = None,
) -> dict[str, float | int]:
    """
    Detects the 'foot' of a rising signal: the region before the rise starts.

    Args:
        x: 1D array of x values (e.g., time or position).
        y: 1D array of y values (same size as x).
        start_basement_range: A range (min, max) representing the initial flat region
            ("foot").
        end_basement_range: A range (min, max) representing the final high region after
            the rise.
        start_rise_ratio: Fraction of the rise height to detect the start of the rise.
        end_time (optional): If provided, only consider data up to this x-value.

    Returns:
        dict: A dictionary with:
            - 'index': Index where the signal reaches the end of the foot.
            - 'threshold': y[index], the ordinate of the end of the foot.
            - 'foot_duration': Duration of the foot region (flat before the rise).
            - 'x_end': x-value where the signal starts to rise.
    """
    if end_time is None:
        if start_rise_ratio is not None:
            end_time = get_crossing_ratio_time(
                x, y, start_basement_range, end_basement_range, start_rise_ratio
            )
            if end_time is None:
                warnings.warn(
                    "Could not determine end time for the foot. Using heuristic method."
                )
                end_time = heuristically_find_foot_end_time(x, y, start_basement_range)
        else:
            heuristically_find_foot_end_time(x, y, start_basement_range)

    idx = int(np.nonzero(x >= end_time)[0][0]) if end_time is not None else len(x) - 1
    return {
        "index": idx,
        "threshold": y[idx] if end_time is not None else y.dtype.type(np.nan),
        "foot_duration": -x[0] + end_time,
        "x_end": x.dtype.type(end_time),
    }


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

    t1 = features.get_crossing_time(
        x[0 : tmax_idx + 1],
        y[0 : tmax_idx + 1],
        level,
    )
    t2 = features.get_crossing_time(
        x[tmax_idx:-1],
        y[tmax_idx:-1],
        level,
    )
    return t1, y.dtype.type(level), t2, y.dtype.type(level)


def full_width_at_ratio(
    x: np.ndarray,
    y: np.ndarray,
    ratio: float,
    start_basement_range: tuple[float, float] | None = None,
    end_basement_range: tuple[float, float] | None = None,
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
        start_basement_range: Range of x-values to estimate the baseline at the start
            of the signal.
        end_basement_range: Range of x-values to estimate the baseline at the end
            of the signal.

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
        - The crossing times are computed using the `features.get_crossing_time`
          function.
    """
    amplitude = get_amplitude(x, y, start_basement_range, end_basement_range)
    polarity = detect_polarity(x, y, start_basement_range, end_basement_range)

    start_basement = np.mean(
        y[np.logical_and(x >= start_basement_range[0], x <= start_basement_range[1])]
        if start_basement_range is not None
        else np.min(y),
        dtype=y.dtype.type,
    )

    if amplitude == 0:
        raise ValueError("Amplitude of your square signal is zero. Check your data.")
    if polarity is None:
        warnings.warn(
            "Polarity could not be determined. Returning NaN for crossing times.",
            RuntimeWarning,
        )
        return (
            x.dtype.type(np.nan),
            y.dtype.type(np.nan),
            x.dtype.type(np.nan),
            y.dtype.type(np.nan),
        )

    y_norm = np.asarray(polarity * (y - start_basement) / amplitude, dtype=y.dtype.type)

    level = y.dtype.type(ratio * polarity * amplitude + start_basement)

    tmax_idx = np.argmax(y_norm)

    x1 = features.get_crossing_time(
        x[0 : tmax_idx + 1],
        y_norm[0 : tmax_idx + 1],
        ratio,
    )
    x2 = features.get_crossing_time(
        x[tmax_idx:-1],
        y_norm[tmax_idx:-1],
        ratio,
    )
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


def get_parameters(
    x: np.ndarray,
    y: np.ndarray,
    start_basement_range: tuple[float, float],
    end_basement_range: tuple[float, float],
    start_rise_ratio: float = 0.1,
    stop_rise_ratio: float = 0.1,
    signal_shape: str | None = None,
) -> dict[str, float | int | None]:
    """
    Compute characteristic parameters of a step or square signal.

    Args:
        x: 1D array of x values (e.g., time).
        y: 1D array of y values (signal).
        start_basement_range: Interval for the first plateau (baseline).
        end_basement_range: Interval for the second plateau (peak).
        signal_shape: Signal type ('step' or 'square').
        start_rise_ratio: Fraction for rise start. Default is 0.1.
        stop_rise_ratio: Fraction for rise end. Default is 0.1.

    Returns:
        Dictionary with the following parameters:
            * polarity: Polarity of the step (+1 or -1).
            * amplitude: Amplitude of the step.
            * rise_time: Time for the signal to rise (rise time).
            * fall_time: Time for the signal to fall (fall time, only for 'square').
            * fwhm: Full width at half maximum (only for 'square').
            * offset: Baseline value (mean of start_basement_range).
            * t50: Time when the signal passes over the half maximum.
            * tmax: Time when the signal reaches the maximum.
            * foot_end_time: Final time of the foot (flat region before the rise).
    """
    if signal_shape is None or signal_shape == "auto":
        signal_shape = heuristically_recognize_shape(
            x, y, start_basement_range, end_basement_range
        )
    if signal_shape not in (SignalShape.STEP, SignalShape.SQUARE):
        raise ValueError(
            f"\nUnknown signal shape '{signal_shape}'. Use 'step' or 'square'."
        )

    polarity = detect_polarity(
        x, y, start_basement_range, end_basement_range, signal_shape=signal_shape
    )
    amplitude = get_amplitude(
        x, y, start_basement_range, end_basement_range, signal_shape=signal_shape
    )
    tmax_idx = np.argmax(y)
    tmax = x[tmax_idx]

    if signal_shape == SignalShape.STEP:
        t_rise = get_step_rise_time(
            x,
            y,
            start_basement_range,
            end_basement_range,
            start_rise_ratio=start_rise_ratio,
            stop_rise_ratio=stop_rise_ratio,
        )
        t50 = get_step_time_at_half_maximum(
            x,
            y,
            start_basement_range,
            end_basement_range,
        )
        foot_info = get_foot_info(
            x, y, start_basement_range, end_basement_range, start_rise_ratio
        )
        t_fall = None
        fwhm_value = None
    elif signal_shape == SignalShape.SQUARE:
        t_rise = get_step_rise_time(
            x[0 : tmax_idx + 1],
            y[0 : tmax_idx + 1],
            start_basement_range=start_basement_range,
            end_basement_range=(x[tmax_idx], x[tmax_idx]),
            start_rise_ratio=start_rise_ratio,
            stop_rise_ratio=stop_rise_ratio,
        )
        t50 = get_step_time_at_half_maximum(
            x[0 : tmax_idx + 1],
            y[0 : tmax_idx + 1],
            start_basement_range=start_basement_range,
            end_basement_range=(x[tmax_idx], x[tmax_idx]),
        )
        t_fall = get_step_rise_time(
            x[tmax_idx:-1],
            y[tmax_idx:-1],
            start_basement_range=(x[tmax_idx], x[tmax_idx]),
            end_basement_range=end_basement_range,
            start_rise_ratio=start_rise_ratio,
            stop_rise_ratio=stop_rise_ratio,
        )

        foot_info = get_foot_info(
            x[0 : tmax_idx + 1],
            y[0 : tmax_idx + 1],
            start_basement_range=start_basement_range,
            end_basement_range=(x[tmax_idx], x[tmax_idx]),
            start_rise_ratio=start_rise_ratio,
        )
        # fwhm = t50 - t50fall  # half maximum value
        x1, _, x2, _ = fwhm(x, y, "zero-crossing")
        fwhm_value = x2 - x1  # full width at half maximum
        mean_x_sampling_time = float(np.mean(np.diff(x)))
        if fwhm_value <= 10 * mean_x_sampling_time:
            # if the fwhm is larger than 10 times the mean sampling time, we cannot rely
            # on rising and falling times, as they are not well measured
            t_fall = None
            t_rise = None
    else:
        raise ValueError("Unknown signal type. Use 'step' or 'square'.")

    offset = np.mean(
        y[np.logical_and(x >= start_basement_range[0], x <= start_basement_range[1])]
    )

    foot_duration = foot_info.get("foot_duration", None)

    return {
        "signal_shape": signal_shape,
        "polarity": polarity,
        "amplitude": amplitude,
        "rise_time": t_rise,
        "fall_time": t_fall,
        "fwhm": fwhm_value,
        "offset": offset,
        "t50": t50,
        "tmax": tmax,
        "foot_duration": foot_duration,
    }
