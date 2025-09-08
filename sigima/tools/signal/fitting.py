# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Curve Fitting Algorithms
=========================

This module provides curve fitting functions without GUI dependencies.
The functions take x,y data and return fitted curves and parameters.

These functions are designed to be used programmatically and in tests,
providing the core fitting algorithms without PlotPy GUI components.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import scipy.optimize
import scipy.special

from sigima.tools.signal import fitmodels, peakdetection


def _fit_with_scipy(x, y, model_func, initial_params, bounds=None):
    """Generic fitting function using scipy.optimize.curve_fit

    Args:
        x: x data
        y: y data
        model_func: fitting function
        initial_params: initial parameter guess
        bounds: parameter bounds (min, max) tuples

    Returns:
        tuple: (fitted_y_values, fitted_parameters)
    """
    if bounds is not None:
        # Convert bounds to scipy format
        lower_bounds = [b[0] for b in bounds]
        upper_bounds = [b[1] for b in bounds]
        bounds_scipy = (lower_bounds, upper_bounds)
    else:
        bounds_scipy = (-np.inf, np.inf)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=scipy.optimize.OptimizeWarning)
        popt, _ = scipy.optimize.curve_fit(
            model_func, x, y, p0=initial_params, bounds=bounds_scipy, maxfev=5000
        )

    fitted_y = model_func(x, *popt)
    return fitted_y, popt


@dataclasses.dataclass
class LinearParams:
    """Linear fit parameters: y = a*x + b"""

    a: float  # slope
    b: float  # intercept


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, LinearParams]:
    """Compute linear fit: y = a*x + b.

    Args:
        x: x data array
        y: y data array

    Returns:
        tuple: (fitted_y_values, LinearParams)
    """
    # Use numpy polyfit for linear regression (more robust)
    coeffs = np.polyfit(x, y, 1)
    a, b = coeffs

    fitted_y = a * x + b
    params = LinearParams(a=a, b=b)

    return fitted_y, params


@dataclasses.dataclass
class PolynomialParams:
    """Polynomial fit parameters"""

    coeffs: list[float]  # polynomial coefficients
    degree: int  # polynomial degree


def polynomial_fit(
    x: np.ndarray, y: np.ndarray, degree: int = 2
) -> tuple[np.ndarray, PolynomialParams]:
    """Compute polynomial fit.

    Args:
        x: x data array
        y: y data array
        degree: polynomial degree

    Returns:
        tuple: (fitted_y_values, PolynomialParams)
    """
    coeffs = np.polyfit(x, y, degree)
    fitted_y = np.polyval(coeffs, x)
    params = PolynomialParams(coeffs=coeffs.tolist(), degree=degree)

    return fitted_y, params


@dataclasses.dataclass
class GaussianParams:
    """Gaussian fit parameters"""

    amp: float  # amplitude parameter (area under curve)
    sigma: float  # standard deviation
    x0: float  # center position
    y0: float  # baseline offset


def gaussian_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, GaussianParams]:
    """Compute Gaussian fit.

    Args:
        x: x data array
        y: y data array

    Returns:
        tuple: (fitted_y_values, GaussianParams)
    """
    # Parameter estimation
    dx = np.max(x) - np.min(x)
    dy = np.max(y) - np.min(y)
    y_min = np.min(y)

    sigma_guess = dx * 0.1
    amp_guess = fitmodels.GaussianModel.get_amp_from_amplitude(dy, sigma_guess)
    x0_guess = peakdetection.xpeak(x, y)
    y0_guess = y_min

    initial_params = [amp_guess, sigma_guess, x0_guess, y0_guess]

    # Parameter bounds
    bounds = [
        (0.0, amp_guess * 2),  # amp
        (sigma_guess * 0.1, sigma_guess * 10),  # sigma
        (np.min(x), np.max(x)),  # x0
        (y_min - 0.2 * dy, y_min + 0.2 * dy),  # y0
    ]

    fitted_y, params_array = _fit_with_scipy(
        x, y, fitmodels.GaussianModel.func, initial_params, bounds
    )

    params = GaussianParams(
        amp=params_array[0],
        sigma=params_array[1],
        x0=params_array[2],
        y0=params_array[3],
    )

    return fitted_y, params


@dataclasses.dataclass
class LorentzianParams:
    """Lorentzian fit parameters"""

    amp: float  # amplitude parameter (area under curve)
    sigma: float  # width parameter
    x0: float  # center position
    y0: float  # baseline offset


def lorentzian_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, LorentzianParams]:
    """Compute Lorentzian fit.

    Args:
        x: x data array
        y: y data array

    Returns:
        tuple: (fitted_y_values, LorentzianParams)
    """
    # Parameter estimation
    dx = np.max(x) - np.min(x)
    dy = np.max(y) - np.min(y)
    y_min = np.min(y)

    sigma_guess = dx * 0.1
    amp_guess = fitmodels.LorentzianModel.get_amp_from_amplitude(dy, sigma_guess)
    x0_guess = peakdetection.xpeak(x, y)
    y0_guess = y_min

    initial_params = [amp_guess, sigma_guess, x0_guess, y0_guess]

    # Parameter bounds
    bounds = [
        (0.0, amp_guess * 2),  # amp
        (sigma_guess * 0.1, sigma_guess * 10),  # sigma
        (np.min(x), np.max(x)),  # x0
        (y_min - 0.2 * dy, y_min + 0.2 * dy),  # y0
    ]

    fitted_y, params_array = _fit_with_scipy(
        x, y, fitmodels.LorentzianModel.func, initial_params, bounds
    )

    params = LorentzianParams(
        amp=params_array[0],
        sigma=params_array[1],
        x0=params_array[2],
        y0=params_array[3],
    )

    return fitted_y, params


@dataclasses.dataclass
class ExponentialParams:
    """Exponential fit parameters: y = a * exp(b * x) + y0"""

    a: float  # amplitude
    b: float  # exponential coefficient
    y0: float  # baseline offset


def exponential_fit(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, ExponentialParams]:
    """Compute exponential fit: y = a * exp(b * x) + y0.

    Args:
        x: x data array
        y: y data array

    Returns:
        tuple: (fitted_y_values, ExponentialParams)
    """
    # Parameter estimation
    y_range = np.max(y) - np.min(y)
    y_min = np.min(y)

    # Estimate from data
    if len(y) > 1:
        # Try to determine if it's growth or decay
        if y[0] > y[-1]:
            # Decay
            a_guess = y_range
            b_guess = -1.0 / (np.max(x) - np.min(x))
        else:
            # Growth
            a_guess = y_range * 0.1
            b_guess = 1.0 / (np.max(x) - np.min(x))
    else:
        a_guess = y_range
        b_guess = -1.0

    y0_guess = y_min

    def exp_func(x, a, b, y0):
        # Clip b to prevent overflow
        b_clipped = np.clip(b, -50, 50)
        return a * np.exp(b_clipped * x) + y0

    initial_params = [a_guess, b_guess, y0_guess]

    # Parameter bounds
    bounds = [
        (-y_range * 1000, y_range * 1000),  # a
        (-10, 10),  # b (reasonable range to prevent overflow)
        (y_min - 0.5 * y_range, y_min + 0.5 * y_range),  # y0
    ]

    fitted_y, params_array = _fit_with_scipy(x, y, exp_func, initial_params, bounds)

    params = ExponentialParams(
        a=params_array[0],
        b=params_array[1],
        y0=params_array[2],
    )

    return fitted_y, params


@dataclasses.dataclass
class PlanckianParams:
    """Planckian (blackbody radiation) fit parameters"""

    amp: float  # amplitude
    x0: float  # peak position
    sigma: float  # width parameter
    y0: float  # baseline offset


def planckian_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, PlanckianParams]:
    """
    Compute Planckian (blackbody radiation) fit.

    Args:
        x: wavelength data array
        y: intensity data array

    Returns:
        tuple: (fitted_y_values, PlanckianParams)
    """
    # Parameter estimation
    dy = np.max(y) - np.min(y)
    x_peak = x[np.argmax(y)]
    y_min = np.min(y)

    amp_guess = dy
    x0_guess = x_peak
    sigma_guess = 1.0  # Width parameter
    y0_guess = y_min

    initial_params = [amp_guess, x0_guess, sigma_guess, y0_guess]

    # Parameter bounds
    bounds = [
        (dy * 0.01, dy * 100),  # amp
        (np.min(x), np.max(x)),  # x0
        (0.1, 5.0),  # sigma
        (y0_guess - 0.2 * dy, y0_guess + 0.2 * dy),  # y0
    ]

    fitted_y, params_array = _fit_with_scipy(
        x, y, fitmodels.PlanckianModel.func, initial_params, bounds
    )

    params = PlanckianParams(
        amp=params_array[0],
        x0=params_array[1],
        sigma=params_array[2],
        y0=params_array[3],
    )

    return fitted_y, params


@dataclasses.dataclass
class TwoHalfGaussianParams:
    """Two half-Gaussian fit parameters."""

    amp_left: float  # left amplitude
    amp_right: float  # right amplitude
    sigma_left: float  # left sigma
    sigma_right: float  # right sigma
    x0: float  # center position
    y0_left: float  # left baseline offset
    y0_right: float  # right baseline offset


def twohalfgaussian_fit(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, TwoHalfGaussianParams]:
    """Compute two half-Gaussian fit for asymmetric peaks with separate baselines.

    Now supports separate amplitudes for even better asymmetric peak fitting.

    Args:
        x: x data array
        y: y data array

    Returns:
        tuple: (fitted_y_values, TwoHalfGaussianParams)
    """
    # Parameter estimation with separate baseline analysis
    dx = np.max(x) - np.min(x)
    dy = np.max(y) - np.min(y)
    x_peak = x[np.argmax(y)]

    # Estimate separate baselines for left and right sides
    left_mask = x < x_peak
    right_mask = x >= x_peak

    # Use the lower quartile of each side for robust baseline estimation
    if np.any(left_mask):
        y0_left_guess = np.percentile(y[left_mask], 25)
    else:
        y0_left_guess = np.min(y)

    if np.any(right_mask):
        y0_right_guess = np.percentile(y[right_mask], 25)
    else:
        y0_right_guess = np.min(y)

    # Peak amplitude estimation (above average baseline)
    avg_baseline = (y0_left_guess + y0_right_guess) / 2
    amp_guess = np.max(y) - avg_baseline
    half_max = avg_baseline + amp_guess * 0.5

    # Find points at half maximum
    left_points = np.where((x < x_peak) & (y >= half_max))[0]
    right_points = np.where((x > x_peak) & (y >= half_max))[0]

    # Estimate sigma values from half-width measurements
    if len(left_points) > 0:
        left_hw = x_peak - x[left_points[0]]
        sigma_left_guess = left_hw / np.sqrt(2 * np.log(2))
    else:
        sigma_left_guess = dx * 0.05

    if len(right_points) > 0:
        right_hw = x[right_points[-1]] - x_peak
        sigma_right_guess = right_hw / np.sqrt(2 * np.log(2))
    else:
        sigma_right_guess = dx * 0.05

    x0_guess = x_peak

    if np.any(left_mask):
        left_peak_val = np.max(y[left_mask])
        amp_left_guess = left_peak_val - y0_left_guess
    else:
        amp_left_guess = dy * 0.5

    if np.any(right_mask):
        right_peak_val = np.max(y[right_mask])
        amp_right_guess = right_peak_val - y0_right_guess
    else:
        amp_right_guess = dy * 0.5

    initial_params = [
        amp_left_guess,
        amp_right_guess,
        sigma_left_guess,
        sigma_right_guess,
        x0_guess,
        y0_left_guess,
        y0_right_guess,
    ]
    bounds = [
        (dy * 0.1, dy * 3),  # amp_left
        (dy * 0.1, dy * 3),  # amp_right
        (dx * 0.001, dx * 0.5),  # sigma_left
        (dx * 0.001, dx * 0.5),  # sigma_right
        (np.min(x), np.max(x)),  # x0
        (y0_left_guess - 0.3 * dy, y0_left_guess + 0.3 * dy),  # y0_left
        (y0_right_guess - 0.3 * dy, y0_right_guess + 0.3 * dy),  # y0_right
    ]
    fitted_y, params_array = _fit_with_scipy(
        x, y, fitmodels.TwoHalfGaussianModel.func, initial_params, bounds
    )
    params = TwoHalfGaussianParams(
        amp_left=params_array[0],
        amp_right=params_array[1],
        sigma_left=params_array[2],
        sigma_right=params_array[3],
        x0=params_array[4],
        y0_left=params_array[5],
        y0_right=params_array[6],
    )
    return fitted_y, params


@dataclasses.dataclass
class DoubleExponentialParams:
    """Double exponential fit parameters"""

    x_center: float  # center position (boundary between left and right components)
    a_left: float  # left component amplitude coefficient
    b_left: float  # left component time constant coefficient
    a_right: float  # right component amplitude coefficient
    b_right: float  # right component time constant coefficient
    y0: float  # baseline offset


def doubleexponential_fit(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, DoubleExponentialParams]:
    """Compute double exponential fit.

    Args:
        x: time data array
        y: intensity data array

    Returns:
        tuple: (fitted_y_values, DoubleExponentialParams)
    """
    y_range = np.max(y) - np.min(y)
    x_range = np.max(x) - np.min(x)
    y_max = np.max(y)

    # Baseline is rarely different from zero:
    y0_guess = 0.0

    # Analyze signal characteristics for better initial guesses
    peak_idx = np.argmax(y)

    # Estimate x_center as the peak position
    x_center_guess = x[peak_idx]

    # Estimate parameters (a_left, b_left, a_right, b_right) by decomposing
    # the signal into growth and decay components based on peak position, and
    # fitting each curve with exponential functions using exponential_fit().
    # X center estimation is very rough here, so we need to remove say 10% of
    # the x range on each side to avoid fitting artifacts.
    x_range = np.max(x) - np.min(x)
    x_left_mask = x < (x_center_guess - 0.1 * x_range)
    x_right_mask = x >= (x_center_guess + 0.1 * x_range)

    x_left, y_left = x[x_left_mask], y[x_left_mask]
    x_right, y_right = x[x_right_mask], y[x_right_mask]

    if np.any(x_left_mask):
        y_left_fitted, left_params = exponential_fit(x_left, y_left)
    else:
        left_params = ExponentialParams(a=0.0, b=0.1, y0=0.0)
    if np.any(x_right_mask):
        y_right_fitted, right_params = exponential_fit(x_right, y_right)
    else:
        right_params = ExponentialParams(a=0.0, b=0.1, y0=0.0)

    a_left_guess = left_params.a
    b_left_guess = left_params.b
    a_right_guess = right_params.a
    b_right_guess = right_params.b
    y0_guess = (left_params.y0 + right_params.y0) / 2

    # Set bounds for parameters - b can be positive or negative
    amp_bound = max(abs(y_max - y0_guess), y_range) * 2
    rate_bound = 5.0 / max(x_range, 1e-6)  # Avoid division by zero

    # Ensure initial parameters are within bounds
    b_left_guess = np.clip(b_left_guess, -rate_bound, rate_bound)
    b_right_guess = np.clip(b_right_guess, -rate_bound, rate_bound)
    a_left_guess = np.clip(a_left_guess, -amp_bound, amp_bound)
    a_right_guess = np.clip(a_right_guess, -amp_bound, amp_bound)

    initial_params = [
        x_center_guess,
        a_left_guess,
        b_left_guess,
        a_right_guess,
        b_right_guess,
        y0_guess,
    ]
    fitted_y, params_array = _fit_with_scipy(
        x, y, fitmodels.DoubleExponentialModel.func, initial_params, None
    )
    params = DoubleExponentialParams(
        x_center=params_array[0],
        a_left=params_array[1],
        b_left=params_array[2],
        a_right=params_array[3],
        b_right=params_array[4],
        y0=params_array[5],
    )
    return fitted_y, params


@dataclasses.dataclass
class MultiLorentzianParams:
    """Multi-Lorentzian fit parameters"""

    peaks: list[dict]  # List of peak parameters (amp, sigma, x0)
    y0: float  # baseline offset


def multilorentzian_fit(
    x: np.ndarray, y: np.ndarray, peak_indices: list[int]
) -> tuple[np.ndarray, MultiLorentzianParams]:
    """Compute multi-Lorentzian fit for multiple peaks.

    Args:
        x: x data array
        y: y data array
        peak_indices: list of peak indices

    Returns:
        tuple: (fitted_y_values, MultiLorentzianParams)
    """
    if not peak_indices:
        raise ValueError("At least one peak index must be provided")

    # Parameter estimation for each peak
    dy = np.max(y) - np.min(y)
    y_min = np.min(y)

    initial_params = []
    bounds = []

    for i, peak_idx in enumerate(peak_indices):
        # Estimate parameters for each Lorentzian
        if i > 0:
            istart = (peak_indices[i - 1] + peak_idx) // 2
        else:
            istart = 0

        if i < len(peak_indices) - 1:
            iend = (peak_indices[i + 1] + peak_idx) // 2
        else:
            iend = len(x) - 1

        local_dx = 0.5 * (x[iend] - x[istart])
        local_dy = np.max(y[istart:iend]) - np.min(y[istart:iend])

        # Lorentzian parameters: amp, sigma, x0
        amp_guess = fitmodels.LorentzianModel.get_amp_from_amplitude(
            local_dy, local_dx * 0.1
        )
        sigma_guess = local_dx * 0.1
        x0_guess = x[peak_idx]

        initial_params.extend([amp_guess, sigma_guess, x0_guess])

        # Bounds for this peak
        bounds.extend(
            [
                (0.0, local_dy * 2),  # amp
                (local_dx * 0.01, local_dx),  # sigma
                (x[istart], x[iend]),  # x0
            ]
        )

    # Add baseline parameter
    initial_params.append(y_min)
    bounds.append((y_min - 0.1 * dy, y_min + 0.1 * dy))

    def multi_lorentzian_func(x_vals, *params):
        """Multi-Lorentzian function"""
        n_peaks = len(peak_indices)
        y_result = np.zeros_like(x_vals) + params[-1]  # baseline

        for i in range(n_peaks):
            amp = params[i * 3]
            sigma = params[i * 3 + 1]
            x0 = params[i * 3 + 2]
            y_result += fitmodels.LorentzianModel.func(x_vals, amp, sigma, x0, 0)

        return y_result

    fitted_y, params_array = _fit_with_scipy(
        x, y, multi_lorentzian_func, initial_params, bounds
    )

    # Parse parameters into dataclass
    n_peaks = len(peak_indices)
    peak_params = []
    for i in range(n_peaks):
        peak_params.append(
            {
                "amp": params_array[i * 3],
                "sigma": params_array[i * 3 + 1],
                "x0": params_array[i * 3 + 2],
            }
        )

    params = MultiLorentzianParams(
        peaks=peak_params,
        y0=params_array[-1],
    )

    return fitted_y, params


@dataclasses.dataclass
class SinusoidalParams:
    """Sinusoidal fit parameters."""

    amplitude: float  # amplitude
    frequency: float  # frequency
    phase: float  # phase
    offset: float  # baseline offset


def sinusoidal_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, SinusoidalParams]:
    """Compute sinusoidal fit.

    Args:
        x: x data array
        y: y data array

    Returns:
        tuple: (fitted_y_values, SinusoidalParams)
    """
    # Parameter estimation using FFT for frequency
    dy = np.max(y) - np.min(y)

    amplitude_guess = dy / 2
    offset_guess = np.mean(y)
    phase_guess = 0.0

    # Estimate frequency using FFT
    if len(x) > 2:
        dt = x[1] - x[0]  # Assuming evenly spaced
        fft_y = np.fft.fft(y - offset_guess)
        freqs = np.fft.fftfreq(len(y), dt)
        # Find dominant frequency (excluding DC component)
        dominant_idx = np.argmax(np.abs(fft_y[1 : len(fft_y) // 2])) + 1
        frequency_guess = np.abs(freqs[dominant_idx])
    else:
        frequency_guess = 1.0 / (np.max(x) - np.min(x))

    def sin_func(x_vals, amplitude, frequency, phase, offset):
        return amplitude * np.sin(2 * np.pi * frequency * x_vals + phase) + offset

    initial_params = [amplitude_guess, frequency_guess, phase_guess, offset_guess]

    # Parameter bounds
    bounds = [
        (0, dy),  # amplitude
        (0, 2 * frequency_guess),  # frequency
        (-2 * np.pi, 2 * np.pi),  # phase
        (offset_guess - dy, offset_guess + dy),  # offset
    ]

    fitted_y, params_array = _fit_with_scipy(x, y, sin_func, initial_params, bounds)

    params = SinusoidalParams(
        amplitude=params_array[0],
        frequency=params_array[1],
        phase=params_array[2],
        offset=params_array[3],
    )

    return fitted_y, params


@dataclasses.dataclass
class VoigtParams:
    """Voigt fit parameters."""

    amplitude: float  # amplitude
    sigma: float  # Gaussian width parameter
    x0: float  # center position
    y0: float  # baseline offset


def voigt_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, VoigtParams]:
    """Compute Voigt fit.

    Args:
        x: x data array
        y: y data array

    Returns:
        tuple: (fitted_y_values, VoigtParams)
    """
    # Parameter estimation
    y_min, y_max = np.min(y), np.max(y)
    dy = y_max - y_min
    x_min, x_max = np.min(x), np.max(x)
    dx = x_max - x_min

    # Initial estimates
    x0_guess = x[np.argmax(y)]  # Center at peak
    y0_guess = y_min  # Baseline
    amplitude_guess = dy  # Amplitude
    sigma_guess = dx / 10  # Width parameter

    def voigt_func(x_vals, amplitude, sigma, x0, y0):
        return fitmodels.VoigtModel.func(x_vals, amplitude, sigma, x0, y0)

    initial_params = [amplitude_guess, sigma_guess, x0_guess, y0_guess]

    # Parameter bounds
    bounds = [
        (0, 10 * dy),  # amplitude
        (dx / 1000, dx),  # sigma
        (x_min, x_max),  # x0
        (y_min - dy, y_max + dy),  # y0
    ]

    fitted_y, params_array = _fit_with_scipy(x, y, voigt_func, initial_params, bounds)

    params = VoigtParams(
        amplitude=params_array[0],
        sigma=params_array[1],
        x0=params_array[2],
        y0=params_array[3],
    )

    return fitted_y, params


@dataclasses.dataclass
class CdfParams:
    """CDF (Cumulative Distribution Function) fit parameters."""

    amplitude: float  # amplitude
    mu: float  # mean
    sigma: float  # standard deviation
    baseline: float  # baseline offset


def cdf_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, CdfParams]:
    """Compute Cumulative Distribution Function (CDF) fit.

    Args:
        x: x data array
        y: y data array

    Returns:
        tuple: (fitted_y_values, CdfParams)
    """
    # Parameter estimation
    y_min, y_max = np.min(y), np.max(y)
    dy = y_max - y_min
    x_min, x_max = np.min(x), np.max(x)
    dx = x_max - x_min

    # Initial estimates
    amplitude_guess = dy
    baseline_guess = dy / 2
    sigma_guess = dx / 10
    mu_guess = (x_max + np.abs(x_min)) / 2

    def cdf_func(x_vals, amplitude, mu, sigma, baseline):
        return (
            amplitude * scipy.special.erf((x_vals - mu) / (sigma * np.sqrt(2)))
            + baseline
        )

    initial_params = [amplitude_guess, mu_guess, sigma_guess, baseline_guess]

    # Parameter bounds
    bounds = [
        (0, 2 * dy),  # amplitude
        (x_min, x_max),  # mu
        (dx / 1000, dx),  # sigma
        (y_min - dy, y_max + dy),  # baseline
    ]

    fitted_y, params_array = _fit_with_scipy(x, y, cdf_func, initial_params, bounds)

    params = CdfParams(
        amplitude=params_array[0],
        mu=params_array[1],
        sigma=params_array[2],
        baseline=params_array[3],
    )

    return fitted_y, params


@dataclasses.dataclass
class SigmoidParams:
    """Sigmoid (Logistic) fit parameters."""

    amplitude: float  # amplitude
    k: float  # growth rate
    x0: float  # horizontal offset (inflection point)
    offset: float  # vertical offset


def sigmoid_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, SigmoidParams]:
    """Compute Sigmoid (Logistic) fit.

    Args:
        x: x data array
        y: y data array

    Returns:
        tuple: (fitted_y_values, SigmoidParams)
    """
    # Parameter estimation
    y_min, y_max = np.min(y), np.max(y)
    dy = y_max - y_min
    x_min, x_max = np.min(x), np.max(x)
    dx = x_max - x_min

    # Initial estimates
    amplitude_guess = dy
    offset_guess = y_min
    x0_guess = (x_min + x_max) / 2  # Inflection point at center
    k_guess = 4 / dx  # Growth rate (4/dx gives reasonable sigmoid shape)

    def sigmoid_func(x_vals, amplitude, k, x0, offset):
        return offset + amplitude / (1.0 + np.exp(-k * (x_vals - x0)))

    initial_params = [amplitude_guess, k_guess, x0_guess, offset_guess]

    # Parameter bounds
    bounds = [
        (0, 10 * dy),  # amplitude
        (1 / dx, 100 / dx),  # k (growth rate)
        (x_min, x_max),  # x0
        (y_min - dy, y_max + dy),  # offset
    ]

    fitted_y, params_array = _fit_with_scipy(x, y, sigmoid_func, initial_params, bounds)

    params = SigmoidParams(
        amplitude=params_array[0],
        k=params_array[1],
        x0=params_array[2],
        offset=params_array[3],
    )

    return fitted_y, params
