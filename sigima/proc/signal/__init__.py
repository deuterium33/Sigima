# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Basic signal processing
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sigima.proc.signal.base
   :members:

Arithmetic
~~~~~~~~~~

.. automodule:: sigima.proc.signal.arithmetic
    :members:

Mathematical Operations
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sigima.proc.signal.mathops
    :members:

Extraction
~~~~~~~~~~

.. automodule:: sigima.proc.signal.extraction
    :members:

Filtering
~~~~~~~~~

.. automodule:: sigima.proc.signal.filtering
    :members:

Processing
~~~~~~~~~~

.. automodule:: sigima.proc.signal.processing
    :members:

Fourier
~~~~~~~

.. automodule:: sigima.proc.signal.fourier
    :members:

Fitting
~~~~~~~

.. automodule:: sigima.proc.signal.fitting
    :members:

Features
~~~~~~~~

.. automodule:: sigima.proc.signal.features
    :members:

Stability
~~~~~~~~~

.. automodule:: sigima.proc.signal.stability
    :members:

Analysis
~~~~~~~~

.. automodule:: sigima.proc.signal.analysis
    :members:
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# MARK: Important notes
# ---------------------
# - All `guidata.dataset.DataSet` classes must also be imported
#   in the `sigima.params` module.
# - All functions decorated by `computation_function` defined in the other modules
#   of this package must be imported right here.

from __future__ import annotations

# Import parameter classes from the main base module
from sigima.proc.signal.analysis import (
    PulseFeaturesParam,
    contrast,
    extract_pulse_features,
    histogram,
    sampling_rate_period,
    x_at_minmax,
)
from sigima.proc.signal.arithmetic import (
    addition,
    addition_constant,
    arithmetic,
    average,
    difference,
    difference_constant,
    division,
    division_constant,
    product,
    product_constant,
    quadratic_difference,
    signals_to_array,
    standard_deviation,
)
from sigima.proc.signal.base import (
    Wrap1to1Func,
    compute_geometry_from_obj,
    is_uncertainty_data_available,
    restore_data_outside_roi,
    signals_dy_to_array,
    signals_y_to_array,
)
from sigima.proc.signal.extraction import (
    extract_roi,
    extract_rois,
)
from sigima.proc.signal.features import (
    AbscissaParam,
    DynamicParam,
    FWHMParam,
    OrdinateParam,
    PeakDetectionParam,
    bandwidth_3db,
    dynamic_parameters,
    full_width_at_y,
    fw1e2,
    fwhm,
    peak_detection,
    stats,
    x_at_y,
    y_at_x,
)
from sigima.proc.signal.filtering import (
    BandPassFilterParam,
    BandStopFilterParam,
    BaseHighLowBandParam,
    FrequencyFilterMethod,
    HighPassFilterParam,
    LowPassFilterParam,
    PadLocation,
    add_gaussian_noise,
    add_poisson_noise,
    add_uniform_noise,
    bandpass,
    bandstop,
    frequency_filter,
    gaussian_filter,
    highpass,
    lowpass,
    moving_average,
    moving_median,
    wiener,
)
from sigima.proc.signal.fitting import (
    PolynomialFitParam,
    cdf_fit,
    doubleexponential_fit,
    evaluate_fit,
    exponential_fit,
    extract_fit_params,
    gaussian_fit,
    linear_fit,
    lorentzian_fit,
    planckian_fit,
    polynomial_fit,
    sigmoid_fit,
    sinusoidal_fit,
    twohalfgaussian_fit,
    voigt_fit,
)
from sigima.proc.signal.fourier import (
    ZeroPadding1DParam,
    fft,
    ifft,
    magnitude_spectrum,
    phase_spectrum,
    psd,
    zero_padding,
)
from sigima.proc.signal.mathops import (
    DataTypeSParam,
    PowerParam,
    absolute,
    astype,
    complex_from_magnitude_phase,
    complex_from_real_imag,
    exp,
    imag,
    inverse,
    log10,
    phase,
    power,
    real,
    sqrt,
    to_cartesian,
    to_polar,
    transpose,
)
from sigima.proc.signal.processing import (
    DetrendingParam,
    InterpolationParam,
    Resampling1DParam,
    WindowingParam,
    XYCalibrateParam,
    apply_window,
    calibration,
    check_same_sample_rate,
    clip,
    convolution,
    deconvolution,
    derivative,
    detrending,
    get_nyquist_frequency,
    integral,
    interpolate,
    normalize,
    offset_correction,
    resampling,
    reverse_x,
    xy_mode,
)
from sigima.proc.signal.stability import (
    AllanVarianceParam,
    allan_deviation,
    allan_variance,
    hadamard_variance,
    modified_allan_variance,
    overlapping_allan_variance,
    time_deviation,
    total_variance,
)

__all__ = [
    "Wrap1to1Func",
    "compute_geometry_from_obj",
    "is_uncertainty_data_available",
    "restore_data_outside_roi",
    "signals_dy_to_array",
    "signals_to_array",
    "signals_y_to_array",
    # Parameter classes
    "PulseFeaturesParam",
    "DetrendingParam",
    "WindowingParam",
    "XYCalibrateParam",
    # Functions
    "addition",
    "addition_constant",
    "arithmetic",
    "average",
    "difference",
    "difference_constant",
    "division",
    "division_constant",
    "product",
    "product_constant",
    "quadratic_difference",
    "standard_deviation",
    "extract_roi",
    "extract_rois",
    "BaseHighLowBandParam",
    "BandPassFilterParam",
    "BandStopFilterParam",
    "FrequencyFilterMethod",
    "HighPassFilterParam",
    "LowPassFilterParam",
    "PadLocation",
    "add_gaussian_noise",
    "add_poisson_noise",
    "add_uniform_noise",
    "bandpass",
    "bandstop",
    "frequency_filter",
    "gaussian_filter",
    "highpass",
    "lowpass",
    "moving_average",
    "moving_median",
    "wiener",
    "fft",
    "ifft",
    "magnitude_spectrum",
    "phase_spectrum",
    "psd",
    "DataTypeSParam",
    "PowerParam",
    "absolute",
    "astype",
    "complex_from_magnitude_phase",
    "complex_from_real_imag",
    "exp",
    "imag",
    "inverse",
    "log10",
    "phase",
    "power",
    "real",
    "sqrt",
    "to_cartesian",
    "to_polar",
    "transpose",
    "InterpolationParam",
    "Resampling1DParam",
    "ZeroPadding1DParam",
    "apply_window",
    "calibration",
    "check_same_sample_rate",
    "clip",
    "convolution",
    "deconvolution",
    "derivative",
    "detrending",
    "get_nyquist_frequency",
    "integral",
    "interpolate",
    "normalize",
    "offset_correction",
    "resampling",
    "reverse_x",
    "xy_mode",
    "zero_padding",
    "PolynomialFitParam",
    "cdf_fit",
    "doubleexponential_fit",
    "evaluate_fit",
    "exponential_fit",
    "extract_fit_params",
    "gaussian_fit",
    "linear_fit",
    "lorentzian_fit",
    "planckian_fit",
    "polynomial_fit",
    "sigmoid_fit",
    "sinusoidal_fit",
    "twohalfgaussian_fit",
    "voigt_fit",
    "AbscissaParam",
    "DynamicParam",
    "FWHMParam",
    "OrdinateParam",
    "PeakDetectionParam",
    "bandwidth_3db",
    "dynamic_parameters",
    "fw1e2",
    "fwhm",
    "full_width_at_y",
    "peak_detection",
    "stats",
    "x_at_y",
    "y_at_x",
    "AllanVarianceParam",
    "allan_deviation",
    "allan_variance",
    "hadamard_variance",
    "modified_allan_variance",
    "overlapping_allan_variance",
    "time_deviation",
    "total_variance",
    "contrast",
    "extract_pulse_features",
    "histogram",
    "sampling_rate_period",
    "x_at_minmax",
]
