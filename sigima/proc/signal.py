# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Signal computation objects (see parent package :mod:`sigima.proc`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# MARK: Important note
# --------------------
# All `guidata.dataset.DataSet` classes must also be imported
# in the `sigima.param` module.

from __future__ import annotations

import warnings
from collections.abc import Callable
from enum import Enum
from math import ceil, log2
from typing import Any

import guidata.dataset as gds
import numpy as np
import scipy.integrate as spt
import scipy.ndimage as spi
import scipy.signal as sps

from sigima.config import _
from sigima.objects import (
    NO_ROI,
    GeometryResult,
    KindShape,
    NormalDistribution1DParam,
    PoissonDistributionParam,
    ROI1DParam,
    SignalObj,
    TableResult,
    TableResultBuilder,
    UniformDistribution1DParam,
    create_signal_from_param,
)
from sigima.proc.base import (
    ArithmeticParam,
    ClipParam,
    ConstantParam,
    FFTParam,
    GaussianParam,
    HistogramParam,
    MovingAverageParam,
    MovingMedianParam,
    NormalizeParam,
    SpectrumParam,
    dst_1_to_1,
    dst_2_to_1,
    dst_n_to_1,
    new_signal_result,
)
from sigima.proc.decorator import computation_function
from sigima.proc.enums import MathOperator, PadLocation, PowerUnit
from sigima.tools import coordinates
from sigima.tools.signal import (
    dynamic,
    features,
    fourier,
    interpolation,
    peakdetection,
    pulse,
    scaling,
    stability,
    windowing,
)

__all__ = [
    "Wrap1to1Func",
    "restore_data_outside_roi",
    "addition",
    "average",
    "product",
    "addition_constant",
    "difference_constant",
    "product_constant",
    "division_constant",
    "arithmetic",
    "difference",
    "quadratic_difference",
    "division",
    "extract_rois",
    "extract_roi",
    "transpose",
    "inverse",
    "absolute",
    "real",
    "imag",
    "DataTypeSParam",
    "astype",
    "log10",
    "exp",
    "sqrt",
    "PowerParam",
    "power",
    "PeakDetectionParam",
    "peak_detection",
    "normalize",
    "derivative",
    "integral",
    "XYCalibrateParam",
    "calibration",
    "clip",
    "offset_correction",
    "gaussian_filter",
    "moving_average",
    "moving_median",
    "wiener",
    "LowPassFilterParam",
    "HighPassFilterParam",
    "BandPassFilterParam",
    "BandStopFilterParam",
    "lowpass",
    "highpass",
    "bandpass",
    "bandstop",
    "ZeroPadding1DParam",
    "zero_padding",
    "fft",
    "ifft",
    "magnitude_spectrum",
    "phase_spectrum",
    "psd",
    "PolynomialFitParam",
    "histogram",
    "InterpolationParam",
    "interpolate",
    "ResamplingParam",
    "resampling",
    "DetrendingParam",
    "detrending",
    "xy_mode",
    "convolution",
    "WindowingParam",
    "apply_window",
    "reverse_x",
    "AngleUnitParam",
    "to_polar",
    "to_cartesian",
    "AllanVarianceParam",
    "allan_variance",
    "allan_deviation",
    "overlapping_allan_variance",
    "modified_allan_variance",
    "hadamard_variance",
    "total_variance",
    "time_deviation",
    "compute_geometry_from_obj",
    "FWHMParam",
    "fwhm",
    "fw1e2",
    "OrdinateParam",
    "full_width_at_y",
    "x_at_y",
    "AbscissaParam",
    "y_at_x",
    "stats",
    "bandwidth_3db",
    "DynamicParam",
    "dynamic_parameters",
    "sampling_rate_period",
    "contrast",
    "x_at_minmax",
    "add_gaussian_noise",
    "add_poisson_noise",
    "add_uniform_noise",
]


def restore_data_outside_roi(dst: SignalObj, src: SignalObj) -> None:
    """Restore data outside the Region Of Interest (ROI) of the input signal
    after a computation, only if the input signal has a ROI,
    and if the output signal has the same ROI as the input signal,
    and if the data types are the same,
    and if the shapes are the same.
    Otherwise, do nothing.

    Args:
        dst: destination signal object
        src: source signal object
    """
    if src.maskdata is not None and dst.maskdata is not None:
        if (
            np.array_equal(src.maskdata, dst.maskdata)
            and dst.xydata.dtype == src.xydata.dtype
            and dst.xydata.shape == src.xydata.shape
        ):
            dst.xydata[src.maskdata] = src.xydata[src.maskdata]


def is_uncertainty_data_available(signals: SignalObj | list[SignalObj]) -> bool:
    """Check if all signals have uncertainty data.

    This functions is used to determine whether enough information is available to
    propagate uncertainty.

    Args:
        signals: Signal object or list of signal objects.

    Returns:
        True if all signals have uncertainty data, False otherwise.
    """
    if isinstance(signals, SignalObj):
        signals = [signals]
    return all(sig.dy is not None for sig in signals)


class Wrap1to1Func:
    """Wrap a 1 array → 1 array function (the simple case of y1 = f(y0)) to produce
    a 1 signal → 1 signal function, which can be used as a Sigima computation function
    and inside DataLab's infrastructure to perform computations with the Signal
    Processor object.

    This wrapping mechanism using a class is necessary for the resulted function to be
    pickable by the ``multiprocessing`` module.

    The instance of this wrapper is callable and returns
    a :class:`sigima.objects.SignalObj` object.

    Example:

        >>> import numpy as np
        >>> from sigima.proc.signal import Wrap1to1Func
        >>> import sigima.objects
        >>> def square(y):
        ...     return y**2
        >>> compute_square = Wrap1to1Func(square)
        >>> x = np.linspace(0, 10, 100)
        >>> y = np.sin(x)
        >>> sig0 = sigima.objects.create_signal("Example", x, y)
        >>> sig1 = compute_square(sig0)

    Args:
        func: 1 array → 1 array function
        *args: Additional positional arguments to pass to the function
        **kwargs: Additional keyword arguments to pass to the function

    .. note::

        If `func_name` is provided in the keyword arguments, it will be used as the
        function name instead of the default name derived from the function itself.
    """

    def __init__(self, func: Callable, *args: Any, **kwargs: Any) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.__name__ = self.kwargs.pop("func_name", func.__name__)
        self.__doc__ = func.__doc__
        self.__call__.__func__.__doc__ = self.func.__doc__

    def __call__(self, src: SignalObj) -> SignalObj:
        """Compute the function on the input signal and return the result signal

        Args:
            src: input signal object

        Returns:
            Result signal object
        """
        suffix = ", ".join(
            [str(arg) for arg in self.args]
            + [f"{k}={v}" for k, v in self.kwargs.items() if v is not None]
        )
        dst = dst_1_to_1(src, self.__name__, suffix)
        x, y = src.get_data()
        dst.set_xydata(x, self.func(y, *self.args, **self.kwargs))

        # Uncertainty propagation for common mathematical functions
        if is_uncertainty_data_available(src):
            if self.func == np.sqrt:
                # σ(√y) = σ(y) / (2√y)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    dst.dy = src.dy / (2 * np.sqrt(src.y))
                    dst.dy[np.isinf(dst.dy) | np.isnan(dst.dy)] = np.nan
            elif self.func == np.log10:
                # σ(log₁₀(y)) = σ(y) / (y * ln(10))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    dst.dy = src.dy / (src.y * np.log(10))
                    dst.dy[np.isinf(dst.dy) | np.isnan(dst.dy)] = np.nan
            elif self.func == np.exp:
                # σ(eʸ) = eʸ * σ(y) = dst.y * σ(y)
                dst.dy = np.abs(dst.y) * src.dy
            elif self.func == np.clip:
                # σ(clip(y)) = σ(y) where not clipped, 0 where clipped
                a_min = self.kwargs.get("a_min", None)
                a_max = self.kwargs.get("a_max", None)
                if a_min is not None:
                    dst.dy[src.y <= a_min] = 0
                if a_max is not None:
                    dst.dy[src.y >= a_max] = 0
            # For absolute, real, imag: uncertainties unchanged (copied by dst_1_to_1)
        restore_data_outside_roi(dst, src)
        return dst


# MARK: n_to_1 functions -------------------------------------------------------
# Functions with N input signals and 1 output signal
# --------------------------------------------------------------------------------------
# Those functions are perfoming a computation on N input signals and return a single
# output signal. If we were only executing these functions locally, we would not need
# to define them here, but since we are using the multiprocessing module, we need to
# define them here so that they can be pickled and sent to the worker processes.
# Also, we need to systematically return the output signal object, even if it is already
# modified in place, because the multiprocessing module will not be able to retrieve
# the modified object from the worker processes.


def signals_to_array(
    signals: list[SignalObj], attr: str = "y", dtype: np.dtype | None = None
) -> np.ndarray:
    """Create an array from a list of signals.

    Args:
        signals: List of signal objects.
        attr: Name of the attribute to extract ("y", "dy", etc.). Defaults to "y".
        dtype: Desired type for the output array. If None, use the first signal's dtype.

    Returns:
        A NumPy array stacking the specified attribute from all signals.

    Raises:
        ValueError: If the signals list is empty.
    """
    if not signals:
        raise ValueError("The signal list is empty.")
    if dtype is None:
        dtype = getattr(signals[0], attr).dtype
    arr = np.array([getattr(sig, attr) for sig in signals], dtype=dtype)
    return arr


def signals_y_to_array(
    signals: SignalObj | list[SignalObj], dtype: np.dtype | None = None
) -> np.ndarray:
    """Create an array from a list of signals, extracting the `y` attribute.

    Args:
        signals: List of signal objects.
        dtype: Desired type for the output array. If None, use the first signal's dtype.

    Returns:
        A NumPy array stacking the `y` attribute from all signals.
    """
    if isinstance(signals, SignalObj):
        signals = [signals]
    return signals_to_array(signals, attr="y", dtype=dtype)


def signals_dy_to_array(
    signals: SignalObj | list[SignalObj], dtype: np.dtype | None = None
) -> np.ndarray:
    """Create an array from a list of signals, extracting the `dy` attribute.

    Args:
        signals: List of signal objects.
        dtype: Desired type for the output array. If None, use the first signal's dtype.

    Returns:
        A NumPy array stacking the `dy` attribute from all signals.
    """
    if isinstance(signals, SignalObj):
        signals = [signals]
    return signals_to_array(signals, attr="dy", dtype=dtype)


@computation_function()
def addition(src_list: list[SignalObj]) -> SignalObj:
    """Compute the element-wise sum of multiple signals.

    The first signal in the list defines the "base" signal. All other signals are added
    element-wise to the base signal.

    .. note::

        If all signals share the same region of interest (ROI), the sum is performed
        only within the ROI.

    .. note::

        Uncertainties are propagated.

    .. warning::

        It is assumed that all signals have the same size and x-coordinates.

    Args:
        src_list: List of source signals.

    Returns:
        Signal object representing the sum of the source signals.
    """
    dst = dst_n_to_1(src_list, "Σ")  # `dst` data is initialized to `src_list[0]` data.
    dst.y = np.sum(signals_y_to_array(src_list), axis=0)
    if is_uncertainty_data_available(src_list):
        dst.dy = np.sqrt(np.sum(signals_dy_to_array(src_list) ** 2, axis=0))
    restore_data_outside_roi(dst, src_list[0])
    return dst


@computation_function()
def average(src_list: list[SignalObj]) -> SignalObj:
    """Compute the element-wise average of multiple signals.

    The first signal in the list defines the "base" signal. All other signals are
    averaged element-wise with the base signal.

    .. note::

        If all signals share the same region of interest (ROI), the average is performed
        only within the ROI.

    .. note::

        Uncertainties are propagated.

    .. warning::

        It is assumed that all signals have the same size and x-coordinates.

    Args:
        src_list: List of source signals.

    Returns:
        Signal object representing the average of the source signals.
    """
    dst = dst_n_to_1(src_list, "µ")  # `dst` data is initialized to `src_list[0]` data.
    dst.y = np.mean(signals_y_to_array(src_list), axis=0)
    if is_uncertainty_data_available(src_list):
        dy_array = signals_dy_to_array(src_list)
        dst.dy = np.sqrt(np.sum(dy_array**2, axis=0) / len(dy_array))
    restore_data_outside_roi(dst, src_list[0])
    return dst


@computation_function()
def product(src_list: list[SignalObj]) -> SignalObj:
    """Compute the element-wise product of multiple signals.

    The first signal in the list defines the "base" signal. All other signals are
    multiplied element-wise with the base signal.

    .. note::

        If all signals share the same region of interest (ROI), the product is performed
        only within the ROI.

    .. note::

        Uncertainties are propagated.

    .. warning::

        It is assumed that all signals have the same size and x-coordinates.

    Args:
        src_list: List of source signals.

    Returns:
        Signal object representing the product of the source signals.
    """
    dst = dst_n_to_1(src_list, "Π")  # `dst` data is initialized to `src_list[0]` data.
    y_array = signals_y_to_array(src_list)
    dst.y = np.prod(y_array, axis=0)
    if is_uncertainty_data_available(src_list):
        dy_array = signals_dy_to_array(src_list)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            uncertainty = np.abs(dst.y) * np.sqrt(
                np.sum((dy_array / y_array) ** 2, axis=0)
            )
            uncertainty[np.isinf(uncertainty)] = np.nan
            dst.dy = uncertainty
    restore_data_outside_roi(dst, src_list[0])
    return dst


@computation_function()
def addition_constant(src: SignalObj, p: ConstantParam) -> SignalObj:
    """Compute the sum of a signal and a constant value.

    The function adds a constant value to each element of the input signal.

    .. note::

        If the signal has a region of interest (ROI), the addition is performed
        only within the ROI.

    .. note::

        Uncertainties are propagated.

    Args:
        src: Input signal object.
        p: Constant value.

    Returns:
        Result signal object representing the sum of the input signal and the constant.
    """
    # Uncertainty propagation: dst_1_to_1() copies all data including uncertainties.
    # For addition with constant: σ(y + c) = σ(y), so no modification needed.
    dst = dst_1_to_1(src, "+", str(p.value))
    dst.y += p.value
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def difference_constant(src: SignalObj, p: ConstantParam) -> SignalObj:
    """Compute the difference between a signal and a constant value.

    The function subtracts a constant value from each element of the input signal.

    .. note::

        If the signal has a region of interest (ROI), the subtraction is performed
        only within the ROI.

    .. note::

        Uncertainties are propagated.

    Args:
        src: Input signal object.
        p: Constant value.

    Returns:
        Result signal object representing the difference between the input signal and
        the constant.
    """
    # Uncertainty propagation: dst_1_to_1() copies all data including uncertainties.
    # For subtraction with constant: σ(y - c) = σ(y), so no modification needed.
    dst = dst_1_to_1(src, "-", str(p.value))
    dst.y -= p.value
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def product_constant(src: SignalObj, p: ConstantParam) -> SignalObj:
    """Compute the product of a signal and a constant value.

    The function multiplies each element of the input signal by a constant value.

    .. note::

        If the signal has a region of interest (ROI), the multiplication is performed
        only within the ROI.

    .. note::

        Uncertainties are propagated.

    Args:
        src: Input signal object.
        p: Constant value.

    Returns:
        Result signal object representing the product of the input signal and the
        constant.
    """
    assert p.value is not None
    # Uncertainty propagation: dst_1_to_1() copies all data including uncertainties.
    # For multiplication with constant: σ(c*y) = |c| * σ(y), so modification needed.
    dst = dst_1_to_1(src, "×", str(p.value))
    dst.y *= p.value
    if is_uncertainty_data_available(src):
        dst.dy *= np.abs(p.value)  # Modify in-place since dy already copied from src
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def division_constant(src: SignalObj, p: ConstantParam) -> SignalObj:
    """Compute the division of a signal by a constant value.

    The function divides each element of the input signal by a constant value.

    .. note::

        If the signal has a region of interest (ROI), the division is performed
        only within the ROI.

    .. note::

        Uncertainties are propagated.

    Args:
        src: Input signal object.
        p: Constant value.

    Returns:
        Result signal object representing the division of the input signal by the
        constant.
    """
    assert p.value is not None
    # Uncertainty propagation: dst_1_to_1() copies all data including uncertainties.
    # For division with constant: σ(y/c) = σ(y) / |c|, so modification needed.
    dst = dst_1_to_1(src, "/", str(p.value))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dst.y /= p.value
        dst.y[np.isinf(dst.y)] = np.nan
        if is_uncertainty_data_available(src):
            dst.dy /= np.abs(p.value)  # Modify in-place since dy already copied
            dst.dy[np.isinf(dst.dy)] = np.nan
    restore_data_outside_roi(dst, src)
    return dst


# MARK: 2_to_1 functions -------------------------------------------------------
# Functions with N input signals + 1 input signal and N output signals
# --------------------------------------------------------------------------------------


@computation_function()
def arithmetic(src1: SignalObj, src2: SignalObj, p: ArithmeticParam) -> SignalObj:
    """Perform an arithmetic operation on two signals.

    The function applies the specified arithmetic operation to each element of the input
    signals.

    .. note::

        The operation is performed only within the region of interest of `src1`.

    .. note::

        Uncertainties are propagated.

    .. warning::

        It is assumed that both signals have the same size and x-coordinates.

    Args:
        src1: First input signal.
        src2: Second input signal.
        p: Arithmetic operation parameters.

    Returns:
        Result signal object representing the arithmetic operation on the input signals.
    """
    initial_dtype = src1.xydata.dtype
    title = p.operation.replace("obj1", "{0}").replace("obj2", "{1}")
    dst = src1.copy(title=title)
    a = ConstantParam.create(value=p.factor)
    b = ConstantParam.create(value=p.constant)
    if p.operator is MathOperator.ADD:
        dst = addition_constant(product_constant(addition([src1, src2]), a), b)
    elif p.operator is MathOperator.SUBTRACT:
        dst = addition_constant(product_constant(difference(src1, src2), a), b)
    elif p.operator is MathOperator.MULTIPLY:
        dst = addition_constant(product_constant(product([src1, src2]), a), b)
    elif p.operator is MathOperator.DIVIDE:
        dst = addition_constant(product_constant(division(src1, src2), a), b)
    # Eventually convert to initial data type
    if p.restore_dtype:
        dst.xydata = dst.xydata.astype(initial_dtype)
    restore_data_outside_roi(dst, src1)
    return dst


@computation_function()
def difference(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Compute the element-wise difference between two signals.

    The function subtracts each element of the second signal from the corresponding
    element of the first signal.

    .. note::

        If both signals share the same region of interest (ROI), the difference is
        performed only within the ROI.

    .. note::

        Uncertainties are propagated.

    .. warning::

        It is assumed that both signals have the same size and x-coordinates.

    Args:
        src1: First input signal.
        src2: Second input signal.

    Returns:
        Result signal object representing the difference between the input signals.
    """
    dst = dst_2_to_1(src1, src2, "-")
    dst.y = src1.y - src2.y
    if is_uncertainty_data_available([src1, src2]):
        dy_array = signals_dy_to_array([src1, src2])
        dst.dy = np.sqrt(np.sum(dy_array**2, axis=0))
    restore_data_outside_roi(dst, src1)
    return dst


@computation_function()
def quadratic_difference(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Compute the normalized difference between two signals.

    The function computes the element-wise difference between the two signals and
    divides the result by sqrt(2.0).

    .. note::

        If both signals share the same region of interest (ROI), the operation is
        performed only within the ROI.

    .. note::

        Uncertainties are propagated. For two input signals with identical standard
        deviations, the standard deviation of the output signal equals the standard
        deviation of each of the input signals.

    .. warning::

        It is assumed that both signals have the same size and x-coordinates.

    Args:
        src1: First input signal.
        src2: Second input signal.

    Returns:
        Result signal object representing the quadratic difference between the input
        signals.
    """
    norm = ConstantParam.create(value=1.0 / np.sqrt(2.0))
    return product_constant(difference(src1, src2), norm)


@computation_function()
def division(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Compute the element-wise division between two signals.

    The function divides each element of the first signal by the corresponding element
    of the second signal.

    .. note::

        If both signals share the same region of interest (ROI), the division is
        performed only within the ROI.

    .. note::

        Uncertainties are propagated.

    .. warning::

        It is assumed that both signals have the same size and x-coordinates.

    Args:
        src1: First input signal.
        src2: Second input signal.

    Returns:
        Result signal object representing the division of the input signals.
    """
    dst = product([src1, inverse(src2)])
    return dst


# MARK: 1_to_1 functions -------------------------------------------------------
# Functions with 1 input image and 1 output image
# --------------------------------------------------------------------------------------


@computation_function()
def extract_rois(src: SignalObj, params: list[ROI1DParam]) -> SignalObj:
    """Extract multiple regions of interest from data

    Args:
        src: source signal
        params: list of ROI parameters

    Returns:
        Signal with multiple regions of interest
    """
    suffix = None
    if len(params) == 1:
        p: ROI1DParam = params[0]
        suffix = f"{p.xmin:.3g}≤x≤{p.xmax:.3g}"
    dst = dst_1_to_1(src, "extract_rois", suffix)
    x, y = src.get_data()
    xout, yout = np.ones_like(x) * np.nan, np.ones_like(y) * np.nan
    for p in params:
        idx1, idx2 = np.searchsorted(x, p.xmin), np.searchsorted(x, p.xmax)
        slice0 = slice(idx1, idx2)
        xout[slice0], yout[slice0] = x[slice0], y[slice0]
    nans = np.isnan(xout) | np.isnan(yout)
    dst.set_xydata(xout[~nans], yout[~nans])
    return dst


@computation_function()
def extract_roi(src: SignalObj, p: ROI1DParam) -> SignalObj:
    """Extract single region of interest from data

    Args:
        src: source signal
        p: ROI parameters

    Returns:
        Signal with single region of interest
    """
    dst = dst_1_to_1(src, "extract_roi", f"{p.xmin:.3g}≤x≤{p.xmax:.3g}")
    x, y = p.get_data(src).copy()
    dst.set_xydata(x, y)
    return dst


@computation_function()
def transpose(src: SignalObj) -> SignalObj:
    """Transpose signal (swap X and Y axes)

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "transpose")
    x, y = src.get_data()
    dst.set_xydata(y, x)
    return dst


@computation_function()
def inverse(src: SignalObj) -> SignalObj:
    """Compute the element-wise inverse of a signal.

    The function computes the reciprocal (1/y) of each element of the input signal.

    .. note::

        If the signal has a region of interest (ROI), the inverse is performed
        only within the ROI.

    .. note::

        Uncertainties are propagated.

    Args:
        src: Input signal object.

    Returns:
        Result signal object representing the inverse of the input signal.
    """
    dst = dst_1_to_1(src, "invert")
    x, y = src.get_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dst.set_xydata(x, np.reciprocal(y))
        dst.y[np.isinf(dst.y)] = np.nan
        if is_uncertainty_data_available(src):
            err = np.abs(dst.y) * (src.dy / np.abs(src.y))
            err[np.isinf(err)] = np.nan
            dst.dy = err
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def absolute(src: SignalObj) -> SignalObj:
    """Compute absolute value with :py:data:`numpy.absolute`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return Wrap1to1Func(np.absolute)(src)


@computation_function()
def real(src: SignalObj) -> SignalObj:
    """Compute real part with :py:func:`numpy.real`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return Wrap1to1Func(np.real)(src)


@computation_function()
def imag(src: SignalObj) -> SignalObj:
    """Compute imaginary part with :py:func:`numpy.imag`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return Wrap1to1Func(np.imag)(src)


class DataTypeSParam(gds.DataSet):
    """Convert signal data type parameters"""

    dtype_str = gds.ChoiceItem(
        _("Destination data type"),
        list(zip(SignalObj.get_valid_dtypenames(), SignalObj.get_valid_dtypenames())),
        help=_("Output image data type."),
    )


@computation_function()
def astype(src: SignalObj, p: DataTypeSParam) -> SignalObj:
    """Convert data type with :py:func:`numpy.astype`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "astype", f"dtype={p.dtype_str}")
    dst.xydata = src.xydata.astype(p.dtype_str)
    return dst


@computation_function()
def log10(src: SignalObj) -> SignalObj:
    """Compute Log10 with :py:data:`numpy.log10`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return Wrap1to1Func(np.log10)(src)


@computation_function()
def exp(src: SignalObj) -> SignalObj:
    """Compute exponential with :py:data:`numpy.exp`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return Wrap1to1Func(np.exp)(src)


@computation_function()
def sqrt(src: SignalObj) -> SignalObj:
    """Compute square root with :py:data:`numpy.sqrt`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return Wrap1to1Func(np.sqrt)(src)


class PowerParam(gds.DataSet):
    """Power parameters"""

    power = gds.FloatItem(_("Power"), default=2.0)


@computation_function()
def power(src: SignalObj, p: PowerParam) -> SignalObj:
    """Compute power with :py:data:`numpy.power`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "^", str(p.power))
    dst.y = np.power(src.y, p.power)

    # Uncertainty propagation: σ(y^n) = |n * y^(n-1)| * σ(y)
    if is_uncertainty_data_available(src):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dst.dy *= np.abs(p.power * np.power(src.y, p.power - 1))
            dst.dy[np.isinf(dst.dy) | np.isnan(dst.dy)] = np.nan

    restore_data_outside_roi(dst, src)
    return dst


class PeakDetectionParam(gds.DataSet):
    """Peak detection parameters"""

    threshold = gds.IntItem(
        _("Threshold"), default=30, min=0, max=100, slider=True, unit="%"
    )
    min_dist = gds.IntItem(_("Minimum distance"), default=1, min=1, unit="points")


@computation_function()
def peak_detection(src: SignalObj, p: PeakDetectionParam) -> SignalObj:
    """Peak detection with
    :py:func:`sigima.tools.signal.peakdetection.peak_indices`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(
        src, "peak_detection", f"threshold={p.threshold}%, min_dist={p.min_dist}pts"
    )
    x, y = src.get_data()
    indices = peakdetection.peak_indices(
        y, thres=p.threshold * 0.01, min_dist=p.min_dist
    )
    dst.set_xydata(x[indices], y[indices])
    dst.set_metadata_option("curvestyle", "Sticks")
    return dst


@computation_function()
def normalize(src: SignalObj, p: NormalizeParam) -> SignalObj:
    """Normalize data with :py:func:`sigima.tools.signal.level.normalize`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "normalize", f"ref={p.method}")
    x, y = src.get_data()
    normalized_y = scaling.normalize(y, p.method)
    dst.set_xydata(x, normalized_y)

    # Uncertainty propagation for normalization
    # σ(y/norm_factor) = σ(y) / norm_factor
    if is_uncertainty_data_available(src):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Calculate normalization factor
            if p.method == "maximum":
                norm_factor = np.max(y)
            elif p.method == "minimum":
                norm_factor = np.min(y)
            elif p.method == "amplitude":
                norm_factor = np.max(y) - np.min(y)
            else:  # mean, rms, etc.
                if p.method == "mean":
                    norm_factor = np.mean(y)
                else:
                    norm_factor = np.sqrt(np.mean(y**2))

            if norm_factor != 0:
                dst.dy /= np.abs(norm_factor)
            else:
                dst.dy[:] = np.nan

    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def derivative(src: SignalObj) -> SignalObj:
    """Compute derivative with :py:func:`numpy.gradient`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "derivative")
    x, y = src.get_data()
    dst.set_xydata(x, np.gradient(y, x))

    # Uncertainty propagation for numerical derivative
    # For gradient using finite differences: σ(dy/dx) ≈ σ(y) / Δx
    if is_uncertainty_data_available(src):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Use the same gradient approach as numpy.gradient for uncertainty
            dst.dy = np.gradient(src.dy, x)
            dst.dy[np.isinf(dst.dy) | np.isnan(dst.dy)] = np.nan

    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def integral(src: SignalObj) -> SignalObj:
    """Compute integral with :py:func:`scipy.integrate.cumulative_trapezoid`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "integral")
    x, y = src.get_data()
    dst.set_xydata(x, spt.cumulative_trapezoid(y, x, initial=0.0))

    # Uncertainty propagation for numerical integration
    # For cumulative trapezoidal integration, uncertainties accumulate
    if is_uncertainty_data_available(src):
        # Propagate uncertainties through cumulative trapezoidal rule
        # σ(∫y dx) ≈ √(Σ(σ(y_i) * Δx_i)²) for independent measurements
        dx = np.diff(x)
        dy_squared = src.dy[:-1] ** 2 + src.dy[1:] ** 2  # Trapezoidal rule uncertainty
        # Propagated variance for trapezoidal integration
        dst.dy[0] = 0.0  # Initial value has no uncertainty
        dst.dy[1:] = np.sqrt(np.cumsum(dy_squared * (dx**2) / 4))

    restore_data_outside_roi(dst, src)
    return dst


class XYCalibrateParam(gds.DataSet):
    """Signal calibration parameters"""

    axes = (("x", _("X-axis")), ("y", _("Y-axis")))
    axis = gds.ChoiceItem(_("Calibrate"), axes, default="y")
    a = gds.FloatItem("a", default=1.0)
    b = gds.FloatItem("b", default=0.0)


@computation_function()
def calibration(src: SignalObj, p: XYCalibrateParam) -> SignalObj:
    """Compute linear calibration

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "calibration", f"{p.axis}={p.a}*{p.axis}+{p.b}")
    x, y = src.get_data()
    if p.axis == "x":
        dst.set_xydata(p.a * x + p.b, y)
        # For X-axis calibration: uncertainties in x are scaled, y unchanged
        if is_uncertainty_data_available(src):
            dst.dx = np.abs(p.a) * src.dx if src.dx is not None else None
            # Y uncertainties remain the same
    else:
        dst.set_xydata(x, p.a * y + p.b)
        # For Y-axis calibration: σ(a*y + b) = |a| * σ(y)
        if is_uncertainty_data_available(src):
            dst.dy *= np.abs(p.a)
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def clip(src: SignalObj, p: ClipParam) -> SignalObj:
    """Compute maximum data clipping with :py:func:`numpy.clip`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    return Wrap1to1Func(np.clip, a_min=p.lower, a_max=p.upper)(src)


@computation_function()
def offset_correction(src: SignalObj, p: ROI1DParam) -> SignalObj:
    """Correct offset: subtract the mean value of the signal in the specified range
    (baseline correction)

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "offset_correction", f"{p.xmin:.3g}≤x≤{p.xmax:.3g}")
    _roi_x, roi_y = p.get_data(src)
    dst.y -= np.mean(roi_y)
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def gaussian_filter(src: SignalObj, p: GaussianParam) -> SignalObj:
    """Compute gaussian filter with :py:func:`scipy.ndimage.gaussian_filter`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    return Wrap1to1Func(spi.gaussian_filter, sigma=p.sigma)(src)


@computation_function()
def moving_average(src: SignalObj, p: MovingAverageParam) -> SignalObj:
    """Compute moving average with :py:func:`scipy.ndimage.uniform_filter`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    return Wrap1to1Func(
        spi.uniform_filter, size=p.n, mode=p.mode.value, func_name="moving_average"
    )(src)


@computation_function()
def moving_median(src: SignalObj, p: MovingMedianParam) -> SignalObj:
    """Compute moving median with :py:func:`scipy.ndimage.median_filter`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    return Wrap1to1Func(
        spi.median_filter, size=p.n, mode=p.mode.value, func_name="moving_median"
    )(src)


@computation_function()
def wiener(src: SignalObj) -> SignalObj:
    """Compute Wiener filter with :py:func:`scipy.signal.wiener`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return Wrap1to1Func(sps.wiener)(src)


@computation_function()
def add_gaussian_noise(src: SignalObj, p: NormalDistribution1DParam) -> SignalObj:
    """Add normal noise to the input signal.

    Args:
        src: Source signal.
        p: Parameters.

    Returns:
        Result signal object.
    """
    param = NormalDistribution1DParam.create(seed=p.seed, mu=p.mu, sigma=p.sigma)
    param.xmin = src.x[0]
    param.xmax = src.x[-1]
    param.size = src.x.size
    noise = create_signal_from_param(param)
    dst = dst_1_to_1(src, "add_gaussian_noise", f"µ={p.mu}, σ={p.sigma}")
    dst.xydata = addition([src, noise]).xydata
    return dst


@computation_function()
def add_poisson_noise(src: SignalObj, p: PoissonDistributionParam) -> SignalObj:
    """Add Poisson noise to the input signal.

    Args:
        src: Source signal.
        p: Parameters.

    Returns:
        Result signal object.
    """
    param = PoissonDistributionParam.create(seed=p.seed, lam=p.lam)
    param.xmin = src.x[0]
    param.xmax = src.x[-1]
    param.size = src.x.size
    noise = create_signal_from_param(param)
    dst = dst_1_to_1(src, "add_poisson_noise", f"λ={p.lam}")
    dst.xydata = addition([src, noise]).xydata
    return dst


@computation_function()
def add_uniform_noise(src: SignalObj, p: UniformDistribution1DParam) -> SignalObj:
    """Add uniform noise to the input signal.

    Args:
        src: Source signal.
        p: Parameters.

    Returns:
        Result signal object.
    """
    param = UniformDistribution1DParam.create(seed=p.seed, vmin=p.vmin, vmax=p.vmax)
    param.xmin = src.x[0]
    param.xmax = src.x[-1]
    param.size = src.x.size
    noise = create_signal_from_param(param)
    dst = dst_1_to_1(src, "add_uniform_noise", f"low={p.vmin}, high={p.vmax}")
    dst.xydata = addition([src, noise]).xydata
    return dst


class FilterType(Enum):
    """Filter types"""

    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    BANDSTOP = "bandstop"


class BaseHighLowBandParam(gds.DataSet):
    """Base class for high-pass, low-pass, band-pass and band-stop filters"""

    methods = (
        ("bessel", _("Bessel")),
        ("brickwall", _("Brick wall")),
        ("butter", _("Butterworth")),
        ("cheby1", _("Chebyshev type 1")),
        ("cheby2", _("Chebyshev type 2")),
        ("ellip", _("Elliptic")),
        ("brickwall", _("Brickwall")),
    )

    TYPE: FilterType = FilterType.LOWPASS
    _type_prop = gds.GetAttrProp("TYPE")

    # Must be overwriten by the child class
    _method_prop = gds.GetAttrProp("method")
    method = gds.ChoiceItem(_("Filter method"), methods).set_prop(
        "display", store=_method_prop
    )

    order = gds.IntItem(_("Filter order"), default=3, min=1).set_prop(
        "display",
        active=gds.FuncProp(_method_prop, lambda x: x not in ("brickwall",)),
    )
    cut0 = gds.FloatItem(
        _("Low cutoff frequency"), min=0.0, nonzero=True, unit="Hz", allow_none=True
    )
    cut1 = gds.FloatItem(
        _("High cutoff frequency"), min=0.0, nonzero=True, unit="Hz", allow_none=True
    ).set_prop(
        "display",
        hide=gds.FuncProp(
            _type_prop, lambda x: x in (FilterType.LOWPASS, FilterType.HIGHPASS)
        ),
    )
    rp = gds.FloatItem(
        _("Passband ripple"), min=0.0, default=1.0, nonzero=True, unit="dB"
    ).set_prop(
        "display",
        active=gds.FuncProp(_method_prop, lambda x: x in ("cheby1", "ellip")),
    )
    rs = gds.FloatItem(
        _("Stopband attenuation"), min=0.0, default=1.0, nonzero=True, unit="dB"
    ).set_prop(
        "display",
        active=gds.FuncProp(_method_prop, lambda x: x in ("cheby2", "ellip")),
    )

    _zp_prop = gds.GetAttrProp("zero_padding")
    zero_padding = gds.BoolItem(
        _("Zero padding"),
        default=True,
    ).set_prop(
        "display",
        active=gds.FuncProp(_method_prop, lambda x: x == "brickwall"),
        store=_zp_prop,
    )
    nfft = gds.IntItem(
        _("Minimum FFT points number"),
        default=0,
    ).set_prop(
        "display",
        active=gds.FuncPropMulti(
            [_method_prop, _zp_prop],
            lambda x, y: x == "brickwall" and y,
        ),
    )

    @staticmethod
    def get_nyquist_frequency(obj: SignalObj) -> float:
        """Return the Nyquist frequency of a signal object

        Args:
            obj: signal object
        """
        fs = float(obj.x.size - 1) / (obj.x[-1] - obj.x[0])
        return fs / 2.0

    def update_from_obj(self, obj: SignalObj) -> None:
        """Update the filter parameters from a signal object

        Args:
            obj: signal object
        """
        f_nyquist = self.get_nyquist_frequency(obj)
        if self.cut0 is None:
            if self.TYPE is FilterType.LOWPASS:
                self.cut0 = 0.1 * f_nyquist
            elif self.TYPE is FilterType.HIGHPASS:
                self.cut0 = 0.9 * f_nyquist
            elif self.TYPE is FilterType.BANDPASS:
                self.cut0 = 0.1 * f_nyquist
                self.cut1 = 0.9 * f_nyquist
            elif self.TYPE is FilterType.BANDSTOP:
                self.cut0 = 0.4 * f_nyquist
                self.cut1 = 0.6 * f_nyquist

    def get_filter_params(self, obj: SignalObj) -> tuple[float | str, float | str]:
        """Return the filter parameters (a and b) as a tuple. These parameters are used
        in the scipy.signal filter functions (eg. `scipy.signal.filtfilt`).

        Args:
            obj: signal object

        Returns:
            tuple: filter parameters
        """
        f_nyquist = self.get_nyquist_frequency(obj)
        func = getattr(sps, self.method)
        args: list[float | str | tuple[float, ...]] = [self.order]  # type: ignore
        if self.method == "cheby1":
            args += [self.rp]
        elif self.method == "cheby2":
            args += [self.rs]
        elif self.method == "ellip":
            args += [self.rp, self.rs]
        if self.TYPE in (FilterType.HIGHPASS, FilterType.LOWPASS):
            args += [self.cut0 / f_nyquist]
        else:
            args += [[self.cut0 / f_nyquist, self.cut1 / f_nyquist]]
        args += [self.TYPE.value]
        return func(*args)


class LowPassFilterParam(BaseHighLowBandParam):
    """Low-pass filter parameters"""

    TYPE = FilterType.LOWPASS

    # Redefine cut0 just to change its label (instead of "Low cutoff frequency")
    cut0 = gds.FloatItem(_("Cutoff frequency"), min=0, nonzero=True, unit="Hz")


class HighPassFilterParam(BaseHighLowBandParam):
    """High-pass filter parameters"""

    TYPE = FilterType.HIGHPASS

    # Redefine cut0 just to change its label (instead of "High cutoff frequency")
    cut0 = gds.FloatItem(_("Cutoff frequency"), min=0, nonzero=True, unit="Hz")


class BandPassFilterParam(BaseHighLowBandParam):
    """Band-pass filter parameters"""

    TYPE = FilterType.BANDPASS


class BandStopFilterParam(BaseHighLowBandParam):
    """Band-stop filter parameters"""

    TYPE = FilterType.BANDSTOP


def frequency_filter(src: SignalObj, p: BaseHighLowBandParam) -> SignalObj:
    """Compute frequency filter (low-pass, high-pass, band-pass, band-stop),
    with :py:func:`scipy.signal.filtfilt`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object

    .. note::

        Uses zero-phase filtering (`filtfilt`) when possible for better phase response.
        If numerical instability occurs (e.g., singular matrix errors), automatically
        falls back to forward filtering (`lfilter`) with a warning. This ensures
        cross-platform compatibility while maintaining optimal filtering when possible.
    """
    name = f"{p.TYPE.value}"
    suffix = "" if p.method == "brickwall" else f"order={p.order:d}, "
    if p.TYPE in (FilterType.LOWPASS, FilterType.HIGHPASS):
        suffix += f"cutoff={p.cut0:.2f}"
    else:
        suffix += f"cutoff={p.cut0:.2f}:{p.cut1:.2f}"
    dst = dst_1_to_1(src, name, suffix)

    if p.method == "brickwall":
        src_padded = src.copy()
        if p.zero_padding and p.nfft is not None:
            size_padded = ZeroPadding1DParam.next_power_of_two(max(p.nfft, src.y.size))
            if size_padded > 1:
                src_padded = zero_padding(
                    src_padded,
                    ZeroPadding1DParam.create(
                        location="append",
                        strategy="custom",
                        n=size_padded,
                    ),
                )
        x_padded, y_padded = src_padded.get_data()
        x, y = fourier.brickwall_filter(
            x_padded, y_padded, p.TYPE.value, p.cut0, p.cut1
        )
        dst.set_xydata(x, y)
    else:
        b, a = p.get_filter_params(dst)
        try:
            # Prefer zero-phase filtering
            dst.y = sps.filtfilt(b, a, dst.y)
        except np.linalg.LinAlgError:
            # Fallback to forward filtering if filtfilt fails due to numerical issues
            warnings.warn(
                "Zero-phase filtering failed due to numerical instability. "
                "Using forward filtering instead.",
                UserWarning,
                stacklevel=2,
            )
            dst.y = sps.lfilter(b, a, dst.y)

    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def lowpass(src: SignalObj, p: LowPassFilterParam) -> SignalObj:
    """Compute low-pass filter with :py:func:`scipy.signal.filtfilt`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    return frequency_filter(src, p)


@computation_function()
def highpass(src: SignalObj, p: HighPassFilterParam) -> SignalObj:
    """Compute high-pass filter with :py:func:`scipy.signal.filtfilt`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    return frequency_filter(src, p)


@computation_function()
def bandpass(src: SignalObj, p: BandPassFilterParam) -> SignalObj:
    """Compute band-pass filter with :py:func:`scipy.signal.filtfilt`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    return frequency_filter(src, p)


@computation_function()
def bandstop(src: SignalObj, p: BandStopFilterParam) -> SignalObj:
    """Compute band-stop filter with :py:func:`scipy.signal.filtfilt`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    return frequency_filter(src, p)


class ZeroPadding1DParam(gds.DataSet):
    """ZeroPadding1DParam manages the parameters for applying zero-padding to signals.

    Attributes:
        strategies: Available strategies ("next_pow2", "double", "triple", "custom").
        strategy: Choice item for selecting the zero-padding strategy.
        locations: Available locations for padding ("append", "prepend", "both").
        location: Choice item for selecting where to add the padding.
        n: Number of points to add as padding (active only for "custom" strategy).
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize zero padding parameters.

        Args:
            *args: Variable length argument list passed to the superclass.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)
        self.__obj: SignalObj | None = None

    def update_from_obj(self, obj: SignalObj) -> None:
        """Update parameters from signal.

        Args:
            obj: Signal object from which to update the dataset.
        """
        self.__obj = obj
        self.strategy_callback(None, self.strategy)

    @staticmethod
    def next_power_of_two(size: int) -> int:
        """Compute the next power of two greater than or equal to the given size.

        Args:
            size: The input integer.

        Returns:
            The smallest power of two greater than or equal to 'size'.
        """
        return 2 ** (ceil(log2(size)))

    def strategy_callback(self, _, value):
        """Callback for strategy choice item.

        Args:
            _: Unused argument (in this context).
            value: The selected strategy value.
        """
        assert self.__obj is not None
        assert self.__obj.x is not None
        size = self.__obj.x.size
        if value == "next_pow2":
            self.n = self.next_power_of_two(size) - size
        elif value == "double":
            self.n = size
        elif value == "triple":
            self.n = 2 * size

    strategies = ("next_pow2", "double", "triple", "custom")
    _prop = gds.GetAttrProp("strategy")
    strategy = gds.ChoiceItem(
        _("Strategy"), zip(strategies, strategies), default=strategies[0]
    ).set_prop("display", store=_prop, callback=strategy_callback)
    location = gds.ChoiceItem(
        _("Location"),
        PadLocation,
        default=PadLocation.APPEND,
        help=_("Where to add the padding"),
    )
    _func_prop = gds.FuncProp(_prop, lambda x: x == "custom")
    n = gds.IntItem(
        _("Number of points"), min=1, default=1, help=_("Number of points to add")
    ).set_prop("display", active=_func_prop)


@computation_function()
def zero_padding(src: SignalObj, p: ZeroPadding1DParam) -> SignalObj:
    """Compute zero padding with :py:func:`sigima.tools.signal.fourier.zero_padding`.

    Args:
        src: Source signal.
        p: Parameters.

    Returns:
        Result signal object.
    """
    if p.strategy == "custom":
        suffix = f"n={p.n}"
    else:
        suffix = f"strategy={p.strategy}"

    assert p.n is not None
    if p.location is PadLocation.APPEND:
        n_prepend = 0
        n_append = p.n
    elif p.location is PadLocation.PREPEND:
        n_prepend = p.n
        n_append = 0
    elif p.location is PadLocation.BOTH:
        n_prepend = p.n // 2
        n_append = p.n - n_prepend
    else:
        raise ValueError(f"Unknown location: {p.location}")

    dst = dst_1_to_1(src, "zero_padding", suffix)
    x, y = src.get_data()
    x_padded, y_padded = fourier.zero_padding(x, y, n_prepend, n_append)
    dst.set_xydata(x_padded, y_padded)

    return dst


@computation_function()
def fft(src: SignalObj, p: FFTParam | None = None) -> SignalObj:
    """Compute FFT with :py:func:`sigima.tools.signal.fourier.fft1d`.

    Args:
        src: Source signal.
        p: Parameters.

    Returns:
        Result signal object.
    """
    dst = dst_1_to_1(src, "fft")
    x, y = src.get_data()
    fft_x, fft_y = fourier.fft1d(x, y, shift=bool(True if p is None else p.shift))
    dst.set_xydata(fft_x, fft_y)
    dst.save_attr_to_metadata("xunit", "Hz" if dst.xunit == "s" else "")
    dst.save_attr_to_metadata("yunit", "")
    dst.save_attr_to_metadata("xlabel", _("Frequency"))
    return dst


@computation_function()
def ifft(src: SignalObj) -> SignalObj:
    """Compute the inverse FFT with :py:func:`sigima.tools.signal.fourier.ifft1d`.

    Args:
        src: Source signal.

    Returns:
        Result signal object.
    """
    dst = dst_1_to_1(src, "ifft")
    f, sp = src.get_data()
    x, y = fourier.ifft1d(f, sp)
    dst.set_xydata(x, y)
    dst.restore_attr_from_metadata("xunit", "s" if src.xunit == "Hz" else "")
    dst.restore_attr_from_metadata("yunit", "")
    dst.restore_attr_from_metadata("xlabel", "")
    return dst


@computation_function()
def magnitude_spectrum(src: SignalObj, p: SpectrumParam | None = None) -> SignalObj:
    """Compute magnitude spectrum.

    This function computes the magnitude spectrum of a signal using
    :py:func:`sigima.tools.signal.fourier.magnitude_spectrum`.

    Args:
        src: Source signal.
        p: Parameters.

    Returns:
        Result signal object.
    """
    decibel = bool(p is not None and p.decibel)
    dst = dst_1_to_1(src, "magnitude_spectrum", f"dB={decibel}")
    x, y = src.get_data()
    mag_x, mag_y = fourier.magnitude_spectrum(x, y, decibel=decibel)
    dst.set_xydata(mag_x, mag_y)
    dst.xlabel = _("Frequency")
    dst.xunit = "Hz" if dst.xunit == "s" else ""
    dst.yunit = "dB" if decibel else ""
    return dst


@computation_function()
def phase_spectrum(src: SignalObj) -> SignalObj:
    """Compute phase spectrum.

    This function computes the phase spectrum of a signal using
    :py:func:`sigima.tools.signal.fourier.phase_spectrum`

    Args:
        src: Source signal.

    Returns:
        Result signal object.
    """
    dst = dst_1_to_1(src, "phase_spectrum")
    x, y = src.get_data()
    phase_x, phase_y = fourier.phase_spectrum(x, y)
    dst.set_xydata(phase_x, phase_y)
    dst.xlabel = _("Frequency")
    dst.xunit = "Hz" if dst.xunit == "s" else ""
    dst.yunit = ""
    return dst


@computation_function()
def psd(src: SignalObj, p: SpectrumParam | None = None) -> SignalObj:
    """Compute power spectral density with :py:func:`sigima.tools.signal.fourier.psd`.

    Args:
        src: Source signal.
        p: Parameters.

    Returns:
        Result signal object.
    """
    decibel = p is not None and p.decibel
    dst = dst_1_to_1(src, "psd", f"dB={decibel}")
    x, y = src.get_data()
    psd_x, psd_y = fourier.psd(x, y, decibel=decibel)
    dst.set_xydata(psd_x, psd_y)
    dst.xlabel = _("Frequency")
    dst.xunit = "Hz" if dst.xunit == "s" else ""
    dst.yunit = "dB/Hz" if decibel else ""
    return dst


class PolynomialFitParam(gds.DataSet):
    """Polynomial fitting parameters"""

    degree = gds.IntItem(_("Degree"), 3, min=1, max=10, slider=True)


@computation_function()
def histogram(src: SignalObj, p: HistogramParam) -> SignalObj:
    """Compute histogram with :py:func:`numpy.histogram`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    data = src.get_masked_view().compressed()
    suffix = p.get_suffix(data)  # Also updates p.lower and p.upper

    # Compute histogram:
    y, bin_edges = np.histogram(data, bins=p.bins, range=(p.lower, p.upper))
    x = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Note: we use the `new_signal_result` function to create the result signal object
    # because the `dst_1_to_1` would copy the source signal, which is not what we want
    # here (we want a brand new signal object).
    dst = new_signal_result(
        src,
        "histogram",
        suffix=suffix,
        units=(src.yunit, ""),
        labels=(src.ylabel, _("Counts")),
    )
    dst.set_xydata(x, y)
    dst.set_metadata_option("shade", 0.5)
    dst.set_metadata_option("curvestyle", "Steps")
    return dst


class InterpolationParam(gds.DataSet):
    """Interpolation parameters"""

    methods = (
        ("linear", _("Linear")),
        ("spline", _("Spline")),
        ("quadratic", _("Quadratic")),
        ("cubic", _("Cubic")),
        ("barycentric", _("Barycentric")),
        ("pchip", _("PCHIP")),
    )
    method = gds.ChoiceItem(_("Interpolation method"), methods, default="linear")
    fill_value = gds.FloatItem(
        _("Fill value"),
        default=None,
        help=_(
            "Value to use for points outside the interpolation domain (used only "
            "with linear, cubic and pchip methods)."
        ),
        check=False,
    )


@computation_function()
def interpolate(src1: SignalObj, src2: SignalObj, p: InterpolationParam) -> SignalObj:
    """Interpolate data with
    :py:func:`sigima.tools.signal.interpolation.interpolate`

    Args:
        src1: source signal 1
        src2: source signal 2
        p: parameters

    Returns:
        Result signal object
    """
    suffix = f"method={p.method}"
    if p.fill_value is not None and p.method in ("linear", "cubic", "pchip"):
        suffix += f", fill_value={p.fill_value}"
    dst = dst_2_to_1(src1, src2, "interpolate", suffix)
    x1, y1 = src1.get_data()
    xnew, _y2 = src2.get_data()
    ynew = interpolation.interpolate(x1, y1, xnew, p.method, p.fill_value)
    dst.set_xydata(xnew, ynew)
    return dst


class ResamplingParam(InterpolationParam):
    """Resample parameters"""

    xmin = gds.FloatItem(_("X<sub>min</sub>"), allow_none=True)
    xmax = gds.FloatItem(_("X<sub>max</sub>"), allow_none=True)
    _prop = gds.GetAttrProp("dx_or_nbpts")
    _modes = (("dx", "ΔX"), ("nbpts", _("Number of points")))
    mode = gds.ChoiceItem(_("Mode"), _modes, default="nbpts", radio=True).set_prop(
        "display", store=_prop
    )
    dx = gds.FloatItem("ΔX", allow_none=True).set_prop(
        "display", active=gds.FuncProp(_prop, lambda x: x == "dx")
    )
    nbpts = gds.IntItem(_("Number of points"), allow_none=True).set_prop(
        "display", active=gds.FuncProp(_prop, lambda x: x == "nbpts")
    )

    def update_from_obj(self, obj: SignalObj) -> None:
        """Update parameters from a signal object."""
        if self.xmin is None:
            self.xmin = obj.x[0]
        if self.xmax is None:
            self.xmax = obj.x[-1]
        if self.dx is None:
            self.dx = obj.x[1] - obj.x[0]
        if self.nbpts is None:
            self.nbpts = len(obj.x)


@computation_function()
def resampling(src: SignalObj, p: ResamplingParam) -> SignalObj:
    """Resample data with :py:func:`sigima.tools.signal.interpolation.interpolate`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    suffix = f"method={p.method}"
    if p.fill_value is not None and p.method in ("linear", "cubic", "pchip"):
        suffix += f", fill_value={p.fill_value}"
    if p.mode == "dx":
        suffix += f", dx={p.dx:.3f}"
    else:
        suffix += f", nbpts={p.nbpts:d}"
    dst = dst_1_to_1(src, "resample", suffix)
    x, y = src.get_data()
    if p.mode == "dx":
        xnew = np.arange(p.xmin, p.xmax, p.dx)
    else:
        xnew = np.linspace(p.xmin, p.xmax, p.nbpts)
    ynew = interpolation.interpolate(x, y, xnew, p.method, p.fill_value)
    dst.set_xydata(xnew, ynew)
    return dst


class DetrendingParam(gds.DataSet):
    """Detrending parameters"""

    methods = (("linear", _("Linear")), ("constant", _("Constant")))
    method = gds.ChoiceItem(_("Detrending method"), methods, default="linear")


@computation_function()
def detrending(src: SignalObj, p: DetrendingParam) -> SignalObj:
    """Detrend data with :py:func:`scipy.signal.detrend`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "detrending", f"method={p.method}")
    x, y = src.get_data()
    dst.set_xydata(x, sps.detrend(y, type=p.method))
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def xy_mode(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Simulate the X-Y mode of an oscilloscope.

    Use the first signal as the X-axis and the second signal as the Y-axis.

    Args:
        src1: First input signal (X-axis).
        src2: Second input signal (Y-axis).

    Returns:
        A signal object representing the X-Y mode.
    """
    dst = dst_2_to_1(src1, src2, "", "X-Y Mode")
    p = ResamplingParam()
    p.xmin = max(src1.x[0], src2.x[0])
    p.xmax = min(src1.x[-1], src2.x[-1])
    assert p.xmin < p.xmax, "X-Y mode: No overlap between signals."
    p.mode = "nbpts"
    p.nbpts = min(src1.x.size, src2.x.size)
    _, y1 = resampling(src1, p).get_data()
    _, y2 = resampling(src2, p).get_data()
    dst.set_xydata(y1, y2)
    dst.title = "{1} = f({0})"
    restore_data_outside_roi(dst, src1)
    return dst


@computation_function()
def convolution(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Compute convolution of two signals
    with :py:func:`scipy.signal.convolve`

    Args:
        src1: source signal 1
        src2: source signal 2

    Returns:
        Result signal object
    """
    dst = dst_2_to_1(src1, src2, "⊛")
    x1, y1 = src1.get_data()
    _x2, y2 = src2.get_data()
    ynew = np.real(sps.convolve(y1, y2, mode="same"))
    dst.set_xydata(x1, ynew)
    restore_data_outside_roi(dst, src1)
    return dst


class WindowingParam(gds.DataSet):
    """Windowing parameters."""

    methods = (
        ("barthann", "Barthann"),
        ("bartlett", "Bartlett"),
        ("blackman", "Blackman"),
        ("blackman-harris", "Blackman-Harris"),
        ("bohman", "Bohman"),
        ("boxcar", "Boxcar"),
        ("cosine", _("Cosine")),
        ("exponential", _("Exponential")),
        ("flat-top", _("Flat top")),
        ("gaussian", _("Gaussian")),
        ("hamming", "Hamming"),
        ("hann", "Hann"),
        ("kaiser", "Kaiser"),
        ("lanczos", "Lanczos"),
        ("nuttall", "Nuttall"),
        ("parzen", "Parzen"),
        ("taylor", "Taylor"),
        ("tukey", "Tukey"),
    )
    _meth_prop = gds.GetAttrProp("method")
    method = gds.ChoiceItem(_("Method"), methods, default="hamming").set_prop(
        "display", store=_meth_prop
    )
    alpha = gds.FloatItem(
        "Alpha",
        default=0.5,
        help=_("Shape parameter of the Tukey windowing function"),
    ).set_prop("display", active=gds.FuncProp(_meth_prop, lambda x: x == "tukey"))
    beta = gds.FloatItem(
        "Beta",
        default=14.0,
        help=_("Shape parameter of the Kaiser windowing function"),
    ).set_prop("display", active=gds.FuncProp(_meth_prop, lambda x: x == "kaiser"))
    sigma = gds.FloatItem(
        "Sigma",
        default=0.5,
        help=_("Shape parameter of the Gaussian windowing function"),
    ).set_prop("display", active=gds.FuncProp(_meth_prop, lambda x: x == "gaussian"))


@computation_function()
def apply_window(src: SignalObj, p: WindowingParam) -> SignalObj:
    """Compute windowing with :py:func:`sigima.tools.signal.windowing.apply_window`.

    Available methods are listed in :py:attr:`WindowingParam.methods`.

    Args:
        src: Source signal.
        p: Parameters for windowing.

    Returns:
        Result signal object.
    """
    suffix = f"method={p.method}"
    if p.method == "gaussian":
        suffix += f", sigma={p.sigma:.3f}"
    elif p.method == "kaiser":
        suffix += f", beta={p.beta:.3f}"
    elif p.method == "tukey":
        suffix += f", alpha={p.alpha:.3f}"
    dst = dst_1_to_1(src, "apply_window", suffix)
    assert p.method is not None
    assert p.alpha is not None
    dst.y = windowing.apply_window(dst.y, p.method, p.alpha)
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def reverse_x(src: SignalObj) -> SignalObj:
    """Reverse x-axis

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "reverse_x")
    dst.y = dst.y[::-1]
    return dst


class AngleUnitParam(gds.DataSet):
    """Choice of angle unit."""

    units = (("rad", _("Radian")), ("deg", _("Degree")))
    unit = gds.ChoiceItem(_("Angle unit"), units, default="rad")


@computation_function()
def to_polar(src: SignalObj, p: AngleUnitParam) -> SignalObj:
    """Convert cartesian coordinates to polar coordinates.

    This function converts the x and y coordinates of a signal to polar coordinates
    using :py:func:`sigima.tools.coordinates.to_polar`.

    Args:
        src: Source signal.
        p: Parameters.

    Returns:
        Result signal object.

    Raises:
        ValueError: If the x and y units are not the same.
    """
    assert p.unit is not None
    if src.xunit != src.yunit:
        warnings.warn(f"X and Y units are not the same: {src.xunit} != {src.yunit}.")
    dst = dst_1_to_1(src, "Polar coordinates", f"unit={p.unit}")
    x, y = src.get_data()
    r, theta = coordinates.to_polar(x, y, p.unit)
    dst.set_xydata(r, theta)
    dst.xlabel = _("Radius")
    dst.ylabel = _("Angle")
    dst.yunit = p.unit
    return dst


@computation_function()
def to_cartesian(src: SignalObj, p: AngleUnitParam) -> SignalObj:
    """Convert polar coordinates to cartesian coordinates.

    This function converts the r and theta coordinates of a signal to cartesian
    coordinates using :py:func:`sigima.tools.coordinates.to_cartesian`.

    Args:
        src: Source signal.
        p: Parameters.

    Returns:
        Result signal object.

    .. note::

        This function assumes that the x-axis represents the radius and the y-axis
        represents the angle. Negative values are not allowed for the radius, and will
        be clipped to 0 (a warning will be raised).
    """
    dst = dst_1_to_1(src, "Cartesian coordinates", f"unit={p.unit}")
    r, theta = src.get_data()
    x, y = coordinates.to_cartesian(r, theta, p.unit)
    dst.set_xydata(x, y)
    dst.xlabel = _("x")
    dst.ylabel = _("y")
    dst.yunit = src.xunit
    return dst


class AllanVarianceParam(gds.DataSet):
    """Allan variance parameters"""

    max_tau = gds.IntItem("Max τ", default=100, min=1, unit="pts")


@computation_function()
def allan_variance(src: SignalObj, p: AllanVarianceParam) -> SignalObj:
    """Compute Allan variance with
    :py:func:`sigima.tools.signal.stability.allan_variance`.

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "allan_variance", f"max_tau={p.max_tau}")
    x, y = src.get_data()
    tau_values = np.arange(1, p.max_tau + 1)
    avar = stability.allan_variance(x, y, tau_values)
    dst.set_xydata(tau_values, avar)
    return dst


@computation_function()
def allan_deviation(src: SignalObj, p: AllanVarianceParam) -> SignalObj:
    """Compute Allan deviation with
    :py:func:`sigima.tools.signal.stability.allan_deviation`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "allan_deviation", f"max_tau={p.max_tau}")
    x, y = src.get_data()
    tau_values = np.arange(1, p.max_tau + 1)
    adev = stability.allan_deviation(x, y, tau_values)
    dst.set_xydata(tau_values, adev)
    return dst


@computation_function()
def overlapping_allan_variance(src: SignalObj, p: AllanVarianceParam) -> SignalObj:
    """Compute Overlapping Allan variance.

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "overlapping_allan_variance", f"max_tau={p.max_tau}")
    x, y = src.get_data()
    tau_values = np.arange(1, p.max_tau + 1)
    oavar = stability.overlapping_allan_variance(x, y, tau_values)
    dst.set_xydata(tau_values, oavar)
    return dst


@computation_function()
def modified_allan_variance(src: SignalObj, p: AllanVarianceParam) -> SignalObj:
    """Compute Modified Allan variance.

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "modified_allan_variance", f"max_tau={p.max_tau}")
    x, y = src.get_data()
    tau_values = np.arange(1, p.max_tau + 1)
    mavar = stability.modified_allan_variance(x, y, tau_values)
    dst.set_xydata(tau_values, mavar)
    return dst


@computation_function()
def hadamard_variance(src: SignalObj, p: AllanVarianceParam) -> SignalObj:
    """Compute Hadamard variance.

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "hadamard_variance", f"max_tau={p.max_tau}")
    x, y = src.get_data()
    tau_values = np.arange(1, p.max_tau + 1)
    hvar = stability.hadamard_variance(x, y, tau_values)
    dst.set_xydata(tau_values, hvar)
    return dst


@computation_function()
def total_variance(src: SignalObj, p: AllanVarianceParam) -> SignalObj:
    """Compute Total variance.

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "total_variance", f"max_tau={p.max_tau}")
    x, y = src.get_data()
    tau_values = np.arange(1, p.max_tau + 1)
    tvar = stability.total_variance(x, y, tau_values)
    dst.set_xydata(tau_values, tvar)
    return dst


@computation_function()
def time_deviation(src: SignalObj, p: AllanVarianceParam) -> SignalObj:
    """Compute Time Deviation (TDEV).

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "time_deviation", f"max_tau={p.max_tau}")
    x, y = src.get_data()
    tau_values = np.arange(1, p.max_tau + 1)
    tdev = stability.time_deviation(x, y, tau_values)
    dst.set_xydata(tau_values, tdev)
    return dst


# MARK: compute_1_to_0 functions -------------------------------------------------------
# Functions with 1 input signal and 0 output signals (ResultShape or ResultProperties)
# --------------------------------------------------------------------------------------


def compute_geometry_from_obj(
    title: str,
    shape: KindShape,
    obj: SignalObj,
    func: Callable,
    *args: Any,
) -> GeometryResult | None:
    """Calculate result shape by executing a computation function on a signal object,
    taking into account the signal ROIs.

    Args:
        title: result title
        shape: result shape kind
        obj: input image object
        func: computation function
        *args: computation function arguments

    Returns:
        Result shape object or None if no result is found

    .. warning::

        The computation function must take either a single argument (the data) or
        multiple arguments (the data followed by the computation parameters).

        Moreover, the computation function must return a 1D NumPy array (or a list,
        or a tuple) containing the result of the computation.
    """
    rows: list[np.ndarray] = []
    roi_idx: list[int] = []
    for i_roi in obj.iterate_roi_indices():
        x, y = obj.get_data(i_roi)
        if args is None:
            results: np.ndarray = func(x, y)
        else:
            results: np.ndarray = func(x, y, *args)
        if not isinstance(results, (np.ndarray, list, tuple)):
            raise ValueError(
                "The computation function must return a NumPy array, a list or a tuple"
            )
        results = np.array(results)
        if results.size:
            if results.ndim != 1:
                raise ValueError(
                    "The computation function must return a 1D NumPy array"
                )
            rows.append(results.tolist())
            roi_idx.append(NO_ROI if i_roi is None else int(i_roi))
    if rows:
        array = np.vstack(rows)
        return GeometryResult(title, shape, array, np.asarray(roi_idx, dtype=int))
    return None


class FWHMParam(gds.DataSet):
    """FWHM parameters"""

    methods = (
        ("zero-crossing", _("Zero-crossing")),
        ("gauss", _("Gaussian fit")),
        ("lorentz", _("Lorentzian fit")),
        ("voigt", _("Voigt fit")),
    )
    method = gds.ChoiceItem(_("Method"), methods, default="zero-crossing")
    xmin = gds.FloatItem(
        "X<sub>MIN</sub>",
        default=None,
        check=False,
        help=_("Lower X boundary (empty for no limit, i.e. start of the signal)"),
    )
    xmax = gds.FloatItem(
        "X<sub>MAX</sub>",
        default=None,
        check=False,
        help=_("Upper X boundary (empty for no limit, i.e. end of the signal)"),
    )


@computation_function()
def fwhm(obj: SignalObj, param: FWHMParam) -> GeometryResult | None:
    """Compute FWHM with :py:func:`sigima.tools.signal.pulse.fwhm`

    Args:
        obj: source signal
        param: parameters

    Returns:
        Segment coordinates
    """
    return compute_geometry_from_obj(
        "fwhm",
        "segment",
        obj,
        pulse.fwhm,
        param.method,
        param.xmin,
        param.xmax,
    )


@computation_function()
def fw1e2(obj: SignalObj) -> GeometryResult | None:
    """Compute FW at 1/e² with :py:func:`sigima.tools.signal.pulse.fw1e2`

    Args:
        obj: source signal

    Returns:
        Segment coordinates
    """
    return compute_geometry_from_obj("fw1e2", "segment", obj, pulse.fw1e2)


class OrdinateParam(gds.DataSet):
    """Ordinate parameter."""

    y = gds.FloatItem(_("Ordinate"), default=0.0)


@computation_function()
def full_width_at_y(obj: SignalObj, p: OrdinateParam) -> GeometryResult | None:
    """
    Compute full width at a given y value for a signal object.

    Args:
        obj: The signal object containing x and y data.
        p: The ordinate parameter dataset

    Returns:
        Segment coordinates
    """
    return compute_geometry_from_obj("∆X", "segment", obj, pulse.full_width_at_y, p.y)


@computation_function()
def x_at_y(obj: SignalObj, p: OrdinateParam) -> TableResult:
    """
    Compute the smallest x-value at a given y-value for a signal object.

    Args:
        obj: The signal object containing x and y data.
        p: The parameter dataset for finding the abscissa.

    Returns:
         An object containing the x-value.
    """
    table = TableResultBuilder(f"x|y={p.y}")
    table.add(
        lambda xy: features.find_first_x_at_given_y_value(xy[0], xy[1], p.y),
        "x@y",
        "x = %g {.xunit}",
    )
    return table.compute(obj)


class AbscissaParam(gds.DataSet):
    """Abscissa parameter."""

    x = gds.FloatItem(_("Abscissa"), default=0.0)


@computation_function()
def y_at_x(obj: SignalObj, p: AbscissaParam) -> TableResult:
    """
    Compute the smallest y-value at a given x-value for a signal object.

    Args:
        obj: The signal object containing x and y data.
        p: The parameter dataset for finding the ordinate.

    Returns:
         An object containing the y-value.
    """
    table = TableResultBuilder(f"y|x={p.x}")
    table.add(
        lambda xy: features.find_y_at_given_x_value(xy[0], xy[1], p.x),
        "y@x",
        "y = %g {.yunit}",
    )
    return table.compute(obj)


@computation_function()
def stats(obj: SignalObj) -> TableResult:
    """Compute statistics on a signal

    Args:
        obj: source signal

    Returns:
        Result properties object
    """
    table = TableResultBuilder(_("Signal statistics"))
    table.add(lambda xy: np.nanmin(xy[1]), "min", "min(y) = %g {.yunit}")
    table.add(lambda xy: np.nanmax(xy[1]), "max", "max(y) = %g {.yunit}")
    table.add(lambda xy: np.nanmean(xy[1]), "mean", "<y> = %g {.yunit}")
    table.add(lambda xy: np.nanmedian(xy[1]), "median", "median(y) = %g {.yunit}")
    table.add(lambda xy: np.nanstd(xy[1]), "std", "σ(y) = %g {.yunit}")
    table.add(lambda xy: np.nanmean(xy[1]) / np.nanstd(xy[1]), "snr", "<y>/σ(y) = %g")
    table.add(
        lambda xy: np.nanmax(xy[1]) - np.nanmin(xy[1]),
        "ptp",
        "peak-to-peak(y) = %g {.yunit}",
    )
    table.add(lambda xy: np.nansum(xy[1]), "sum", "Σ(y) = %g {.yunit}")
    table.add(lambda xy: spt.trapezoid(xy[1], xy[0]), "trapz", "∫ydx = %g {.yunit}")
    return table.compute(obj)


@computation_function()
def bandwidth_3db(obj: SignalObj) -> GeometryResult | None:
    """Compute bandwidth at -3 dB with
    :py:func:`sigima.tools.signal.misc.bandwidth`

    .. note::

       The bandwidth is defined as the range of frequencies over which the signal
       maintains a certain level relative to its peak.

    .. warning::

        The signal is assumed to be smooth enough for the bandwidth calculation to be
        meaningful. If the signal contains excessive noise, multiple peaks, or is not
        sufficiently continuous, the computed bandwidth may not accurately represent the
        true -3dB range. It is recommended to preprocess the signal to ensure reliable
        results.

    Args:
        obj: Source signal.

    Returns:
        Result shape with bandwidth.
    """
    return compute_geometry_from_obj(
        "bandwidth", "segment", obj, features.bandwidth, -3.0
    )


class DynamicParam(gds.DataSet):
    """Parameters for dynamic range computation (ENOB, SNR, SINAD, THD, SFDR)"""

    full_scale = gds.FloatItem(_("Full scale"), default=0.16, min=0.0, unit="V")
    unit = gds.ChoiceItem(
        _("Unit"), PowerUnit, default=PowerUnit.DBC, help=_("Unit for SINAD")
    )
    nb_harm = gds.IntItem(
        _("Number of harmonics"),
        default=5,
        min=1,
        help=_("Number of harmonics to consider for THD"),
    )


@computation_function()
def dynamic_parameters(src: SignalObj, p: DynamicParam) -> TableResult:
    """Compute Dynamic parameters
    using the following functions:

    - Freq: :py:func:`sigima.tools.signal.dynamic.sinus_frequency`
    - ENOB: :py:func:`sigima.tools.signal.dynamic.enob`
    - SNR: :py:func:`sigima.tools.signal.dynamic.snr`
    - SINAD: :py:func:`sigima.tools.signal.dynamic.sinad`
    - THD: :py:func:`sigima.tools.signal.dynamic.thd`
    - SFDR: :py:func:`sigima.tools.signal.dynamic.sfdr`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result properties with ENOB, SNR, SINAD, THD, SFDR
    """
    dsfx = f" = %g {p.unit}"
    table = TableResultBuilder(_("Dynamic parameters"))
    table.add(lambda xy: dynamic.sinus_frequency(xy[0], xy[1]), "freq")
    table.add(
        lambda xy: dynamic.enob(xy[0], xy[1], p.full_scale), "enob", "ENOB = %.1f bits"
    )
    table.add(lambda xy: dynamic.snr(xy[0], xy[1], p.unit), "snr", "SNR" + dsfx)
    table.add(lambda xy: dynamic.sinad(xy[0], xy[1], p.unit), "sinad", "SINAD" + dsfx)
    table.add(
        lambda xy: dynamic.thd(xy[0], xy[1], p.full_scale, p.unit, p.nb_harm),
        "thd",
        "THD" + dsfx,
    )
    table.add(
        lambda xy: dynamic.sfdr(xy[0], xy[1], p.full_scale, p.unit),
        "sfdr",
        "SFDR" + dsfx,
    )
    return table.compute(src)


@computation_function()
def sampling_rate_period(obj: SignalObj) -> TableResult:
    """Compute sampling rate and period
    using the following functions:

    - fs: :py:func:`sigima.tools.signal.dynamic.sampling_rate`
    - T: :py:func:`sigima.tools.signal.dynamic.sampling_period`

    Args:
        obj: source signal

    Returns:
        Result properties with sampling rate and period
    """
    table = TableResultBuilder(_("Sampling rate and period"))
    table.add(lambda xy: dynamic.sampling_rate(xy[0]), "fs", "fs = %g")
    table.add(lambda xy: dynamic.sampling_period(xy[0]), "T", "T = %g {.xunit}")
    return table.compute(obj)


@computation_function()
def contrast(obj: SignalObj) -> TableResult:
    """Compute contrast with :py:func:`sigima.tools.signal.misc.contrast`"""
    table = TableResultBuilder(_("Contrast"))
    table.add(lambda xy: features.contrast(xy[1]), "contrast")
    return table.compute(obj)


@computation_function()
def x_at_minmax(obj: SignalObj) -> TableResult:
    """
    Compute the smallest argument at the minima and the smallest argument at the maxima.

    Args:
        obj: The signal object.

    Returns:
        An object containing the x-values at the minima and the maxima.
    """
    table = TableResultBuilder(_("X at min/max"))
    table.add(lambda xy: xy[0][np.argmin(xy[1])], "X@Ymin", "X@Ymin = %g {.xunit}")
    table.add(lambda xy: xy[0][np.argmax(xy[1])], "X@Ymax", "X@Ymax = %g {.xunit}")
    return table.compute(obj)
