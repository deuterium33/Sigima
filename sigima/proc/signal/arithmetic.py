# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Arithmetic operations on signals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

import warnings

import numpy as np

from sigima.enums import MathOperator
from sigima.objects import SignalObj
from sigima.proc.base import (
    ArithmeticParam,
    ConstantParam,
    dst_1_to_1,
    dst_2_to_1,
    dst_n_to_1,
)
from sigima.proc.decorator import computation_function
from sigima.proc.signal.base import (
    is_uncertainty_data_available,
    restore_data_outside_roi,
    signals_dy_to_array,
    signals_y_to_array,
)
from sigima.proc.signal.mathops import inverse


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
        dst.dy = np.sqrt(np.sum(dy_array**2, axis=0)) / len(src_list)
    restore_data_outside_roi(dst, src_list[0])
    return dst


@computation_function()
def standard_deviation(src_list: list[SignalObj]) -> SignalObj:
    """Compute the element-wise standard deviation of multiple signals.

    The first signal in the list defines the "base" signal. All other signals are
    used to compute the element-wise standard deviation with the base signal.

    .. note::

        If all signals share the same region of interest (ROI), the standard deviation
        is computed only within the ROI.

    .. warning::

        It is assumed that all signals have the same size and x-coordinates.

    Args:
        src_list: List of source signals.

    Returns:
        Signal object representing the standard deviation of the source signals.
    """
    dst = dst_n_to_1(src_list, "𝜎")  # `dst` data is initialized to `src_list[0]` data
    dst.y = np.std(signals_y_to_array(src_list), axis=0, ddof=0)
    if is_uncertainty_data_available(src_list):
        dst.dy = dst.y / np.sqrt(2 * (len(src_list) - 1))
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
        dst = addition_constant(product_constant(product([src1, inverse(src2)]), a), b)
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
