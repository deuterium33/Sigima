# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Testing the decorator for computation functions.

This test checks:
  - The decorator can be applied to a function
  - The function can be called with and without DataSet parameters
  - The metadata is correctly set and can be introspected
"""

from __future__ import annotations

import guidata.dataset as gds
import numpy as np
import pytest

from sigima.objects import ImageObj, SignalObj, create_image, create_signal
from sigima.proc.base import dst_1_to_1
from sigima.proc.decorator import (
    computation_function,
    get_computation_metadata,
    is_computation_function,
)
from sigima.tests.helpers import check_array_result


def this_is_not_a_computation_function() -> None:
    """A dummy function that is not a computation function."""


def test_non_computation_function_marker() -> None:
    """Test that a non-computation function is not marked as such."""
    assert not is_computation_function(this_is_not_a_computation_function)
    with pytest.raises(ValueError):
        get_computation_metadata(this_is_not_a_computation_function)


class DummySignalParam(gds.DataSet):
    """Dummy DataSet for testing purposes"""

    a = gds.FloatItem("X value", default=1.0)
    b = gds.FloatItem("Y value", default=5.0)
    methods = (("linear", "Linear"), ("quadratic", "Quadratic"))
    method = gds.ChoiceItem("Method", choices=methods, default="linear")


SCF_NAME = "dummy_signal_func"
SCF_DESCRIPTION = "A dummy signal function"


@computation_function(name=SCF_NAME, description=SCF_DESCRIPTION)
def dummy_signal_func(src: SignalObj, p: DummySignalParam) -> SignalObj:
    """A dummy function that adds two parameters from a DataSet.

    Args:
        src: The source SignalObj.
        param: The parameters from the DummySignalParam DataSet.

    Returns:
        The signal with the operation applied.
    """
    dst = dst_1_to_1(src, SCF_NAME, f"x={p.a:.3f}, y={p.b:.3f}")
    if p.method == "linear":
        dst.y = src.y + src.x * p.a + p.b
    else:  # Quadratic method
        dst.y = src.y + src.x**2 * p.a + p.b
    return dst


class DummyImageParam(gds.DataSet):
    """Dummy DataSet for testing purposes"""

    alpha = gds.FloatItem("Alpha value", default=0.5)


ICF_NAME = "dummy_image_func"
ICF_DESCRIPTION = "A dummy image function"


@computation_function(name=ICF_NAME, description=ICF_DESCRIPTION)
def dummy_image_func(src: ImageObj, param: DummyImageParam) -> ImageObj:
    """A dummy function that applies a simple operation based on a DataSet parameter.

    Args:
        src: The source ImageObj.
        param: The parameters from the DummyImageParam DataSet.

    Returns:
        The image with the operation applied.
    """
    dst = dst_1_to_1(src, ICF_NAME, f"sigma={param.alpha:.3f}")
    dst.data = src.data * param.alpha  # Simplified operation for testing
    return dst


def test_signal_decorator_marker() -> None:
    """Test the computation function decorator marker for signals"""
    # Check if the function is marked as a computation function
    assert is_computation_function(dummy_signal_func)


def test_signal_decorator_metadata() -> None:
    """Test the computation function decorator metadata for signals"""
    # Check if the metadata is correctly set
    metadata = get_computation_metadata(dummy_signal_func)
    assert metadata.name == SCF_NAME
    assert metadata.description == SCF_DESCRIPTION


def test_signal_decorator_signature() -> None:
    """Test the computation function decorator signature for signals"""
    x = np.linspace(0, 10, 100)
    orig = create_signal("test_signal", x=x, y=x)

    # Call the function with a DataSet parameter
    p = DummySignalParam.create(a=3.0, b=4.0, method="quadratic")
    res_ds = dummy_signal_func(orig, p)
    name = "Signal[DataSet parameter]"
    check_array_result(f"{name} x", res_ds.x, orig.x)
    check_array_result(f"{name} y", res_ds.y, orig.y + orig.x**2 * 3.0 + 4.0)

    # Call the function with keyword arguments
    # pylint: disable=no-value-for-parameter
    res_kw = dummy_signal_func(orig, a=3.0, b=4.0)
    name = "Signal[keyword arguments]"
    check_array_result(f"{name} x", res_kw.x, orig.x)
    check_array_result(f"{name} y", res_kw.y, orig.y + orig.x * 3.0 + 4.0)

    # Call the function with both DataSet and keyword arguments
    # The DataSet should take precedence, kwargs for DataSet items are ignored
    res_both = dummy_signal_func(orig, p, a=100.0, b=200.0, method="linear")
    # The result should match the DataSet values, not the conflicting kwargs
    check_array_result(
        "Signal[DataSet param + kwargs: DataSet wins] x", res_both.x, orig.x
    )
    check_array_result(
        "Signal[DataSet param + kwargs: DataSet wins] y",
        res_both.y,
        orig.y + orig.x**2 * 3.0 + 4.0,
    )


def test_image_decorator_marker() -> None:
    """Test the computation function decorator marker for images"""
    # Check if the function is marked as a computation function
    assert is_computation_function(dummy_image_func)


def test_image_decorator_metadata() -> None:
    """Test the computation function decorator metadata for images"""
    # Check if the metadata is correctly set
    metadata = get_computation_metadata(dummy_image_func)
    assert metadata.name == ICF_NAME
    assert metadata.description == ICF_DESCRIPTION


def test_image_decorator_signature() -> None:
    """Test the computation function decorator signature for images"""
    orig = create_image("test_image", data=np.random.rand(64, 64))

    # Call the function with an ImageObj and DummyImageParam
    p = DummyImageParam.create(alpha=0.8)
    res_ds = dummy_image_func(orig, p)
    check_array_result("Image data", res_ds.data, orig.data * p.alpha)

    # Call the function with keyword arguments
    # pylint: disable=no-value-for-parameter
    res_kw = dummy_image_func(orig, alpha=0.8)
    check_array_result("Image data", res_kw.data, orig.data * 0.8)

    # Call the function with both DataSet and keyword arguments
    # The DataSet should take precedence, kwargs for DataSet items are ignored
    res_both = dummy_image_func(orig, p, alpha=0.4)
    check_array_result(
        "Image data [DataSet param + kwargs: DataSet wins]",
        res_both.data,
        orig.data * p.alpha,
    )


if __name__ == "__main__":
    test_signal_decorator_marker()
    test_signal_decorator_metadata()
    test_signal_decorator_signature()
    test_image_decorator_marker()
    test_image_decorator_metadata()
    test_image_decorator_signature()
