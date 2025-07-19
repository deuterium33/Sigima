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

from sigima.objects import ImageObj, SignalObj, create_image, create_signal
from sigima.proc.base import dst_1_to_1
from sigima.proc.decorator import (
    computation_function,
    get_computation_metadata,
    is_computation_function,
)
from sigima.tests.helpers import check_array_result


class DummySignalParam(gds.DataSet):
    """Dummy DataSet for testing purposes"""

    a = gds.FloatItem("X value", default=1.0)
    b = gds.FloatItem("Y value", default=5.0)


SCF_NAME = "dummy_signal_func"
SCF_DESCRIPTION = "A dummy signal function"


@computation_function(name=SCF_NAME, description=SCF_DESCRIPTION)
def dummy_signal_func(src: SignalObj, param: DummySignalParam) -> SignalObj:
    """A dummy function that adds two parameters from a DataSet"""
    dst = dst_1_to_1(src, SCF_NAME, f"x={param.a:.3f}, y={param.b:.3f}")
    dst.y = src.y + src.x * param.a + param.b
    return dst


class DummyImageParam(gds.DataSet):
    """Dummy DataSet for testing purposes"""

    alpha = gds.FloatItem("Alpha value", default=0.5)


ICF_NAME = "dummy_image_func"
ICF_DESCRIPTION = "A dummy image function"


@computation_function(name=ICF_NAME, description=ICF_DESCRIPTION)
def dummy_image_func(src: ImageObj, param: DummyImageParam) -> ImageObj:
    """A dummy function that applies a simple operation based on a DataSet parameter"""
    dst = dst_1_to_1(src, ICF_NAME, f"sigma={param.alpha:.3f}")
    dst.data = src.data * param.alpha  # Simplified operation for testing
    return dst


def test_signal_computation_function_decorator() -> None:
    """Test the computation function decorator for signals"""
    # Check if the function is marked as a computation function
    assert is_computation_function(dummy_signal_func)

    # Check if the metadata is correctly set
    metadata = get_computation_metadata(dummy_signal_func)
    assert metadata.name == SCF_NAME
    assert metadata.description == SCF_DESCRIPTION

    # Call the function with a SignalObj and DummySignalParam
    x = np.linspace(0, 10, 100)
    orig = create_signal("test_signal", x=x, y=x)
    p = DummySignalParam.create(a=3.0, b=4.0)
    res = dummy_signal_func(orig, p)

    # Check the result
    check_array_result("Signal x", res.x, orig.x)
    check_array_result("Signal y", res.y, orig.y + orig.x * p.a + p.b)


def test_image_computation_function_decorator() -> None:
    """Test the computation function decorator for images"""
    # Check if the function is marked as a computation function
    assert is_computation_function(dummy_image_func)

    # Check if the metadata is correctly set
    metadata = get_computation_metadata(dummy_image_func)
    assert metadata.name == ICF_NAME
    assert metadata.description == ICF_DESCRIPTION

    # Call the function with an ImageObj and DummyImageParam
    orig = create_image("test_image", data=np.random.rand(64, 64))
    p = DummyImageParam.create(alpha=0.8)
    res = dummy_image_func(orig, p)

    # Check the result
    check_array_result("Image data", res.data, orig.data * p.alpha)


if __name__ == "__main__":
    test_signal_computation_function_decorator()
    test_image_computation_function_decorator()
