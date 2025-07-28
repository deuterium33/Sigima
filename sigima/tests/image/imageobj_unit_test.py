# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests around the `ImageObj` class and its creation from parameters.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import os.path as osp

import numpy as np
import pytest

import sigima.io
import sigima.objects
from sigima.io.image import ImageIORegistry
from sigima.objects.image import Gauss2DParam, Ramp2DParam
from sigima.tests.data import create_test_image_with_metadata, iterate_image_creation
from sigima.tests.env import execenv
from sigima.tests.helpers import (
    WorkdirRestoringTempDir,
    check_scalar_result,
    compare_metadata,
    read_test_objects,
)


def preprocess_image_parameters(param: sigima.objects.NewImageParam) -> None:
    """Preprocess image parameters before creating the image.

    Args:
        param: The image parameters to preprocess.
    """
    if isinstance(param, Ramp2DParam):
        param.a = 1.0
        param.b = 2.0
        param.c = 3.0
        param.xmin = -1.0
        param.xmax = 2.0
        param.ymin = -5.0
        param.ymax = 4.0
    elif isinstance(param, Gauss2DParam):
        param.x0 = param.y0 = 3
        param.sigma = 5


def postprocess_image_object(
    obj: sigima.objects.ImageObj, itype: sigima.objects.ImageTypes
) -> None:
    """Postprocess the image object after creation.

    Args:
        obj: The image object to postprocess.
        itype: The type of the image.
    """
    if itype == sigima.objects.ImageTypes.ZEROS:
        assert (obj.data == 0).all()
    elif itype == sigima.objects.ImageTypes.RAMP:
        assert obj.data is not None
        check_scalar_result("Top-left corner", obj.data[0][0], -8.0)
        check_scalar_result("Top-right corner", obj.data[0][-1], -5.0)
        check_scalar_result("Bottom-left corner", obj.data[-1][0], 10.0)
        check_scalar_result("Bottom-right", obj.data[-1][-1], 13.0)
    else:
        assert obj.data is not None


def test_all_image_types() -> None:
    """Testing image creation from parameters"""
    execenv.print(f"{test_all_image_types.__doc__}:")
    for image in iterate_image_creation(
        preproc=preprocess_image_parameters,
        postproc=postprocess_image_object,
    ):
        assert image.data is not None
    execenv.print(f"{test_all_image_types.__doc__}: OK")


def __get_filenames_and_images() -> list[tuple[str, sigima.objects.ImageObj]]:
    """Get test filenames and images from the registry"""
    fi_list = [
        (fname, obj)
        for fname, obj in read_test_objects(ImageIORegistry)
        if obj is not None
    ]
    fi_list.append(("test_image_with_metadata", create_test_image_with_metadata()))
    return fi_list


def test_hdf5_image_io() -> None:
    """Test HDF5 I/O for image objects"""
    execenv.print(f"{test_hdf5_image_io.__doc__}:")
    with WorkdirRestoringTempDir() as tmpdir:
        for fname, orig_image in __get_filenames_and_images():
            if orig_image is None:
                execenv.print(f"  Skipping {fname} (not implemented)")
                continue
            # Save to HDF5
            filename = osp.join(tmpdir, f"test_{osp.basename(fname)}.h5")
            sigima.io.write_image(filename, orig_image)
            execenv.print(f"  Saved {filename}")
            # Read back
            fetch_image = sigima.io.read_image(filename)
            execenv.print(f"  Read {filename}")
            data = fetch_image.data
            orig_data = orig_image.data
            assert isinstance(data, np.ndarray)
            assert isinstance(orig_data, np.ndarray)
            assert data.shape == orig_data.shape
            assert data.dtype == orig_data.dtype
            assert np.isclose(data, orig_data, atol=0.0).all()
            assert compare_metadata(fetch_image.metadata, orig_image.metadata.copy())
    execenv.print(f"{test_hdf5_image_io.__doc__}: OK")


@pytest.mark.gui
def test_image_parameters_interactive() -> None:
    """Test interactive creation of image parameters"""
    execenv.print(f"{test_image_parameters_interactive.__doc__}:")
    # pylint: disable=import-outside-toplevel
    from guidata.qthelpers import qt_app_context

    with qt_app_context():
        for itype in sigima.objects.ImageTypes:
            param = sigima.objects.create_image_parameters(itype)
            if param.edit():
                execenv.print(f"  Edited parameters for {itype.value}:")
                execenv.print(f"    {param}")
            else:
                execenv.print(f"  Skipped editing parameters for {itype.value}")
    execenv.print(f"{test_image_parameters_interactive.__doc__}: OK")


def test_create_image() -> None:
    """Test creation of an image object using `create_image` function"""
    execenv.print(f"{test_create_image.__doc__}:")
    # pylint: disable=import-outside-toplevel

    # Test all combinations of input parameters
    title = "Some Image"
    data = np.random.rand(10, 10)
    metadata = {"key": "value"}
    units = ("x unit", "y unit", "z unit")
    labels = ("x label", "y label", "z label")

    # 1. Create image with all parameters
    image = sigima.objects.create_image(
        title=title,
        data=data,
        metadata=metadata,
        units=units,
        labels=labels,
    )
    assert isinstance(image, sigima.objects.ImageObj)
    assert image.title == title
    assert image.data is data  # Data should be the same object (not a copy)
    assert image.metadata == metadata
    assert (image.xunit, image.yunit, image.zunit) == units
    assert (image.xlabel, image.ylabel, image.zlabel) == labels

    # 2. Create image with only data
    image = sigima.objects.create_image("", data=data)
    assert isinstance(image, sigima.objects.ImageObj)
    assert np.array_equal(image.data, data)
    assert image.metadata == {}
    assert (image.xunit, image.yunit, image.zunit) == ("", "", "")
    assert (image.xlabel, image.ylabel, image.zlabel) == ("", "", "")

    execenv.print(f"{test_create_image.__doc__}: OK")


if __name__ == "__main__":
    test_image_parameters_interactive()
    test_all_image_types()
    test_hdf5_image_io()
    test_create_image()
