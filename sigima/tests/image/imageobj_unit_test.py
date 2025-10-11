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
from sigima.tests import guiutils
from sigima.tests.data import (
    create_annotated_image,
    create_test_image_with_metadata,
    iterate_image_creation,
)
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
        param.x0 = param.y0 = 3.0
        param.sigma = 5.0


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
    fi_list.append(("annotated_image", create_annotated_image()))
    return fi_list


def test_hdf5_image_io() -> None:
    """Test HDF5 I/O for image objects with uniform and non-uniform coordinates"""
    execenv.print(f"{test_hdf5_image_io.__doc__}:")
    with WorkdirRestoringTempDir() as tmpdir:
        for fname, orig_image in __get_filenames_and_images():
            if orig_image is None:
                execenv.print(f"  Skipping {fname} (not implemented)")
                continue

            # Test Case 1: Original image with uniform coordinates (default)
            filename = osp.join(tmpdir, f"test_{osp.basename(fname)}_uniform.h5ima")
            sigima.io.write_image(filename, orig_image)
            execenv.print(f"  Saved {filename} (uniform coords)")

            # Read back
            fetch_image = sigima.io.read_image(filename)
            execenv.print(f"  Read {filename}")

            # Verify data
            data = fetch_image.data
            orig_data = orig_image.data
            assert isinstance(data, np.ndarray)
            assert isinstance(orig_data, np.ndarray)
            assert data.shape == orig_data.shape
            assert data.dtype == orig_data.dtype
            assert fetch_image.annotations == orig_image.annotations
            assert np.allclose(data, orig_data, atol=0.0, equal_nan=True)
            assert compare_metadata(fetch_image.metadata, orig_image.metadata.copy())

            # Verify uniform coordinate attributes are preserved
            if orig_image.is_uniform_coords:
                assert fetch_image.is_uniform_coords
                assert fetch_image.dx == orig_image.dx
                assert fetch_image.dy == orig_image.dy
                assert fetch_image.x0 == orig_image.x0
                assert fetch_image.y0 == orig_image.y0
                execenv.print("    ✓ Uniform coordinates preserved")

            # Test Case 2: Same image with non-uniform coordinates
            # Create a modified version with non-uniform coordinates
            nonuniform_image = sigima.objects.create_image(
                title=orig_image.title + " (non-uniform)",
                data=orig_image.data.copy(),
                metadata=orig_image.metadata.copy(),
                units=(orig_image.xunit, orig_image.yunit, orig_image.zunit),
                labels=(orig_image.xlabel, orig_image.ylabel, orig_image.zlabel),
            )
            # Set non-uniform coordinates
            ny, nx = nonuniform_image.data.shape
            xcoords = np.linspace(0, 1, nx)
            ycoords = np.linspace(0, 1, ny) ** 2  # Quadratic spacing
            nonuniform_image.set_coords(xcoords=xcoords, ycoords=ycoords)

            # Save non-uniform version
            filename_nu = osp.join(
                tmpdir, f"test_{osp.basename(fname)}_nonuniform.h5ima"
            )
            sigima.io.write_image(filename_nu, nonuniform_image)
            execenv.print(f"  Saved {filename_nu} (non-uniform coords)")

            # Read back
            fetch_image_nu = sigima.io.read_image(filename_nu)
            execenv.print(f"  Read {filename_nu}")

            # Verify data
            assert np.allclose(
                fetch_image_nu.data, nonuniform_image.data, atol=0.0, equal_nan=True
            )

            # Verify non-uniform coordinate attributes are preserved
            assert not fetch_image_nu.is_uniform_coords
            assert np.array_equal(fetch_image_nu.xcoords, xcoords)
            assert np.array_equal(fetch_image_nu.ycoords, ycoords)
            execenv.print("    ✓ Non-uniform coordinates preserved")

    execenv.print(f"{test_hdf5_image_io.__doc__}: OK")


@pytest.mark.gui
def test_image_parameters_interactive() -> None:
    """Test interactive creation of image parameters"""
    execenv.print(f"{test_image_parameters_interactive.__doc__}:")
    with guiutils.lazy_qt_app_context(force=True):
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

    # 1. Create image with all parameters, and uniform coordinates
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
    dx, dy, x0, y0 = 0.1, 0.2, 50.0, 100.0
    image.set_uniform_coords(dx, dy, x0=x0, y0=y0)
    assert image.is_uniform_coords
    assert image.dx == dx
    assert image.dy == dy
    assert image.x0 == x0
    assert image.y0 == y0

    guiutils.view_images_if_gui(image, title=title)

    # 2. Create image with non-uniform coordinates
    xcoords = np.linspace(0, 1, 10)
    ycoords = np.linspace(0, 1, 10) ** 2
    image.set_coords(xcoords=xcoords, ycoords=ycoords)
    assert not image.is_uniform_coords
    assert np.array_equal(image.xcoords, xcoords)
    assert np.array_equal(image.ycoords, ycoords)

    guiutils.view_images_if_gui(image, title=title + " (non-uniform coords)")

    # 3. Create image with only data
    image = sigima.objects.create_image("", data=data)
    assert isinstance(image, sigima.objects.ImageObj)
    assert np.array_equal(image.data, data)
    assert not image.metadata
    assert (image.xunit, image.yunit, image.zunit) == ("", "", "")
    assert (image.xlabel, image.ylabel, image.zlabel) == ("", "", "")

    execenv.print(f"{test_create_image.__doc__}: OK")


def test_create_image_from_param() -> None:
    """Test creation of an image object using `create_image_from_param` function"""
    execenv.print(f"{test_create_image_from_param.__doc__}:")

    # Test 1: Basic parameter with defaults
    param = sigima.objects.NewImageParam()
    param.title = "Test Image"
    param.height = 100
    param.width = 200
    param.dtype = sigima.objects.ImageDatatypes.UINT16

    image = sigima.objects.create_image_from_param(param)
    assert isinstance(image, sigima.objects.ImageObj)
    assert image.title == "Test Image"
    assert image.data is not None
    assert image.data.shape == (100, 200)
    assert image.data.dtype == np.uint16
    assert (image.data == 0).all()  # NewImageParam generates zeros by default

    # Test 2: Parameter with default values (no explicit setting)
    param_defaults = sigima.objects.NewImageParam()
    # Don't set any values, use defaults

    image_defaults = sigima.objects.create_image_from_param(param_defaults)
    assert isinstance(image_defaults, sigima.objects.ImageObj)
    assert image_defaults.data is not None
    assert image_defaults.data.shape == (1024, 1024)  # Default dimensions
    assert image_defaults.data.dtype == np.float64  # Default dtype from NewImageParam

    # Test 3: Different image types using create_image_parameters
    test_cases = [
        (sigima.objects.ImageTypes.ZEROS, sigima.objects.ImageDatatypes.UINT8),
        (
            sigima.objects.ImageTypes.UNIFORM_DISTRIBUTION,
            sigima.objects.ImageDatatypes.FLOAT32,
        ),
        (
            sigima.objects.ImageTypes.NORMAL_DISTRIBUTION,
            sigima.objects.ImageDatatypes.FLOAT64,
        ),
        (sigima.objects.ImageTypes.GAUSS, sigima.objects.ImageDatatypes.UINT16),
        (sigima.objects.ImageTypes.RAMP, sigima.objects.ImageDatatypes.FLOAT64),
    ]

    for img_type, dtype in test_cases:
        param_type = sigima.objects.create_image_parameters(
            img_type,
            title=f"Test {img_type.value}",
            height=50,
            width=60,
            idtype=dtype,
        )

        # Preprocess parameters for specific types
        preprocess_image_parameters(param_type)

        image_type = sigima.objects.create_image_from_param(param_type)
        assert isinstance(image_type, sigima.objects.ImageObj)
        assert image_type.data is not None
        assert image_type.data.shape == (50, 60)
        assert image_type.data.dtype == dtype.value

        # Validate image type-specific properties
        if img_type == sigima.objects.ImageTypes.ZEROS:
            assert (image_type.data == 0).all()
        elif img_type == sigima.objects.ImageTypes.UNIFORM_DISTRIBUTION:
            # Uniform distribution should have varying values
            assert not (image_type.data == image_type.data[0, 0]).all()
            assert np.isfinite(image_type.data).all()
        elif img_type == sigima.objects.ImageTypes.NORMAL_DISTRIBUTION:
            # Normal distribution should have reasonable values
            assert not (image_type.data == 0).all()
            assert np.isfinite(image_type.data).all()
        elif img_type == sigima.objects.ImageTypes.GAUSS:
            # 2D Gaussian should have non-zero values
            assert not (image_type.data == 0).all()
            assert np.isfinite(image_type.data).all()
        elif img_type == sigima.objects.ImageTypes.RAMP:
            # Ramp should have varying values
            assert not (image_type.data == image_type.data[0, 0]).all()
            assert np.isfinite(image_type.data).all()

    # Test 4: Gaussian parameters with specific values
    gauss_param = sigima.objects.Gauss2DParam()
    gauss_param.title = "Custom Gauss"
    gauss_param.height = 80
    gauss_param.width = 80
    gauss_param.dtype = sigima.objects.ImageDatatypes.FLOAT32

    gauss_image = sigima.objects.create_image_from_param(gauss_param)
    assert isinstance(gauss_image, sigima.objects.ImageObj)
    assert gauss_image.title == "Custom Gauss"
    assert gauss_image.data.shape == (80, 80)
    assert gauss_image.data.dtype == np.float32
    # Center should have highest value for Gaussian
    center_val = gauss_image.data[40, 40]
    corner_val = gauss_image.data[0, 0]
    assert center_val > corner_val

    # Test 5: Ramp parameters with specific values
    ramp_param = sigima.objects.Ramp2DParam()
    ramp_param.title = "Custom Ramp"
    ramp_param.height = 60
    ramp_param.width = 40
    ramp_param.dtype = sigima.objects.ImageDatatypes.FLOAT64

    ramp_image = sigima.objects.create_image_from_param(ramp_param)
    assert isinstance(ramp_image, sigima.objects.ImageObj)
    assert ramp_image.title == "Custom Ramp"
    assert ramp_image.data.shape == (60, 40)
    assert ramp_image.data.dtype == np.float64
    # Ramp should have different values at different positions
    assert ramp_image.data[0, 0] != ramp_image.data[-1, -1]

    execenv.print(f"{test_create_image_from_param.__doc__}: OK")


if __name__ == "__main__":
    guiutils.enable_gui()
    test_create_image()
    test_image_parameters_interactive()
    test_all_image_types()
    test_hdf5_image_io()
    test_create_image_from_param()
