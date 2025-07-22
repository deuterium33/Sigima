# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests around the `ImageObj` class and its creation from parameters.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import os.path as osp
from collections.abc import Generator

import numpy as np
import pytest

import sigima.io
import sigima.objects
from sigima.io.image import ImageIORegistry
from sigima.tests.data import create_test_image_with_metadata
from sigima.tests.env import execenv
from sigima.tests.helpers import (
    WorkdirRestoringTempDir,
    check_scalar_result,
    compare_metadata,
    read_test_objects,
)


def iterate_image_creation(
    data_size: int = 500, non_zero: bool = False, verbose: bool = True
) -> Generator[sigima.objects.ImageObj, None, None]:
    """Iterate over all possible images created from parameters"""
    if verbose:
        execenv.print(
            f"  Iterating over {len(sigima.objects.ImageTypes)} image types "
            f"(size={data_size}, non_zero={non_zero}):"
        )
    for itype in sigima.objects.ImageTypes:
        if non_zero and itype in (
            sigima.objects.ImageTypes.EMPTY,
            sigima.objects.ImageTypes.ZEROS,
        ):
            continue
        if verbose:
            execenv.print(f"    {itype.value}")
        yield from _iterate_image_datatypes(itype, data_size, verbose)


def _iterate_image_datatypes(
    itype: sigima.objects.ImageTypes, data_size: int, verbose: bool
) -> Generator[sigima.objects.ImageObj | None, None, None]:
    for dtype in sigima.objects.ImageDatatypes:
        if verbose:
            execenv.print(f"      {dtype.value}")
        param_class = sigima.objects.get_image_parameters_class(itype)
        param = param_class.create(dtype=dtype, width=data_size, height=data_size)
        if itype == sigima.objects.ImageTypes.RAMP:
            if dtype is not sigima.objects.ImageDatatypes.FLOAT64:
                continue  # Testing only float64 for ramp
            assert isinstance(param, sigima.objects.Ramp2DParam)
            param.a = 1.0
            param.b = 2.0
            param.c = 3.0
            param.xmin = -1.0
            param.xmax = 2.0
            param.ymin = -5.0
            param.ymax = 4.0
        elif itype == sigima.objects.ImageTypes.GAUSS:
            assert isinstance(param, sigima.objects.Gauss2DParam)
            param.x0 = param.y0 = 3
            param.sigma = 5
        elif itype == sigima.objects.ImageTypes.UNIFORMRANDOM:
            assert isinstance(param, sigima.objects.UniformRandom2DParam)
            param.set_from_datatype(dtype.value)
        elif itype == sigima.objects.ImageTypes.NORMALRANDOM:
            assert isinstance(param, sigima.objects.NormalRandom2DParam)
            param.set_from_datatype(dtype.value)
        image = sigima.objects.create_image_from_param(param)
        if image is not None:
            _test_image_data(itype, image)
        yield image


def _test_image_data(
    itype: sigima.objects.ImageTypes, image: sigima.objects.ImageObj
) -> None:
    """Tests the data of an image based on its type.

    Args:
        itype: The type of the image.
        image: The image object containing the data to be tested.

    Raises:
        AssertionError: If the image data does not match the expected values
         for the given image type.
    """
    if itype == sigima.objects.ImageTypes.ZEROS:
        assert (image.data == 0).all()
    elif itype == sigima.objects.ImageTypes.RAMP:
        assert image.data is not None
        check_scalar_result("Top-left corner", image.data[0][0], -8.0)
        check_scalar_result("Top-right corner", image.data[0][-1], -5.0)
        check_scalar_result("Bottom-left corner", image.data[-1][0], 10.0)
        check_scalar_result("Bottom-right", image.data[-1][-1], 13.0)
    else:
        assert image.data is not None


def test_all_image_types() -> None:
    """Testing image creation from parameters"""
    execenv.print(f"{test_all_image_types.__doc__}:")
    for image in iterate_image_creation():
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
            param = sigima.objects.get_image_parameters_class(itype)()
            if param.edit():
                execenv.print(f"  Edited parameters for {itype.value}:")
                execenv.print(f"    {param}")
            else:
                execenv.print(f"  Skipped editing parameters for {itype.value}")
    execenv.print(f"{test_image_parameters_interactive.__doc__}: OK")


if __name__ == "__main__":
    test_image_parameters_interactive()
    test_all_image_types()
    test_hdf5_image_io()
