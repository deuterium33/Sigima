# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests around the `ImageObj` class and its creation from parameters.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

from collections.abc import Generator

import sigima.objects
from sigima.tests.env import execenv


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
        base_param = sigima.objects.NewImageParam.create(
            itype=itype, dtype=dtype, width=data_size, height=data_size
        )
        extra_param = _get_additional_param(itype, dtype)
        image = sigima.objects.create_image_from_param(
            base_param, extra_param=extra_param
        )
        if image is not None:
            _test_image_data(itype, image)
        yield image


def _get_additional_param(
    itype: sigima.objects.ImageTypes, dtype: sigima.objects.ImageDatatypes
) -> (
    sigima.objects.Gauss2DParam
    | sigima.objects.UniformRandomParam
    | sigima.objects.NormalRandomParam
    | None
):
    if itype == sigima.objects.ImageTypes.GAUSS:
        addparam = sigima.objects.Gauss2DParam()
        addparam.x0 = addparam.y0 = 3
        addparam.sigma = 5
    elif itype == sigima.objects.ImageTypes.UNIFORMRANDOM:
        addparam = sigima.objects.UniformRandomParam()
        addparam.set_from_datatype(dtype.value)
    elif itype == sigima.objects.ImageTypes.NORMALRANDOM:
        addparam = sigima.objects.NormalRandomParam()
        addparam.set_from_datatype(dtype.value)
    else:
        addparam = None
    return addparam


def _test_image_data(
    itype: sigima.objects.ImageTypes, image: sigima.objects.ImageObj
) -> None:
    """
    Tests the data of an image based on its type.

    Args:
        itype: The type of the image.
        image: The image object containing the data to be tested.

    Raises:
        AssertionError: If the image data does not match the expected values
         for the given image type.
    """
    if itype == sigima.objects.ImageTypes.ZEROS:
        assert (image.data == 0).all()
    else:
        assert image.data is not None


def test_all_image_types() -> None:
    """Testing image creation from parameters"""
    execenv.print(f"{test_all_image_types.__doc__}:")
    for image in iterate_image_creation():
        assert image.data is not None
    execenv.print(f"{test_all_image_types.__doc__}: OK")


if __name__ == "__main__":
    test_all_image_types()
