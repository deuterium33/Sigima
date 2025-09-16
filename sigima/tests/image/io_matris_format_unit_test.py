# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for opening matris format files.
"""

import numpy as np

from sigima.io.image.formats import MatrisFileReader
from sigima.tests.helpers import check_array_result, get_test_fnames


def test_read_image_basic():
    """Basic test to read a simple matris image file"""
    path = get_test_fnames("matris/image.txt")[0]
    imgs = MatrisFileReader.read_images(path)
    assert len(imgs) == 1, f"Expected 1 image, got {len(imgs)}"
    arr = np.asarray(imgs[0].data)
    expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    check_array_result("test read image.txt", arr, expected)


def test_read_image_with_unit():
    """Test to read a matris image file with units in metadata"""
    path = get_test_fnames("matris/image_with_unit.txt")[0]
    imgs = MatrisFileReader.read_images(path)
    assert len(imgs) == 1, f"Expected 1 image, got {len(imgs)}"
    img = imgs[0]
    # units should come from metadata (X, Y, Z)
    check_array_result(
        "test read image_with_unit.txt",
        np.asarray(img.data),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )

    assert img.xunit == "mm", (
        f"matris file X unit not read correctly {img.xunit}  given but mm expected"
    )
    assert img.yunit == "nm", (
        f"matris file Y unit not read correctly {img.yunit} given but nm expected"
    )
    assert img.zunit == "A", (
        f"matris file Z unit not read correctly {img.zunit} given but A expected"
    )


def test_read_image_with_nan():
    """Test to read a matris image file with NaN values"""
    path = get_test_fnames("matris/image_with_nan.txt")[0]
    imgs = MatrisFileReader.read_images(path)
    assert len(imgs) == 1, f"Expected 1 image, got {len(imgs)}"
    arr = np.asarray(imgs[0].data)
    # expected NaN positions from the test file
    assert np.isnan(arr[0, 2]), "expected NaN at position (0,2), got {arr[0,2]}"
    assert np.isnan(arr[1, 0]), "expected NaN at position (1,0), got {arr[1,0]}"
    assert np.isnan(arr[1, 1]), "expected NaN at position (1,1), got {arr[1,1]}"
    # and a valid value
    assert arr[0, 0] == 1, "expected 1 at position (0,0), got {arr[0,0]}"
    assert arr[1, 2] == 6, "expected 6 at position (1,2), got {arr[1,2]}"


def test_read_complex_image_and_error():
    """Test to read a matris complex image file with associated error image"""
    path = get_test_fnames("matris/complex_image.txt")[0]
    imgs = MatrisFileReader.read_images(path)
    # should return main image and error image
    assert len(imgs) == 2, f"Expected 2 images, got {len(imgs)}"
    img, img_err = imgs[0], imgs[1]
    # data should be complex
    assert np.iscomplexobj(np.asarray(img.data)), (
        f"expected complex data, got {np.asarray(img.data).dtype}"
    )
    assert np.iscomplexobj(np.asarray(img_err.data)), (
        f"expected complex data, got {np.asarray(img_err.data).dtype}"
    )
    # check first element values (from first data line)
    first_val = img.data[0, 0]
    expected = complex(3.678795e-01, 3.678795e-01)
    np.testing.assert_allclose(first_val, expected, rtol=1e-7, atol=1e-12)
    first_err = img_err.data[0, 0]
    expected_err = complex(1.839397e-01, -3.678795e-01)
    np.testing.assert_allclose(first_err, expected_err, rtol=1e-7, atol=1e-12)


if __name__ == "__main__":
    test_read_image_basic()
    test_read_image_with_unit()
    test_read_image_with_nan()
    test_read_complex_image_and_error()
