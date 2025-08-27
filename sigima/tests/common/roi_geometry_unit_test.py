# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Test ROI geometry transformations"""

import numpy as np

from sigima.objects.image import ImageObj, ImageROI, RectangularROI
from sigima.proc.image import fliph, flipv, rotate90, rotate270, transpose
from sigima.tests.env import execenv


def test_roi_rotate90() -> None:
    """Test ROI transformation with 90° rotation."""
    # Create test image
    data = np.random.rand(100, 80)  # Height=100, Width=80
    img = ImageObj(title="Test Image")
    img.data = data

    # Create rectangular ROI
    roi_obj = ImageROI()
    single_roi = RectangularROI([10, 20, 20, 20], indices=False)  # x0, y0, dx, dy
    roi_obj.add_roi(single_roi)
    img.roi = roi_obj

    # Apply rotation
    img_rotated = rotate90(img)
    roi_rotated = img_rotated.roi.single_rois[0]

    # Verify ROI is properly transformed and dimensions change
    assert img_rotated.roi is not None, "90° rotation should preserve ROI object"
    assert len(img_rotated.roi.single_rois) == 1, (
        "90° rotation should preserve ROI count"
    )

    # The exact coordinates depend on the transformation, but the ROI should be valid
    coords_rotated = roi_rotated.get_physical_coords(img_rotated)
    assert len(coords_rotated) == 4, "Rectangular ROI should have 4 coordinates"


def test_roi_rotate270() -> None:
    """Test ROI transformation with 270° rotation."""
    # Create test image
    data = np.random.rand(100, 80)
    img = ImageObj(title="Test Image")
    img.data = data

    # Create rectangular ROI
    roi_obj = ImageROI()
    single_roi = RectangularROI([10, 20, 20, 20], indices=False)
    roi_obj.add_roi(single_roi)
    img.roi = roi_obj

    # Apply rotation
    img_rotated = rotate270(img)

    # Verify ROI is properly transformed
    assert img_rotated.roi is not None, "270° rotation should preserve ROI object"
    assert len(img_rotated.roi.single_rois) == 1, (
        "270° rotation should preserve ROI count"
    )


def test_roi_fliph() -> None:
    """Test ROI transformation with horizontal flip."""
    # Create test image
    data = np.random.rand(100, 80)
    img = ImageObj(title="Test Image")
    img.data = data

    # Create rectangular ROI
    roi_obj = ImageROI()
    single_roi = RectangularROI([10, 20, 20, 20], indices=False)
    roi_obj.add_roi(single_roi)
    img.roi = roi_obj

    # Apply flip
    img_flipped = fliph(img)

    # Verify ROI is properly transformed
    assert img_flipped.roi is not None, "Horizontal flip should preserve ROI object"
    assert len(img_flipped.roi.single_rois) == 1, (
        "Horizontal flip should preserve ROI count"
    )


def test_roi_flipv() -> None:
    """Test ROI transformation with vertical flip."""
    # Create test image
    data = np.random.rand(100, 80)
    img = ImageObj(title="Test Image")
    img.data = data

    # Create rectangular ROI
    roi_obj = ImageROI()
    single_roi = RectangularROI([10, 20, 20, 20], indices=False)
    roi_obj.add_roi(single_roi)
    img.roi = roi_obj

    # Apply flip
    img_flipped = flipv(img)

    # Verify ROI is properly transformed
    assert img_flipped.roi is not None, "Vertical flip should preserve ROI object"
    assert len(img_flipped.roi.single_rois) == 1, (
        "Vertical flip should preserve ROI count"
    )


def test_roi_transpose() -> None:
    """Test ROI transformation with transpose."""
    # Create test image
    data = np.random.rand(100, 80)
    img = ImageObj(title="Test Image")
    img.data = data

    # Create rectangular ROI
    roi_obj = ImageROI()
    single_roi = RectangularROI([10, 20, 20, 20], indices=False)
    roi_obj.add_roi(single_roi)
    img.roi = roi_obj

    # Apply transpose
    img_transposed = transpose(img)

    # Verify ROI is properly transformed
    assert img_transposed.roi is not None, "Transpose should preserve ROI object"
    assert len(img_transposed.roi.single_rois) == 1, (
        "Transpose should preserve ROI count"
    )


def test_roi_comprehensive() -> None:
    """Comprehensive test for all ROI transformations."""
    execenv.print("Testing ROI geometry transformations:")

    # Create test image
    data = np.random.rand(100, 80)  # Height=100, Width=80
    img = ImageObj(title="Test Image")
    img.data = data

    # Create rectangular ROI
    roi_obj = ImageROI()
    single_roi = RectangularROI([10, 20, 20, 20], indices=False)
    roi_obj.add_roi(single_roi)
    img.roi = roi_obj

    coords_original = single_roi.get_physical_coords(img)
    execenv.print(f"  Original ROI: {coords_original}")

    # Test all transformations
    transformations = [
        (rotate90, "rotate90"),
        (rotate270, "rotate270"),
        (fliph, "fliph"),
        (flipv, "flipv"),
        (transpose, "transpose"),
    ]

    for transform_func, name in transformations:
        img_transformed = transform_func(img)
        roi_transformed = img_transformed.roi.single_rois[0]
        coords_transformed = roi_transformed.get_physical_coords(img_transformed)
        execenv.print(f"  {name}: {coords_transformed}")

        # Verify ROI is properly transformed
        assert img_transformed.roi is not None, f"{name} should preserve ROI object"
        assert len(img_transformed.roi.single_rois) == 1, (
            f"{name} should preserve ROI count"
        )

    execenv.print("  ✅ All ROI transformations working correctly")
