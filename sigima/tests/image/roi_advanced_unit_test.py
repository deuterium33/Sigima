# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Image ROI advanced unit tests"""

from __future__ import annotations

import numpy as np
import pytest
from skimage import draw

import sigima.objects
import sigima.params
import sigima.proc.image
from sigima.tests import guiutils
from sigima.tests.data import create_multigaussian_image
from sigima.tests.helpers import print_obj_data_dimensions


def test_image_roi_merge() -> None:
    """Test image ROI merge"""
    # Create an image object with a single ROI, and another one with another ROI.
    # Compute the average of the two objects, and check if the resulting object
    # has the expected ROI (i.e. the union of the original object's ROI).
    obj1 = create_multigaussian_image()
    obj2 = create_multigaussian_image()
    obj2.roi = sigima.objects.create_image_roi(
        "rectangle", [600, 800, 1000, 1200], inside=False
    )
    obj1.roi = sigima.objects.create_image_roi(
        "rectangle", [500, 750, 1000, 1250], inside=False
    )

    # Compute the average of the two objects
    obj3 = sigima.proc.image.average([obj1, obj2])
    assert obj3.roi is not None, "Merged object should have a ROI"
    assert len(obj3.roi) == 2, "Merged object should have two single ROIs"
    for single_roi in obj3.roi:
        assert single_roi.get_indices_coords(obj3) in (
            [500, 750, 1000, 1250],
            [600, 800, 1000, 1200],
        ), "Merged object should have the union of the original object's ROIs"


def test_image_roi_combine() -> None:
    """Test `ImageROI.combine_with` method"""
    coords1, coords2 = [600, 800, 1000, 1200], [500, 750, 1000, 1250]
    roi1 = sigima.objects.create_image_roi(
        "rectangle", coords1, indices=True, inside=False
    )
    roi2 = sigima.objects.create_image_roi(
        "rectangle", coords2, indices=True, inside=False
    )
    exp_combined = sigima.objects.create_image_roi(
        "rectangle", [coords1, coords2], indices=True, inside=False
    )
    # Check that combining two ROIs results in a new ROI with both coordinates:
    roi3 = roi1.combine_with(roi2)
    assert roi3 == exp_combined, "Combined ROI should match expected"
    # Check that combining again with the same ROI does not change it:
    roi3 = roi1.combine_with(roi2)
    assert roi3 == exp_combined, "Combining with the same ROI should not change it"
    # Check that combining with a signal ROI raises an error:
    with pytest.raises(
        TypeError, match=r"Cannot combine([\S ]*)ImageROI([\S ]*)SignalROI"
    ):
        roi1.combine_with(sigima.objects.create_signal_roi([50, 100], indices=True))


SIZE = 200

# Image ROIs:
IROI1 = [100, 100, 75, 100]  # Rectangle
IROI2 = [66, 100, 50]  # Circle
# Polygon (triangle, that is intentionally inside the rectangle, so that this ROI
# has no impact on the mask calculations in the tests)
IROI3 = [100, 100, 100, 150, 150, 133]


def __roi_str(obj: sigima.objects.ImageObj) -> str:
    """Return a string representation of a ImageROI object for context."""
    if obj.roi is None:
        return "None"
    if obj.roi.is_empty():
        return "Empty"
    return ", ".join(
        f"{single_roi.__class__.__name__}({single_roi.get_indices_coords(obj)})"
        for single_roi in obj.roi.single_rois
    )


def __create_test_roi() -> sigima.objects.ImageROI:
    """Create test ROI"""
    roi = sigima.objects.create_image_roi("rectangle", IROI1, inside=False)
    roi.add_roi(sigima.objects.create_image_roi("circle", IROI2, inside=False))
    roi.add_roi(sigima.objects.create_image_roi("polygon", IROI3, inside=False))
    return roi


def __create_test_image() -> sigima.objects.ImageObj:
    """Create test image"""
    param = sigima.objects.NewImageParam.create(height=SIZE, width=SIZE)
    ima = create_multigaussian_image(param)
    ima.data += 1  # Ensure that the image has non-zero values (for ROI check tests)
    return ima


def __test_processing_in_roi(src: sigima.objects.ImageObj) -> None:
    """Run image processing in ROI

    Args:
        src: Source image object (with or without ROI)
    """
    print_obj_data_dimensions(src)
    value = 1
    p = sigima.params.ConstantParam.create(value=value)
    dst = sigima.proc.image.addition_constant(src, p)
    orig = src.data
    new = dst.data
    context = f" [ROI: {__roi_str(src)}]"
    if src.roi is not None and not src.roi.is_empty():
        # A ROI has been set in the source image.
        assert np.all(
            new[IROI1[1] : IROI1[3] + IROI1[1], IROI1[0] : IROI1[2] + IROI1[0]]
            == orig[IROI1[1] : IROI1[3] + IROI1[1], IROI1[0] : IROI1[2] + IROI1[0]]
            + value
        ), f"Image ROI 1 data mismatch{context}"
        assert np.all(
            new[IROI2[1] : IROI1[1] + 1, IROI2[0] : IROI2[0] + 2 * IROI2[2]]
            == orig[IROI2[1] : IROI1[1] + 1, IROI2[0] : IROI2[0] + 2 * IROI2[2]] + value
        ), f"Image ROI 2 data mismatch{context}"
        first_col = min(IROI1[0], IROI2[0] - IROI2[2])
        first_row = min(IROI1[1], IROI2[1] - IROI2[2])
        last_col = max(IROI1[0] + IROI1[2], IROI2[0] + 2 * IROI2[2])
        last_row = max(IROI1[1] + IROI1[3], IROI2[1] + 2 * IROI2[2])
        assert np.all(
            new[:first_row, :first_col] == np.array(orig[:first_row, :first_col], float)
        ), f"Image before ROIs data mismatch{context}"
        assert np.all(new[:first_row, last_col:] == orig[:first_row, last_col:]), (
            f"Image after ROIs data mismatch{context}"
        )
        assert np.all(new[last_row:, :first_col] == orig[last_row:, :first_col]), (
            f"Image before ROIs data mismatch{context}"
        )
        assert np.all(new[last_row:, last_col:] == orig[last_row:, last_col:]), (
            f"Image after ROIs data mismatch{context}"
        )
    else:
        # No ROI has been set in the source image.
        assert np.all(new == orig + value), f"Image data mismatch{context}"


def test_image_roi_processing() -> None:
    """Test image ROI processing"""
    src = __create_test_image()
    base_roi = __create_test_roi()
    empty_roi = sigima.objects.ImageROI()
    for roi in (empty_roi, base_roi):
        src.roi = roi
        __test_processing_in_roi(src)


def test_empty_image_roi() -> None:
    """Test empty image ROI"""
    src = __create_test_image()
    empty_roi = sigima.objects.ImageROI()
    for roi in (None, empty_roi):
        src.roi = roi
        context = f" [ROI: {__roi_str(src)}]"
        assert src.roi is None or src.roi.is_empty(), (
            f"Source object ROI should be empty or None{context}"
        )
        if src.roi is not None:
            # No ROI has been set in the source image
            im1 = sigima.proc.image.extract_roi(src, src.roi.to_params(src))
            assert im1.data.shape == (0, 0), f"Extracted image should be empty{context}"


@pytest.mark.validation
def test_image_extract_rois() -> None:
    """Validation test for image ROI extraction into a single object"""
    src = __create_test_image()
    src.roi = __create_test_roi()
    context = f" [ROI: {__roi_str(src)}]"
    nzroi = f"Non-zero values expected in ROI{context}"
    zroi = f"Zero values expected outside ROI{context}"

    im1 = sigima.proc.image.extract_rois(src, src.roi.to_params(src))

    mask1 = np.zeros(shape=(SIZE, SIZE), dtype=bool)
    mask1[IROI1[1] : IROI1[1] + IROI1[3], IROI1[0] : IROI1[0] + IROI1[2]] = 1
    xc, yc, r = IROI2
    mask2 = np.zeros(shape=(SIZE, SIZE), dtype=bool)
    rr, cc = draw.disk((yc, xc), r)
    mask2[rr, cc] = 1
    mask = mask1 | mask2
    row_min = int(min(IROI1[1], IROI2[1] - r))
    col_min = int(min(IROI1[0], IROI2[0] - r))
    row_max = int(max(IROI1[1] + IROI1[3], IROI2[1] + r))
    col_max = int(max(IROI1[0] + IROI1[2], IROI2[0] + r))
    mask = mask[row_min:row_max, col_min:col_max]

    assert np.all(im1.data[mask] != 0), nzroi
    assert np.all(im1.data[~mask] == 0), zroi


@pytest.mark.validation
def test_image_extract_roi() -> None:
    """Validation test for image ROI extraction into multiple objects"""
    src = __create_test_image()
    src.roi = __create_test_roi()
    context = f" [ROI: {__roi_str(src)}]"
    nzroi = f"Non-zero values expected in ROI{context}"
    roisham = f"ROI shape mismatch{context}"

    images: list[sigima.objects.ImageObj] = []
    for index, single_roi in enumerate(src.roi):
        roiparam = single_roi.to_param(src, index)
        image = sigima.proc.image.extract_roi(src, roiparam)
        images.append(image)
    assert len(images) == 3, f"Three images expected{context}"
    im1, im2 = images[:2]  # pylint: disable=unbalanced-tuple-unpacking
    assert np.all(im1.data != 0), nzroi
    assert im1.data.shape == (IROI1[3], IROI1[2]), roisham
    assert np.all(im2.data != 0), nzroi
    assert im2.data.shape == (IROI2[2] * 2, IROI2[2] * 2), roisham
    mask2 = np.zeros(shape=im2.data.shape, dtype=bool)
    xc = yc = r = IROI2[2]  # Adjust for ROI origin
    rr, cc = draw.disk((yc, xc), r, shape=im2.data.shape)
    mask2[rr, cc] = 1
    assert np.all(im2.maskdata == ~mask2), f"Mask data mismatch{context}"


def test_roi_coordinates_validation() -> None:
    """Test ROI coordinates validation"""
    # Create a 20x20 Gaussian image
    param = sigima.objects.Gauss2DParam.create(a=10.0, height=20, width=20)
    src = sigima.objects.create_image_from_param(param)

    # Create ROI coordinates
    rect_coords = np.array([4.5, 4.5, 10.0, 10.0])
    circ_coords = np.array([9.5, 9.5, 5.0])
    poly_coords = np.array([5.1, 15.1, 14.7, 12.0, 12.5, 7.0, 5.2, 4.9])

    # Create ROIs
    rect_roi = sigima.objects.create_image_roi(
        "rectangle", rect_coords, title="rectangular", inside=False
    )
    circ_roi = sigima.objects.create_image_roi(
        "circle", circ_coords, title="circular", inside=False
    )
    poly_roi = sigima.objects.create_image_roi(
        "polygon", poly_coords, title="polygonal", inside=False
    )

    # Check that coordinates are correct
    assert np.all(rect_roi.get_single_roi(0).get_physical_coords(src) == rect_coords)
    assert np.all(circ_roi.get_single_roi(0).get_physical_coords(src) == circ_coords)
    assert np.all(poly_roi.get_single_roi(0).get_physical_coords(src) == poly_coords)

    # Check that extracted images have correct data
    for roi in (rect_roi, circ_roi, poly_roi):
        extracted = sigima.proc.image.extract_roi(src, roi.to_params(src)[0])
        assert np.all(extracted.data != 0), "Extracted image should have non-zero data"
        assert extracted.data.shape == (10, 10), "Extracted image shape mismatch"

    # Display the original image and the ROIs
    if guiutils.is_gui_enabled():
        images = [src]
        titles = ["Original Image"]
        for inside in (True, False):
            for roi in (rect_roi, circ_roi, poly_roi):
                src2 = src.copy()
                roi.get_single_roi(0).inside = inside
                src2.roi = roi
                images.append(src2)
                roi_title = roi.get_single_roi(0).title
                inside_str = "inside" if inside else "outside"
                titles.append(f"Image with {roi_title} ROI ({inside_str})")
        guiutils.view_images_side_by_side_if_gui(
            images, titles, rows=2, title="Image ROIs"
        )


def test_create_image_roi_inside_parameter() -> None:
    """Test create_image_roi function with inside parameter functionality"""
    # Test 1: Single ROI with inside=True
    roi1 = sigima.objects.create_image_roi("rectangle", [10, 20, 30, 40], inside=True)
    assert len(roi1) == 1, "Should create one ROI"
    assert roi1.single_rois[0].inside is True, "ROI should have inside=True"

    # Test 2: Single ROI with inside=True (default)
    roi2 = sigima.objects.create_image_roi("rectangle", [10, 20, 30, 40])
    assert roi2.single_rois[0].inside is True, "ROI should have default inside=True"

    # Test 3: Multiple ROIs with global inside parameter
    coords = [[10, 20, 30, 40], [50, 60, 70, 80]]
    roi3 = sigima.objects.create_image_roi("rectangle", coords, inside=True)
    assert len(roi3) == 2, "Should create two ROIs"
    assert all(single_roi.inside is True for single_roi in roi3.single_rois), (
        "All ROIs should have inside=True"
    )

    # Test 4: Multiple ROIs with individual inside parameters
    inside_params = [True, False]
    roi4 = sigima.objects.create_image_roi("rectangle", coords, inside=inside_params)
    assert len(roi4) == 2, "Should create two ROIs"
    assert roi4.single_rois[0].inside is True, "First ROI should have inside=True"
    assert roi4.single_rois[1].inside is False, "Second ROI should have inside=False"

    # Test 5: Circle ROIs with mixed inside parameters
    circle_coords = [[50, 50, 25], [150, 150, 30]]
    roi5 = sigima.objects.create_image_roi(
        "circle", circle_coords, inside=[False, True]
    )
    assert len(roi5) == 2, "Should create two circle ROIs"
    assert roi5.single_rois[0].inside is False, "First circle should have inside=False"
    assert roi5.single_rois[1].inside is True, "Second circle should have inside=True"

    # Test 6: Polygon ROIs with varying vertex counts and mixed inside parameters
    polygon_coords = [
        [0, 0, 10, 0, 5, 8],  # Triangle (3 vertices)
        [20, 20, 30, 20, 30, 30, 20, 30],  # Rectangle (4 vertices)
    ]
    roi6 = sigima.objects.create_image_roi(
        "polygon", polygon_coords, inside=[True, False]
    )
    assert len(roi6) == 2, "Should create two polygon ROIs"
    assert roi6.single_rois[0].inside is True, "Triangle should have inside=True"
    assert roi6.single_rois[1].inside is False, "Rectangle should have inside=False"


def test_create_image_roi_inside_parameter_errors() -> None:
    """Test error handling for inside parameter in create_image_roi"""
    # Test error when inside parameter count doesn't match ROI count
    coords = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
    inside_params = [True, False]  # Only 2 values for 3 ROIs

    with pytest.raises(
        ValueError,
        match=r"Number of inside values \(2\) must match number of ROIs \(3\)",
    ):
        sigima.objects.create_image_roi("rectangle", coords, inside=inside_params)

    # Test with too many inside values
    inside_params_too_many = [True, False, True, False]  # 4 values for 3 ROIs
    with pytest.raises(
        ValueError,
        match=r"Number of inside values \(4\) must match number of ROIs \(3\)",
    ):
        sigima.objects.create_image_roi(
            "rectangle", coords, inside=inside_params_too_many
        )


def test_roi_inside_mask_behavior() -> None:
    """Test that inside parameter affects mask generation correctly"""
    # Create a test image
    img = __create_test_image()

    # Test rectangle ROI with inside=True vs inside=False
    rect_coords = [75, 75, 50, 50]  # Rectangle that should be inside image bounds

    # ROI with inside=True (mask is True inside the rectangle)
    roi_inside = sigima.objects.create_image_roi("rectangle", rect_coords, inside=True)
    mask_inside = roi_inside.to_mask(img)

    # ROI with inside=False (mask is True outside the rectangle)
    roi_outside = sigima.objects.create_image_roi(
        "rectangle", rect_coords, inside=False
    )
    mask_outside = roi_outside.to_mask(img)

    # The two masks should be inverse of each other
    assert np.array_equal(mask_inside, ~mask_outside), (
        "Inside and outside masks should be inverse of each other"
    )

    # Check that inside mask has True values inside the rectangle region
    # For a rectangle [x0, y0, dx, dy], the region is [x0:x0+dx, y0:y0+dy]
    x0, y0, dx, dy = rect_coords
    expected_inside_region = np.zeros_like(img.data, dtype=bool)
    expected_inside_region[y0 : y0 + dy, x0 : x0 + dx] = True

    assert np.array_equal(mask_inside, expected_inside_region), (
        "Inside mask should match expected rectangular region"
    )


def test_roi_inside_serialization() -> None:
    """Test that inside parameter is preserved during serialization/deserialization"""
    # Create ROIs with mixed inside parameters
    coords = [[10, 20, 30, 40], [50, 60, 70, 80]]
    inside_params = [True, False]
    original_roi = sigima.objects.create_image_roi(
        "rectangle", coords, inside=inside_params
    )

    # Serialize to dictionary
    roi_dict = original_roi.to_dict()

    # Deserialize from dictionary
    restored_roi = sigima.objects.ImageROI.from_dict(roi_dict)

    # Check that inside parameters are preserved
    assert len(restored_roi) == len(original_roi), "ROI count should be preserved"
    for i in range(len(original_roi)):
        original_inside = original_roi.single_rois[i].inside
        restored_inside = restored_roi.single_rois[i].inside
        assert original_inside == restored_inside, (
            f"Inside parameter for ROI {i} should be preserved "
            f"(expected {original_inside}, got {restored_inside})"
        )


def test_roi_inside_parameter_conversion() -> None:
    """Test that inside parameter works correctly with parameter conversion"""
    img = __create_test_image()

    # Create ROI with inside=True
    roi = sigima.objects.create_image_roi("rectangle", [50, 50, 40, 40], inside=True)

    # Convert to parameters
    params = roi.to_params(img)
    assert len(params) == 1, "Should create one parameter"

    # Check that inside parameter is preserved in the parameter
    param = params[0]
    assert hasattr(param, "inside"), "Parameter should have inside attribute"
    assert param.inside is True, "Parameter should preserve inside=True"

    # Create ROI from parameter and check inside is preserved
    new_roi = sigima.objects.ImageROI.from_params(img, params)
    assert len(new_roi) == 1, "Should recreate one ROI"
    assert new_roi.single_rois[0].inside is True, (
        "Recreated ROI should have inside=True"
    )


if __name__ == "__main__":
    guiutils.enable_gui()
    test_roi_coordinates_validation()
    # test_image_roi_merge()
    # test_image_roi_combine()
    # test_image_roi_processing()
    # test_empty_image_roi()
    # test_image_extract_rois()
    # test_image_extract_roi()
    # test_create_image_roi_inside_parameter()
    # test_create_image_roi_inside_parameter_errors()
    # test_roi_inside_mask_behavior()
    # test_roi_inside_serialization()
    # test_roi_inside_parameter_conversion()
