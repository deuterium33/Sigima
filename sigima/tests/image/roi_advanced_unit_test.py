# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Image ROI advanced unit tests"""

from __future__ import annotations

import numpy as np
import pytest
from skimage import draw

import sigima.objects
import sigima.params
import sigima.proc.image
from sigima.objects import ImageObj, ImageROI, NewImageParam, create_image_roi
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
    obj2.roi = sigima.objects.create_image_roi("rectangle", [600, 800, 1000, 1200])
    obj1.roi = sigima.objects.create_image_roi("rectangle", [500, 750, 1000, 1250])

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
    roi1 = sigima.objects.create_image_roi("rectangle", coords1, indices=True)
    roi2 = sigima.objects.create_image_roi("rectangle", coords2, indices=True)
    exp_combined = sigima.objects.create_image_roi(
        "rectangle", [coords1, coords2], indices=True
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


def __create_test_roi() -> ImageROI:
    """Create test ROI"""
    roi = create_image_roi("rectangle", IROI1)
    roi.add_roi(create_image_roi("circle", IROI2))
    roi.add_roi(create_image_roi("polygon", IROI3))
    return roi


def __create_test_image() -> ImageObj:
    """Create test image"""
    param = NewImageParam.create(height=SIZE, width=SIZE)
    ima = create_multigaussian_image(param)
    ima.data += 1  # Ensure that the image has non-zero values (for ROI check tests)
    return ima


def __test_processing_in_roi(src: ImageObj) -> None:
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
    if src.roi is not None and not src.roi.is_empty():
        # A ROI has been set in the source image.
        assert np.all(
            new[IROI1[1] : IROI1[3] + IROI1[1], IROI1[0] : IROI1[2] + IROI1[0]]
            == orig[IROI1[1] : IROI1[3] + IROI1[1], IROI1[0] : IROI1[2] + IROI1[0]]
            + value
        ), "Image ROI 1 data mismatch"
        assert np.all(
            new[IROI2[1] : IROI1[1] + 1, IROI2[0] : IROI2[0] + 2 * IROI2[2]]
            == orig[IROI2[1] : IROI1[1] + 1, IROI2[0] : IROI2[0] + 2 * IROI2[2]] + value
        ), "Image ROI 2 data mismatch"
        first_col = min(IROI1[0], IROI2[0] - IROI2[2])
        first_row = min(IROI1[1], IROI2[1] - IROI2[2])
        last_col = max(IROI1[0] + IROI1[2], IROI2[0] + 2 * IROI2[2])
        last_row = max(IROI1[1] + IROI1[3], IROI2[1] + 2 * IROI2[2])
        assert np.all(
            new[:first_row, :first_col] == np.array(orig[:first_row, :first_col], float)
        ), "Image before ROIs data mismatch"
        assert np.all(new[:first_row, last_col:] == orig[:first_row, last_col:]), (
            "Image after ROIs data mismatch"
        )
        assert np.all(new[last_row:, :first_col] == orig[last_row:, :first_col]), (
            "Image before ROIs data mismatch"
        )
        assert np.all(new[last_row:, last_col:] == orig[last_row:, last_col:]), (
            "Image after ROIs data mismatch"
        )
    else:
        # No ROI has been set in the source image.
        assert np.all(new == orig + value), "Image data mismatch"


def __test_extracting_from_roi(src: ImageObj, singleobj: bool | None = None) -> None:
    """Run image extraction from ROI

    Args:
        src: Source image object (with or without ROI)
        singleobj: Whether to use single object ROI
    """
    # Assertions texts:
    nzroi = "Non-zero values expected in ROI"
    zroi = "Zero values expected outside ROI"
    roisham = "ROI shape mismatch"

    if src.roi is not None and not src.roi.is_empty():
        # A ROI has been set in the source image.
        if singleobj:
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
        else:
            images: list[ImageObj] = []
            for index, single_roi in enumerate(src.roi):
                roiparam = single_roi.to_param(src, index)
                image = sigima.proc.image.extract_roi(src, roiparam)
                images.append(image)
            assert len(images) == 3, "Three images expected"
            im1, im2 = images[:2]  # pylint: disable=unbalanced-tuple-unpacking
            assert np.all(im1.data != 0), nzroi
            assert im1.data.shape == (IROI1[3], IROI1[2]), roisham
            assert np.all(im2.data != 0), nzroi
            assert im2.data.shape == (IROI2[2] * 2, IROI2[2] * 2), roisham
            mask2 = np.zeros(shape=im2.data.shape, dtype=bool)
            xc = yc = r = IROI2[2]  # Adjust for ROI origin
            rr, cc = draw.disk((yc, xc), r, shape=im2.data.shape)
            mask2[rr, cc] = 1
            assert np.all(im2.maskdata == ~mask2), "Mask data mismatch"
    else:
        # No ROI has been set in the source image.
        im1 = sigima.proc.image.extract_roi(src, src.roi.to_params(src))
        assert im1.data.shape == (0, 0), "Extracted image should be empty"


def test_image_roi_processing() -> None:
    """Test image ROI processing"""
    src = __create_test_image()
    base_roi = __create_test_roi()
    empty_roi = ImageROI()
    for roi in (empty_roi, base_roi):
        src.roi = roi
        __test_processing_in_roi(src)


def test_image_roi_extraction() -> None:
    """Test image ROI extraction"""
    src = __create_test_image()
    base_roi = __create_test_roi()
    empty_roi = ImageROI()
    for roi in (empty_roi, base_roi):
        for singleobj in (False, True):
            src.roi = roi
            __test_extracting_from_roi(src, singleobj)


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
    rect_roi = create_image_roi("rectangle", rect_coords, title="rectangular")
    circ_roi = create_image_roi("circle", circ_coords, title="circular")
    poly_roi = create_image_roi("polygon", poly_coords, title="polygonal")

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
        for roi in (rect_roi, circ_roi, poly_roi):
            src2 = src.copy()
            src2.roi = roi
            images.append(src2)
            titles.append(f"Image with {roi.get_single_roi(0).title} ROI")
        guiutils.view_images_side_by_side_if_gui_enabled(
            images, titles, rows=2, title="Image ROIs"
        )


if __name__ == "__main__":
    guiutils.enable_gui()
    test_roi_coordinates_validation()
    test_image_roi_merge()
    test_image_roi_combine()
    test_image_roi_processing()
    test_image_roi_extraction()
