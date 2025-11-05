# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
ROI advanced unit tests
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

import numpy as np
import pytest

import sigima.objects
import sigima.params
import sigima.proc.signal
from sigima.tests.data import create_paracetamol_signal
from sigima.tests.helpers import print_obj_data_dimensions

SIZE = 200


def __create_test_signal() -> sigima.objects.SignalObj:
    """Create a test signal."""
    return create_paracetamol_signal(size=SIZE)


def test_signal_roi_param() -> None:
    """Test signal ROI parameter conversion"""
    obj = __create_test_signal()
    coords = [50, 100]
    roi = sigima.objects.create_signal_roi(coords, indices=True)
    roiparam = roi.to_params(obj)[0]
    assert isinstance(roiparam, sigima.objects.ROI1DParam), (
        "ROI parameter should be of type ROI1DParam"
    )
    # Check that converting back to single ROI gives the same coordinates
    single_roi = roiparam.to_single_roi(obj)
    single_roi_coords = single_roi.get_indices_coords(obj)
    assert np.array_equal(single_roi_coords, coords), (
        "Converted single ROI coordinates should match original"
    )


def test_signal_roi_merge() -> None:
    """Test signal ROI merge"""
    # Create a signal object with a single ROI, and another one with another ROI.
    # Compute the average of the two objects, and check if the resulting object
    # has the expected ROI (i.e. the union of the original object's ROI).
    obj1 = __create_test_signal()
    obj2 = __create_test_signal()
    obj2.roi = sigima.objects.create_signal_roi([60, 120], indices=True)
    obj1.roi = sigima.objects.create_signal_roi([50, 100], indices=True)

    # Compute the average of the two objects
    obj3 = sigima.proc.signal.average([obj1, obj2])
    assert obj3.roi is not None, "Merged object should have a ROI"
    assert len(obj3.roi) == 2, "Merged object should have two single ROIs"
    for single_roi in obj3.roi:
        assert single_roi.get_indices_coords(obj3) in ([50, 100], [60, 120]), (
            "Merged object should have the union of the original object's ROIs"
        )


def test_signal_roi_combine() -> None:
    """Test `SignalROI.combine_with` method"""
    coords1, coords2 = [60, 120], [50, 100]
    roi1 = sigima.objects.create_signal_roi(coords1, indices=True)
    roi2 = sigima.objects.create_signal_roi(coords2, indices=True)
    exp_combined = sigima.objects.create_signal_roi([coords1, coords2], indices=True)
    # Check that combining two ROIs results in a new ROI with both coordinates:
    roi3 = roi1.combine_with(roi2)
    assert roi3 == exp_combined, "Combined ROI should match expected"
    # Check that combining again with the same ROI does not change it:
    roi3 = roi1.combine_with(roi2)
    assert roi3 == exp_combined, "Combining with the same ROI should not change it"
    # Check that combining with an image ROI raises an error:
    with pytest.raises(
        TypeError, match=r"Cannot combine([\S ]*)SignalROI([\S ]*)ImageROI"
    ):
        roi1.combine_with(sigima.objects.create_image_roi("rectangle", [0, 0, 10, 10]))


# Signal ROIs:
SROI1 = [26, 41]
SROI2 = [125, 146]


def __roi_str(obj: sigima.objects.SignalObj) -> str:
    """Return a string representation of a SignalROI object for context."""
    if obj.roi is None:
        return "None"
    if obj.roi.is_empty():
        return "Empty"
    return ", ".join(
        f"[{r.get_indices_coords(obj)[0]}, {r.get_indices_coords(obj)[1]}]"
        for r in obj.roi.single_rois
    )


def __create_test_roi() -> sigima.objects.SignalROI:
    """Create a test ROI."""
    return sigima.objects.create_signal_roi([SROI1, SROI2], indices=True)


def __test_processing_in_roi(src: sigima.objects.SignalObj) -> None:
    """Run signal processing in ROI.

    Args:
        src: The source signal object (with or without ROI)
    """
    print_obj_data_dimensions(src)
    value = 1
    p = sigima.params.ConstantParam.create(value=value)
    dst = sigima.proc.signal.addition_constant(src, p)
    orig = src.data
    new = dst.data
    context = f" [ROI: {__roi_str(src)}]"
    if src.roi is not None and not src.roi.is_empty():
        # Check if the processed data is correct: signal should be the same as the
        # original data outside the ROI, and should be different inside the ROI.
        assert not np.any(new[SROI1[0] : SROI1[1]] == orig[SROI1[0] : SROI1[1]]), (
            f"Signal ROI 1 data mismatch{context}"
        )
        assert not np.any(new[SROI2[0] : SROI2[1]] == orig[SROI2[0] : SROI2[1]]), (
            f"Signal ROI 2 data mismatch{context}"
        )
        assert np.all(new[: SROI1[0]] == orig[: SROI1[0]]), (
            f"Signal before ROI 1 data mismatch{context}"
        )
        assert np.all(new[SROI1[1] : SROI2[0]] == orig[SROI1[1] : SROI2[0]]), (
            f"Signal between ROIs data mismatch{context}"
        )
        assert np.all(new[SROI2[1] :] == orig[SROI2[1] :]), (
            f"Signal after ROI 2 data mismatch{context}"
        )
    else:
        # No ROI: all data should be changed
        assert np.all(new == orig + value), f"Signal data mismatch{context}"


def test_signal_roi_processing() -> None:
    """Test signal ROI processing"""
    src = __create_test_signal()
    base_roi = __create_test_roi()
    empty_roi = sigima.objects.SignalROI()
    for roi in (None, empty_roi, base_roi):
        src.roi = roi
        __test_processing_in_roi(src)


def test_empty_signal_roi() -> None:
    """Test empty signal ROI"""
    src = __create_test_signal()
    empty_roi = sigima.objects.SignalROI()
    for roi in (None, empty_roi):
        src.roi = roi
        context = f" [ROI: {__roi_str(src)}]"
        assert src.roi is None or src.roi.is_empty(), (
            f"Source object ROI should be empty or None{context}"
        )
        if src.roi is not None:
            # No ROI has been set in the source signal
            sig1 = sigima.proc.signal.extract_roi(src, src.roi.to_params(src))
            assert sig1.data.size == 0, f"Extracted signal should be empty{context}"


@pytest.mark.validation
def test_signal_extract_rois() -> None:
    """Validation test for signal ROI extraction into a single object"""
    src = __create_test_signal()
    src.roi = __create_test_roi()
    context = f" [ROI: {__roi_str(src)}]"
    size_roi1, size_roi2 = SROI1[1] - SROI1[0], SROI2[1] - SROI2[0]
    assert len(src.roi) == 2, f"Source object should have two ROIs{context}"
    # Single object mode: merge all ROIs into a single object
    sig1 = sigima.proc.signal.extract_rois(src, src.roi.to_params(src))
    assert sig1.data.size == size_roi1 + size_roi2, f"Signal size mismatch{context}"
    assert np.all(sig1.data[:size_roi1] == src.data[SROI1[0] : SROI1[1]]), (
        f"Signal 1 data mismatch{context}"
    )
    assert np.all(sig1.data[size_roi1:] == src.data[SROI2[0] : SROI2[1]]), (
        f"Signal 2 data mismatch{context}"
    )
    # Verify that extracted signal doesn't have ROI (bug fix verification)
    assert sig1.roi is None, (
        "Extracted signal should not have ROI (bug fix: ROI should not be "
        "copied to extracted signal)"
    )

    # Create a single ROI and extract it (for test coverage)
    single_roi = sigima.objects.create_signal_roi([SROI1], indices=True)
    sig2 = sigima.proc.signal.extract_rois(src, single_roi.to_params(src))
    assert sig2.data.size == size_roi1, f"Signal size mismatch{context}"


@pytest.mark.validation
def test_signal_extract_roi() -> None:
    """Validation test for signal ROI extraction into multiple objects"""
    src = __create_test_signal()
    src.roi = __create_test_roi()
    context = f" [ROI: {__roi_str(src)}]"
    size_roi1, size_roi2 = SROI1[1] - SROI1[0], SROI2[1] - SROI2[0]
    assert len(src.roi) == 2, f"Source object should have two ROIs{context}"
    # Multiple objects mode: extract each ROI as a separate object
    signals: list[sigima.objects.SignalObj] = []
    for index, single_roi in enumerate(src.roi):
        roiparam = single_roi.to_param(src, index)
        signal = sigima.proc.signal.extract_roi(src, roiparam)
        signals.append(signal)
    assert len(signals) == len(src.roi), (
        f"Number of extracted signals mismatch{context}"
    )
    assert signals[0].data.size == size_roi1, f"Signal 1 size mismatch{context}"
    assert signals[1].data.size == size_roi2, f"Signal 2 size mismatch{context}"
    assert np.all(signals[0].data == src.data[SROI1[0] : SROI1[1]]), (
        f"Signal 1 data mismatch{context}"
    )
    assert np.all(signals[1].data == src.data[SROI2[0] : SROI2[1]]), (
        f"Signal 2 data mismatch{context}"
    )
    # Verify that extracted signals don't have ROIs (bug fix verification)
    assert signals[0].roi is None, (
        "Extracted signal 1 should not have ROI (bug fix: ROIs should not be "
        "copied to extracted signals)"
    )
    assert signals[1].roi is None, (
        "Extracted signal 2 should not have ROI (bug fix: ROIs should not be "
        "copied to extracted signals)"
    )


def test_signal_roi_union() -> None:
    """Test signal ROI union operation"""
    # Test union of overlapping ROIs
    roi1 = sigima.objects.create_signal_roi([[10, 30], [20, 40]], indices=True)
    roi_union = roi1.union()
    assert len(roi_union) == 1, "Overlapping ROIs should merge into one"
    assert np.array_equal(roi_union.single_rois[0].coords, [10, 40]), (
        "Union should span full range"
    )

    # Test union of non-overlapping ROIs
    roi2 = sigima.objects.create_signal_roi([[10, 20], [30, 40]], indices=True)
    roi_union2 = roi2.union()
    assert len(roi_union2) == 2, "Non-overlapping ROIs should remain separate"

    # Test union of adjacent ROIs
    roi3 = sigima.objects.create_signal_roi([[10, 20], [20, 30]], indices=True)
    roi_union3 = roi3.union()
    assert len(roi_union3) == 1, "Adjacent ROIs should merge"
    assert np.array_equal(roi_union3.single_rois[0].coords, [10, 30]), (
        "Adjacent union should span full range"
    )

    # Test empty ROI union
    empty_roi = sigima.objects.SignalROI()
    empty_union = empty_roi.union()
    assert len(empty_union) == 0, "Empty ROI union should be empty"


def test_signal_roi_clipping() -> None:
    """Test signal ROI clipping operation"""
    src = __create_test_signal()
    x_min, x_max = src.x[0], src.x[-1]

    # Test clipping ROI within signal range
    roi = sigima.objects.create_signal_roi([[x_min + 10, x_max - 10]], indices=False)
    original_coords = roi.single_rois[0].coords.copy()
    clipped_roi = roi.clipped(x_min, x_max)
    assert len(clipped_roi) == 1, "ROI within range should remain"
    assert np.array_equal(clipped_roi.single_rois[0].coords, original_coords), (
        "ROI within range should be unchanged"
    )

    # Test clipping ROI partially outside signal range
    roi2 = sigima.objects.create_signal_roi(
        [[x_min - 5, x_min + 10], [x_max - 10, x_max + 5]], indices=False
    )
    clipped_roi2 = roi2.clipped(x_min, x_max)
    assert len(clipped_roi2) == 2, "Partially outside ROIs should be clipped"
    assert clipped_roi2.single_rois[0].coords[0] == x_min, (
        "Left boundary should be clipped to x_min"
    )
    assert clipped_roi2.single_rois[1].coords[1] == x_max, (
        "Right boundary should be clipped to x_max"
    )

    # Test clipping ROI completely outside signal range
    roi3 = sigima.objects.create_signal_roi([[x_max + 1, x_max + 10]], indices=False)
    clipped_roi3 = roi3.clipped(x_min, x_max)
    assert len(clipped_roi3) == 0, "ROI completely outside range should be removed"


def test_signal_roi_inversion() -> None:
    """Test signal ROI inversion operation"""
    src = __create_test_signal()
    x_min, x_max = src.x[0], src.x[-1]

    # Test inversion of single ROI in middle
    roi = sigima.objects.create_signal_roi([[20, 30]], indices=False)
    inverted = roi.inverted(x_min, x_max)
    assert len(inverted) == 2, "Single middle ROI should create two inverted segments"
    assert inverted.single_rois[0].coords[0] == x_min, (
        "First segment should start at x_min"
    )
    assert inverted.single_rois[0].coords[1] == 20, (
        "First segment should end at ROI start"
    )
    assert inverted.single_rois[1].coords[0] == 30, (
        "Second segment should start at ROI end"
    )
    assert inverted.single_rois[1].coords[1] == x_max, (
        "Second segment should end at x_max"
    )

    # Test inversion of ROI at signal start
    roi_start = sigima.objects.create_signal_roi([[x_min, 20]], indices=False)
    inverted_start = roi_start.inverted(x_min, x_max)
    assert len(inverted_start) == 1, "ROI at start should create one inverted segment"
    assert inverted_start.single_rois[0].coords[0] == 20, (
        "Inverted segment should start after ROI"
    )
    assert inverted_start.single_rois[0].coords[1] == x_max, (
        "Inverted segment should end at x_max"
    )

    # Test inversion of ROI at signal end
    roi_end = sigima.objects.create_signal_roi([[30, x_max]], indices=False)
    inverted_end = roi_end.inverted(x_min, x_max)
    assert len(inverted_end) == 1, "ROI at end should create one inverted segment"
    assert inverted_end.single_rois[0].coords[0] == x_min, (
        "Inverted segment should start at x_min"
    )
    assert inverted_end.single_rois[0].coords[1] == 30, (
        "Inverted segment should end before ROI"
    )

    # Test inversion of multiple ROIs
    roi_multi = sigima.objects.create_signal_roi([[10, 15], [20, 30]], indices=False)
    inverted_multi = roi_multi.inverted(x_min, x_max)
    assert len(inverted_multi) == 3, "Two ROIs should create three inverted segments"

    # Test error case: empty ROI inversion
    empty_roi = sigima.objects.SignalROI()
    with pytest.raises(ValueError, match="No ROIs defined, cannot invert"):
        empty_roi.inverted(x_min, x_max)


def test_signal_roi_mask() -> None:
    """Test signal ROI mask creation"""
    src = __create_test_signal()

    # Test mask for single ROI
    roi = sigima.objects.create_signal_roi([SROI1], indices=True)
    mask = roi.to_mask(src)
    assert mask.shape == src.xydata.shape, "Mask should have same shape as data"
    assert mask.dtype == bool, "Mask should be boolean array"
    # Check that ROI region is masked (False values)
    assert not np.any(mask[:, SROI1[0] : SROI1[1]]), "ROI region should be masked"
    # Check that non-ROI regions are not masked (True values)
    assert np.all(mask[:, : SROI1[0]]), "Region before ROI should not be masked"
    assert np.all(mask[:, SROI1[1] :]), "Region after ROI should not be masked"

    # Test mask for multiple ROIs
    roi_multi = __create_test_roi()
    mask_multi = roi_multi.to_mask(src)
    assert not np.any(mask_multi[:, SROI1[0] : SROI1[1]]), (
        "First ROI region should be masked"
    )
    assert not np.any(mask_multi[:, SROI2[0] : SROI2[1]]), (
        "Second ROI region should be masked"
    )
    assert np.all(mask_multi[:, SROI1[1] : SROI2[0]]), (
        "Region between ROIs should not be masked"
    )

    # Test mask for empty ROI
    empty_roi = sigima.objects.SignalROI()
    empty_mask = empty_roi.to_mask(src)
    assert not np.any(empty_mask), "Empty ROI should mask everything"


def test_signal_roi_operations_edge_cases() -> None:
    """Test edge cases for signal ROI operations"""
    src = __create_test_signal()
    x_min, x_max = src.x[0], src.x[-1]

    # Test union with identical ROIs
    roi_identical = sigima.objects.create_signal_roi([[10, 20], [10, 20]], indices=True)
    union_identical = roi_identical.union()
    assert len(union_identical) == 1, "Identical ROIs should merge to one"

    # Test clipping with ROI exactly at boundaries
    roi_boundary = sigima.objects.create_signal_roi([[x_min, x_max]], indices=False)
    roi_boundary.clipped(x_min, x_max)
    assert len(roi_boundary) == 1, "ROI at exact boundaries should remain"
    assert np.array_equal(roi_boundary.single_rois[0].coords, [x_min, x_max]), (
        "Boundary ROI should be unchanged"
    )

    # Test inversion with ROI covering entire signal
    roi_full = sigima.objects.create_signal_roi([[x_min, x_max]], indices=False)
    inverted_full = roi_full.inverted(x_min, x_max)
    assert len(inverted_full) == 0, "Full signal ROI should invert to empty"

    # Test operations with very small ROIs
    small_roi = sigima.objects.create_signal_roi([[10, 10.1]], indices=False)
    small_union = small_roi.union()
    assert len(small_union) == 1, "Small ROI should remain in union"

    small_roi.clipped(0, 100)
    assert len(small_roi) == 1, "Small ROI within range should remain after clipping"


if __name__ == "__main__":
    test_signal_roi_merge()
    test_signal_roi_combine()
    test_signal_roi_processing()
    test_empty_signal_roi()
    test_signal_extract_rois()
    test_signal_extract_roi()
    test_signal_roi_union()
    test_signal_roi_clipping()
    test_signal_roi_inversion()
    test_signal_roi_mask()
    test_signal_roi_operations_edge_cases()
