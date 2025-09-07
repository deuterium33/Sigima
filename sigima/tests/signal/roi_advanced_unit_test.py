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


if __name__ == "__main__":
    test_signal_roi_merge()
    test_signal_roi_combine()
    test_signal_roi_processing()
    test_empty_signal_roi()
    test_signal_extract_rois()
    test_signal_extract_roi()
