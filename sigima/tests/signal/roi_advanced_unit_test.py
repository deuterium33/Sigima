# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
ROI advanced unit tests
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

import pytest

import sigima.objects
import sigima.proc.signal
from sigima.tests.data import create_paracetamol_signal


def test_signal_roi_merge() -> None:
    """Test signal ROI merge"""
    # Create a signal object with a single ROI, and another one with another ROI.
    # Compute the average of the two objects, and check if the resulting object
    # has the expected ROI (i.e. the union of the original object's ROI).
    obj1 = create_paracetamol_signal()
    obj2 = create_paracetamol_signal()
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


if __name__ == "__main__":
    test_signal_roi_merge()
    test_signal_roi_combine()
