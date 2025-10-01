# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for TableResultBuilder (sigima.objects.scalar).
"""

from numpy import ma

from sigima.objects.scalar import TableResultBuilder
from sigima.objects.signal import SignalObj
from sigima.tests.data import create_paracetamol_signal, create_test_signal_rois


def create_dummy_signal() -> SignalObj:
    """Create a simple SignalObj with a single ROI."""
    sig = create_paracetamol_signal()
    roi = list(create_test_signal_rois(sig))[0]
    sig.roi = roi
    return sig


def test_table_result_builder_basic() -> None:
    """Basic test of TableResultBuilder API with a SignalObj."""
    sig = create_dummy_signal()

    builder = TableResultBuilder("Signal Stats")
    builder.add(ma.min, "min")
    builder.add(ma.max, "max")
    builder.add(ma.mean, "mean")

    table = builder.compute(sig)

    assert table.title == "Signal Stats"
    assert len(table.data) == 2 and len(table.data[0]) == 3  # [None, ROI_0] x 3 stats
    assert list(table.headers) == ["min", "max", "mean"]
    assert table.roi_indices[0] == -1  # NO_ROI
    assert table.roi_indices[1] == 0

    # Check actual values
    row_none = table.data[0]
    row_roi = table.data[1]
    assert isinstance(row_none[0], float)
    assert isinstance(row_roi[1], float)
