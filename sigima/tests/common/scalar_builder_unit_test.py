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


class TestTableResultBuilder:
    """Test class for TableResultBuilder basic functionality."""

    def test_basic_functionality(self) -> None:
        """Test basic TableResultBuilder API with a SignalObj."""
        sig = create_dummy_signal()

        builder = TableResultBuilder("Signal Stats")
        builder.add(ma.min, "min")
        builder.add(ma.max, "max")
        builder.add(ma.mean, "mean")

        table = builder.compute(sig)

        assert table.title == "Signal Stats"
        # [None, ROI_0] x 3 stats
        assert len(table.data) == 2 and len(table.data[0]) == 3
        assert list(table.headers) == ["min", "max", "mean"]
        assert table.roi_indices[0] == -1  # NO_ROI
        assert table.roi_indices[1] == 0

        # Check actual values
        row_none = table.data[0]
        row_roi = table.data[1]
        assert isinstance(row_none[0], float)
        assert isinstance(row_roi[1], float)


class TestTableResultBuilderHideColumns:
    """Test class for TableResultBuilder hide_columns functionality."""

    def setup_method(self) -> None:
        """Set up test data for each test method."""
        self.sig = create_dummy_signal()

    def _create_basic_builder(self) -> TableResultBuilder:
        """Helper method to create a basic builder with standard columns."""
        builder = TableResultBuilder("Signal Stats")
        builder.add(ma.min, "min")
        builder.add(ma.max, "max")
        builder.add(ma.mean, "mean")
        builder.add(ma.std, "std")
        return builder

    def _assert_display_preferences(
        self, table, expected_prefs: dict[str, bool]
    ) -> None:
        """Helper method to check display preferences and visible headers."""
        prefs = table.get_display_preferences()
        assert prefs == expected_prefs

        expected_visible = [name for name, visible in expected_prefs.items() if visible]
        visible_headers = table.get_visible_headers()
        assert set(visible_headers) == set(expected_visible)

    def test_hide_some_columns(self) -> None:
        """Test hiding some columns."""
        builder = self._create_basic_builder()
        builder.hide_columns(["max", "std"])

        table = builder.compute(self.sig)

        assert table.title == "Signal Stats"
        # All headers still present
        assert list(table.headers) == ["min", "max", "mean", "std"]

        self._assert_display_preferences(
            table, {"min": True, "max": False, "mean": True, "std": False}
        )

    def test_hide_nonexistent_columns(self) -> None:
        """Test hiding non-existent columns."""
        builder = TableResultBuilder("Signal Stats")
        builder.add(ma.min, "min")
        builder.add(ma.max, "max")

        # Try to hide non-existent column - should not cause error
        builder.hide_columns(["nonexistent", "min"])

        table = builder.compute(self.sig)

        self._assert_display_preferences(table, {"min": False, "max": True})

    def test_hide_empty_list(self) -> None:
        """Test hiding empty list of columns."""
        builder = TableResultBuilder("Signal Stats")
        builder.add(ma.min, "min")
        builder.add(ma.max, "max")

        builder.hide_columns([])

        table = builder.compute(self.sig)

        self._assert_display_preferences(table, {"min": True, "max": True})

    def test_hide_multiple_calls(self) -> None:
        """Test multiple hide_columns calls accumulate."""
        builder = self._create_basic_builder()

        # Hide columns in multiple calls
        builder.hide_columns(["max"])
        builder.hide_columns(["std"])

        table = builder.compute(self.sig)

        self._assert_display_preferences(
            table, {"min": True, "max": False, "mean": True, "std": False}
        )

    def test_hide_all_columns(self) -> None:
        """Test hiding all columns."""
        builder = TableResultBuilder("Signal Stats")
        builder.add(ma.min, "min")
        builder.add(ma.max, "max")

        builder.hide_columns(["min", "max"])

        table = builder.compute(self.sig)

        self._assert_display_preferences(table, {"min": False, "max": False})
