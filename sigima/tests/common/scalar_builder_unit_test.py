# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for TableResultBuilder (sigima.objects.scalar).
"""

from numpy import ma

from sigima.objects import (
    NO_ROI,
    SignalObj,
    TableKind,
    TableResultBuilder,
    create_signal_roi,
)
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
        # pylint: disable=attribute-defined-outside-init
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


class TestTableResultBuilderROIComputationModes:
    """Test class for TableResultBuilder ROI computation modes."""

    def setup_method(self) -> None:
        """Set up test data for each test method."""
        # pylint: disable=attribute-defined-outside-init
        # Create signal with multiple ROIs (3 segments)
        self.sig = create_paracetamol_signal()
        # Create a ROI with 3 segments manually
        roi = create_signal_roi([[10, 20], [30, 40], [50, 60]], indices=False)
        self.sig.roi = roi

    def test_statistics_computes_whole_and_rois(self) -> None:
        """Test STATISTICS kind computes both whole object and ROIs."""
        builder = TableResultBuilder("Signal Statistics", kind=TableKind.STATISTICS)
        builder.add(ma.min, "min")
        builder.add(ma.max, "max")

        table = builder.compute(self.sig)

        # Should have results for whole object (NO_ROI) + 3 ROIs = 4 rows
        assert len(table.data) == 4
        assert table.roi_indices[0] == NO_ROI  # First row is whole object
        assert table.roi_indices[1] == 0  # Then ROI 0
        assert table.roi_indices[2] == 1  # Then ROI 1
        assert table.roi_indices[3] == 2  # Then ROI 2

    def test_pulse_features_computes_only_rois(self) -> None:
        """Test PULSE_FEATURES kind computes ONLY ROIs when they exist."""
        builder = TableResultBuilder("Pulse Features", kind=TableKind.PULSE_FEATURES)
        builder.add(ma.min, "min")
        builder.add(ma.max, "max")

        table = builder.compute(self.sig)

        # Should have results ONLY for ROIs (no whole object), so 3 rows
        assert len(table.data) == 3
        assert table.roi_indices[0] == 0  # First row is ROI 0
        assert table.roi_indices[1] == 1  # Then ROI 1
        assert table.roi_indices[2] == 2  # Then ROI 2
        # Verify NO_ROI is not present
        assert NO_ROI not in table.roi_indices

    def test_pulse_features_computes_whole_when_no_rois(self) -> None:
        """Test PULSE_FEATURES kind computes whole object when no ROIs exist."""
        # Remove ROI from signal
        sig_no_roi = create_paracetamol_signal()

        builder = TableResultBuilder("Pulse Features", kind=TableKind.PULSE_FEATURES)
        builder.add(ma.min, "min")
        builder.add(ma.max, "max")

        table = builder.compute(sig_no_roi)

        # Should have result for whole object only (1 row)
        assert len(table.data) == 1
        assert table.roi_indices[0] == NO_ROI

    def test_custom_kind_default_behavior(self) -> None:
        """Test CUSTOM kind uses default behavior (whole + ROIs like STATISTICS)."""
        builder = TableResultBuilder("Custom Results", kind=TableKind.CUSTOM)
        builder.add(ma.min, "min")
        builder.add(ma.max, "max")

        table = builder.compute(self.sig)

        # Should have results for whole object + ROIs = 4 rows (same as STATISTICS)
        assert len(table.data) == 4
        assert table.roi_indices[0] == NO_ROI
        assert table.roi_indices[1] == 0
        assert table.roi_indices[2] == 1
        assert table.roi_indices[3] == 2

    def test_string_kind_pulse_features(self) -> None:
        """Test using string 'pulse_features' as kind value."""
        builder = TableResultBuilder("Pulse Features", kind="pulse_features")
        builder.add(ma.min, "min")
        builder.add(ma.max, "max")

        table = builder.compute(self.sig)

        # Should behave like PULSE_FEATURES enum: only ROIs, no whole object
        assert len(table.data) == 3
        assert NO_ROI not in table.roi_indices

    def test_string_kind_statistics(self) -> None:
        """Test using string 'statistics' as kind value."""
        builder = TableResultBuilder("Signal Statistics", kind="statistics")
        builder.add(ma.min, "min")
        builder.add(ma.max, "max")

        table = builder.compute(self.sig)

        # Should behave like STATISTICS enum: whole + ROIs
        assert len(table.data) == 4
        assert table.roi_indices[0] == NO_ROI
