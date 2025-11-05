# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
TextCallbackWorker end-to-end test
-----------------------------------

This test validates the TextCallbackWorker using real-world scenarios,
including reading signal data from CSV files with progress tracking and
cancellation support.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: skip

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from sigima.io.signal.formats import CSVSignalFormat
from sigima.worker import TextCallbackWorker


def create_large_csv_file(filepath: Path, nrows: int = 1000) -> None:
    """Create a large CSV file for testing progress tracking.

    Args:
        filepath: Path to the CSV file
        nrows: Number of rows to generate
    """
    with open(filepath, "w", encoding="utf-8") as f:
        # Write header
        f.write("x,y\n")
        # Write data
        for i in range(nrows):
            f.write(f"{i},{np.sin(i / 10.0)}\n")


class CancelingWorker(TextCallbackWorker):
    """Worker that cancels itself after a certain progress threshold."""

    def __init__(self, cancel_threshold: float = 0.5) -> None:
        """Initialize the canceling worker.

        Args:
            cancel_threshold: Progress value at which to cancel (0.0-1.0)
        """
        super().__init__()
        self.cancel_threshold = cancel_threshold
        self.progress_calls = []

    def set_progress(self, value: float) -> None:
        """Set progress and cancel if threshold is reached.

        Args:
            value: Progress value (0.0-1.0)
        """
        super().set_progress(value)
        self.progress_calls.append(value)
        if value >= self.cancel_threshold:
            self.cancel()


def test_worker_basic_functionality():
    """Test basic TextCallbackWorker functionality."""
    worker = TextCallbackWorker()

    # Initial state
    assert worker.get_progress() == 0.0
    assert not worker.was_canceled()

    # Set progress
    worker.set_progress(0.5)
    assert worker.get_progress() == 0.5
    assert not worker.was_canceled()

    # Progress clamping (below 0)
    worker.set_progress(-0.1)
    assert worker.get_progress() == 0.0

    # Progress clamping (above 1)
    worker.set_progress(1.5)
    assert worker.get_progress() == 1.0

    # Cancel
    worker.cancel()
    assert worker.was_canceled()


def test_worker_with_signal_reading(capsys):
    """End-to-end test: read signal data with progress tracking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a larger test CSV file to ensure chunked reading
        csv_file = Path(tmpdir) / "test_signal.csv"
        nrows = 50000  # Large enough to trigger multiple chunks
        create_large_csv_file(csv_file, nrows=nrows)

        # Read with worker using the CSV format directly
        worker = TextCallbackWorker()
        csv_format = CSVSignalFormat()
        signals = csv_format.read(str(csv_file), worker=worker)

        # Verify the signal was read correctly
        assert len(signals) == 1  # Single signal from x,y columns
        assert signals[0].data.shape[0] == nrows

        # Verify progress reached 100%
        assert worker.get_progress() == 1.0
        assert not worker.was_canceled()

        # Check that progress messages were printed
        captured = capsys.readouterr()
        assert "[sigima] Progress:" in captured.out
        assert "100%" in captured.out


def test_worker_with_cancellation():
    """End-to-end test: cancel signal reading mid-operation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a larger test CSV file to ensure chunked reading
        csv_file = Path(tmpdir) / "test_signal.csv"
        nrows = 50000  # Large enough to trigger multiple chunks
        create_large_csv_file(csv_file, nrows=nrows)

        # Use a worker that cancels at 50% progress
        worker = CancelingWorker(cancel_threshold=0.5)
        csv_format = CSVSignalFormat()
        signals = csv_format.read(str(csv_file), worker=worker)

        # Verify the operation was canceled
        assert worker.was_canceled()
        assert len(worker.progress_calls) > 0

        # The signals should still be returned (partial result)
        # but may have fewer rows than expected
        assert len(signals) >= 1
        assert signals[0].data.shape[0] <= nrows


def test_worker_without_output(capsys):
    """Test worker progress output is correctly formatted."""
    worker = TextCallbackWorker()

    # Capture output
    worker.set_progress(0.0)
    worker.set_progress(0.25)
    worker.set_progress(0.5)
    worker.set_progress(0.75)
    worker.set_progress(1.0)

    captured = capsys.readouterr()

    # Verify progress messages
    assert "[sigima] Progress: 0%" in captured.out
    assert "[sigima] Progress: 25%" in captured.out
    assert "[sigima] Progress: 50%" in captured.out
    assert "[sigima] Progress: 75%" in captured.out
    assert "[sigima] Progress: 100%" in captured.out


def test_worker_concurrent_operations():
    """Test multiple workers can operate independently."""
    worker1 = TextCallbackWorker()
    worker2 = TextCallbackWorker()

    worker1.set_progress(0.3)
    worker2.set_progress(0.7)

    assert worker1.get_progress() == 0.3
    assert worker2.get_progress() == 0.7

    worker1.cancel()

    assert worker1.was_canceled()
    assert not worker2.was_canceled()


if __name__ == "__main__":
    # Run pytest with verbose output
    pytest.main([__file__, "-v"])
