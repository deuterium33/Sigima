# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Testing validation test introspection and CSV generation.
"""

import pytest

from sigima.proc.validation import ValidationStatistics
from sigima.tests.helpers import WorkdirRestoringTempDir


@pytest.mark.skip("Punctual check: there is no need to run this test every time")
def test_validation_statistics() -> None:
    """Test validation statistics introspection and CSV generation."""
    stats = ValidationStatistics()
    stats.collect_validation_status(verbose=True)
    stats.get_validation_info()
    with WorkdirRestoringTempDir() as tmpdir:
        stats.generate_csv_files(tmpdir)
        stats.generate_statistics_csv(tmpdir)


if __name__ == "__main__":
    test_validation_statistics()
