# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
CDLSignal2 test
---------------

Checking DataLab misc analysis functions
"""

from sigima.tests.helpers import check_scalar_result
from sigima.tests.signal.utils import generate_step_signal
from sigima.tools.signal import features


def test_get_crossing_time():
    """Test get_crossing_time for both positive and negative polarity step signals."""
    # positive polarity
    x, y_noisy = generate_step_signal(seed=0)

    crossing_time = features.get_crossing_time(x, y_noisy, 1)
    check_scalar_result(
        "step, get crossing time, positive polarity", crossing_time, 3.44
    )

    # negative polarity
    x, y_noisy = generate_step_signal(seed=0, y_initial=5, y_final=2)

    crossing_time = features.get_crossing_time(x, y_noisy, 4, False)
    check_scalar_result(
        "step,  get crossing time, negative polarity", crossing_time, 3.645
    )
