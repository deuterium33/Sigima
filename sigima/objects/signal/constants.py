# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal I/O constants - especially for datetime metadata handling
"""

# Datetime metadata keys
# These keys are used to store datetime-related information in SignalObj.metadata
DATETIME_X_KEY = "x_datetime"  # Boolean: True if x represents datetime data
DATETIME_X_FORMAT_KEY = "x_datetime_format"  # String: format for display

# Time unit conversion factors (to seconds)
# Used to convert datetime values to float timestamps
TIME_UNIT_FACTORS = {
    "ns": 1e-9,  # nanoseconds
    "us": 1e-6,  # microseconds
    "ms": 1e-3,  # milliseconds
    "s": 1.0,  # seconds (base unit)
    "min": 60.0,  # minutes
    "h": 3600.0,  # hours
}

# Valid time units (ordered from smallest to largest)
VALID_TIME_UNITS = list(TIME_UNIT_FACTORS.keys())

# Default datetime format string
DEFAULT_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
