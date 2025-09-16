# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Common enum definitions for Sigima I/O support."""

# pylint: disable=invalid-name  # Allows short reference names like x, y...

from __future__ import annotations

from enum import Enum


class FileEncoding(str, Enum):
    """File encodings."""

    UTF8 = "utf-8"
    UTF8_SIG = "utf-8-sig"
    LATIN1 = "latin-1"
