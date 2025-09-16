# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""I/O utility functions."""

# pylint: disable=invalid-name  # Allows short reference names like x, y...

from __future__ import annotations

import os
from itertools import islice

from sigima.io.enums import FileEncoding


def count_lines(filename: str | os.PathLike[str]) -> int:
    """Count the number of lines in a file.

    Args:
        filename: File name or path.

    Returns:
        The number of lines in the file.

    Raises:
        IOError: If the file cannot be read.
    """
    for encoding in FileEncoding:
        try:
            with open(filename, "r", encoding=encoding) as file:
                line_count = sum(1 for _ in file)
            return line_count
        except UnicodeDecodeError:
            # Try next encoding.
            pass
    raise IOError(f"Cannot read file {filename}.")


def read_first_n_lines(filename: str | os.PathLike[str], n: int = 100000) -> str:
    """Read the first `n` lines of a file.

    Args:
        filename: File name or path.
        n: Number of lines to read.

    Returns:
        The first `n` lines of the file.

    Raises:
        IOError: If the file cannot be read.
    """
    for encoding in FileEncoding:
        try:
            with open(filename, "r", encoding=encoding) as file:
                return "".join(islice(file, n))
        except UnicodeDecodeError:
            # Try next encoding.
            pass
    raise IOError(f"Cannot read file {filename}.")
