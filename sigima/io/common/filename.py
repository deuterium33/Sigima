# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Common functions for file name handling"""

from __future__ import annotations

import re
import string
import sys
import unicodedata
from typing import Any, Iterable

from sigima.objects.image import ImageObj
from sigima.objects.signal import SignalObj


class CustomFormatter(string.Formatter):
    """Custom string formatter to handle uppercase and lowercase strings."""

    def format_field(self, value, format_spec):
        if isinstance(value, str):
            if format_spec.endswith("u"):
                value = value.upper()
                format_spec = format_spec[:-1]
            elif format_spec.endswith("l"):
                value = value.lower()
                format_spec = format_spec[:-1]
        return super().format_field(value, format_spec)


def sanitize_filename(title: str, replacement: str = "_") -> str:
    """
    Sanitize a string to create a valid filename for the current operating system.

    This function removes or replaces characters that are invalid in filenames,
    depending on the underlying OS (Windows, macOS, Linux). It also strips
    trailing dots and spaces on Windows and normalizes unicode characters.

    Args:
        title (str): Input string (e.g., signal or image title).
        replacement (str): Replacement string for invalid characters (default: "_").

    Returns:
        str: A sanitized string that can safely be used as a filename.
    """
    # Normalize unicode characters (NFKD form for decomposing accents, etc.)
    title = unicodedata.normalize("NFKD", title)
    title = title.encode("ascii", "ignore").decode("ascii")

    # Characters not allowed in filenames (platform-dependent)
    if sys.platform.startswith("win"):
        # Reserved characters on Windows
        invalid_chars = r'[<>:"/\\|?*\x00-\x1F]'
        reserved_names = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        }
    else:
        # Only '/' is disallowed on Unix-based systems
        invalid_chars = r"/"

        reserved_names = set()

    # Replace invalid characters
    sanitized = re.sub(invalid_chars, replacement, title)

    # Strip leading/trailing whitespace and dots (Windows limitation)
    sanitized = sanitized.strip(" .")

    # Truncate to a reasonable length to avoid OS/path issues
    sanitized = sanitized[:255]

    # Avoid reserved filenames on Windows
    if sanitized.upper() in reserved_names:
        sanitized += "_"

    # If result is empty, fallback
    if not sanitized:
        sanitized = "untitled"

    return sanitized


def format_object_names(
    objects: Iterable[SignalObj | ImageObj],
    fmt: str,
    replacement: str = "_",
) -> list[str]:
    """
    Generate a list of sanitized names for the given Signal/Image objects.

    The `fmt` parameter is a standard Python format string consumed with `str.format`.
    Available placeholders (use any subset):
      - {title}: sanitized object title (already sanitized with `replacement`)
      - {type}: "signal" or "image"
      - {index}: 1-based index
      - {i}: 0-based index
      - {n}: total number of objects
      - {xlabel}, {xunit}, {ylabel}, {yunit}: axis titles/units if present (signals)
      - {metadata}: the metadata mapping (you may use e.g. {metadata[key]} in fmt)

    Args:
        objects: Iterable of SignalObj or ImageObj
        fmt: Python format string (e.g., "{name}_{index:02d}")
        replacement: Replacement string for invalid filename characters

    Returns:
        List of sanitized names corresponding to each object.

    Raises:
        KeyError: if fmt references an unknown placeholder
        ValueError: if fmt is otherwise invalid
    """
    items = list(objects)
    n = len(items)
    result: list[str] = []

    for i, obj in enumerate(items):
        otype = "signal" if isinstance(obj, SignalObj) else "image"
        non_sanitized_title = getattr(obj, "title", "") or ""
        title = sanitize_filename(str(non_sanitized_title), replacement=replacement)

        context: dict[str, Any] = {
            "title": title,
            "type": otype,
            "index": i + 1,
            "i": i,
            "n": n,
            # Signal fields (present on SignalObj, may be absent on ImageObj)
            "xlabel": getattr(obj, "xlabel", ""),
            "xunit": getattr(obj, "xunit", ""),
            "ylabel": getattr(obj, "ylabel", ""),
            "yunit": getattr(obj, "yunit", ""),
            "metadata": getattr(obj, "metadata", {}),
        }

        try:
            formatted = CustomFormatter().format(fmt, **context)
        except KeyError as e:
            raise KeyError(f"Unknown format key in fmt: {e.args[0]!r}") from e
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Invalid format string: {e}") from e

        # Sanitize final result to ensure it's a safe filename
        final_name = sanitize_filename(formatted, replacement=replacement)
        if not final_name:
            # Fallback to a minimal safe name
            final_name = f"{title or 'untitled'}_{i + 1}"

        result.append(final_name)

    return result
