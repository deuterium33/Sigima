# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Title formatting system for computation results
-----------------------------------------------

This module provides a configurable title formatting system for computation results.
It allows different applications (Sigima vs DataLab) to use different title formatting
strategies while maintaining compatibility.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = [
    "PlaceholderTitleFormatter",
    "SimpleTitleFormatter",
    "TitleFormatter",
    "get_default_title_formatter",
    "set_default_title_formatter",
]


@runtime_checkable
class TitleFormatter(Protocol):
    """Protocol for title formatting strategies used in computation functions.

    This protocol allows different title formatting approaches:
    - Simple descriptive titles for Sigima-only usage
    - Placeholder-based titles for DataLab integration
    - Custom formatting for specific use cases
    """

    def format_1_to_1_title(self, name: str, suffix: str | None = None) -> str:
        """Format title for 1-to-1 computation (single input → single output).

        Args:
            name: Name of the computation function
            suffix: Optional suffix to add to the title

        Returns:
            Formatted title string
        """

    def format_n_to_1_title(
        self, name: str, n_inputs: int, suffix: str | None = None
    ) -> str:
        """Format title for n-to-1 computation (multiple inputs → single output).

        Args:
            name: Name of the computation function
            n_inputs: Number of input objects
            suffix: Optional suffix to add to the title

        Returns:
            Formatted title string
        """

    def format_2_to_1_title(self, name: str, suffix: str | None = None) -> str:
        """Format title for 2-to-1 computation (two inputs → single output).

        Args:
            name: Name of the computation function
            suffix: Optional suffix to add to the title

        Returns:
            Formatted title string
        """


class SimpleTitleFormatter:
    """Simple descriptive title formatter for Sigima-only usage.

    Creates human-readable titles without placeholders, suitable for
    standalone Sigima usage where object relationships are less critical.
    """

    def format_1_to_1_title(self, name: str, suffix: str | None = None) -> str:
        """Format title for 1-to-1 computation."""
        # Convert function names to human-readable format
        readable_name = name.replace("_", " ").title()
        base_title = f"{readable_name} Result"

        if suffix:
            base_title += f" ({suffix})"
        return base_title

    def format_n_to_1_title(
        self, name: str, n_inputs: int, suffix: str | None = None
    ) -> str:
        """Format title for n-to-1 computation."""
        readable_name = name.replace("_", " ").title()
        base_title = f"{readable_name} of {n_inputs} Objects"

        if suffix:
            base_title += f" ({suffix})"
        return base_title

    def format_2_to_1_title(self, name: str, suffix: str | None = None) -> str:
        """Format title for 2-to-1 computation."""
        if len(name) == 1:  # This is an operator
            base_title = f"Binary Operation {name}"
        else:
            readable_name = name.replace("_", " ").title()
            base_title = f"{readable_name} Result"

        if suffix:
            base_title += f" ({suffix})"
        return base_title


class PlaceholderTitleFormatter:
    """Placeholder-based title formatter compatible with DataLab.

    Creates titles with placeholders that can be resolved later by DataLab's
    patch_title_with_ids() function.
    """

    def format_1_to_1_title(self, name: str, suffix: str | None = None) -> str:
        """Format title for 1-to-1 computation with placeholder."""
        title = f"{name}({{0}})"
        if suffix:
            title += "|" + suffix
        return title

    def format_n_to_1_title(
        self, name: str, n_inputs: int, suffix: str | None = None
    ) -> str:
        """Format title for n-to-1 computation with placeholders."""
        placeholders = ", ".join(f"{{{i}}}" for i in range(n_inputs))
        title = f"{name}({placeholders})"
        if suffix:
            title += "|" + suffix
        return title

    def format_2_to_1_title(self, name: str, suffix: str | None = None) -> str:
        """Format title for 2-to-1 computation with placeholders."""
        if len(name) == 1:  # This is an operator
            title = f"{{0}}{name}{{1}}"
        else:
            title = f"{name}({{0}}, {{1}})"
        if suffix:
            title += "|" + suffix
        return title


# Global default title formatter
_default_title_formatter: TitleFormatter = SimpleTitleFormatter()


def get_default_title_formatter() -> TitleFormatter:
    """Get the current default title formatter.

    Returns:
        Current default title formatter
    """
    return _default_title_formatter


def set_default_title_formatter(formatter: TitleFormatter) -> None:
    """Set the default title formatter.

    Args:
        formatter: Title formatter to use as default
    """
    global _default_title_formatter  # pylint: disable=global-statement
    _default_title_formatter = formatter
