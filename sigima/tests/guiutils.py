# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Utilities to manage GUI activation for tests executed with pytest
or as standalone scripts.
"""

from __future__ import annotations

import types

_CURRENT_REQUEST: DummyRequest | None = None


def enable_gui(state: bool = True) -> None:
    """Enable or disable GUI mode.

    Args:
        state: Whether to enable or disable GUI mode.
    """
    global _CURRENT_REQUEST  # pylint: disable=global-statement
    _CURRENT_REQUEST = DummyRequest(state)


def is_gui_enabled() -> bool:
    """
    Return True if GUI mode is enabled (i.e. pytest was run with --gui),
    or if a DummyRequest with --gui was set (for __main__ execution).
    """
    return bool(_CURRENT_REQUEST and _CURRENT_REQUEST.config.getoption("--gui"))


class DummyRequest:
    """
    Dummy request object to simulate pytest --gui when running a test manually.

    Example usage:
        test_x(request=DummyRequest(gui=True))
    """

    def __init__(self, gui: bool = True):
        self.config = types.SimpleNamespace()
        self.config.getoption = lambda name: gui if name == "--gui" else None


def view_images_side_by_side_if_gui_enabled(*args, **kwargs) -> None:
    """
    Display images side-by-side with PlotPy if GUI mode enabled.

    Forwards all arguments to :py:func:`sigima.tests.vistools.view_images_side_by_side`.

    Args:
        *args: Named arguments.
        **kwargs: Keyword arguments.
    """
    if is_gui_enabled():
        # pylint: disable=import-outside-toplevel
        from guidata.qthelpers import qt_app_context

        from sigima.tests.vistools import view_images_side_by_side

        with qt_app_context():
            view_images_side_by_side(*args, **kwargs)


def view_signals_and_images_if_gui_enabled(*args, **kwargs) -> None:
    """
    Display signals and images with PlotPy if GUI mode enabled.

    Forwards all arguments to :py:func:`sigima.tests.vistools.view_curves_and_images`.

    Args:
        *args: Named arguments.
        **kwargs: Keyword arguments.
    """
    if is_gui_enabled():
        # pylint: disable=import-outside-toplevel
        from guidata.qthelpers import qt_app_context

        from sigima.tests.vistools import view_curves_and_images

        with qt_app_context():
            view_curves_and_images(*args, **kwargs)


def view_signals_if_gui_enabled(*args, **kwargs) -> None:
    """
    Display one or more signals with PlotPy if GUI mode enabled.

    Forwards all arguments to :py:func:`sigima.tests.vistools.view_curves`.

    Args:
        *args: Named arguments.
        **kwargs: Keyword arguments.
    """
    if is_gui_enabled():
        # pylint: disable=import-outside-toplevel
        from guidata.qthelpers import qt_app_context

        from sigima.tests.vistools import view_curves

        with qt_app_context():
            view_curves(*args, **kwargs)
