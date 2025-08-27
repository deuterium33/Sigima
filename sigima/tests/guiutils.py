# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Utilities to manage GUI activation for tests executed with pytest
or as standalone scripts.

⚠️ This module must not import any Qt-related module at the top level,
    as Qt is an optional dependency of Sigima.
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator, Optional

_CURRENT_REQUEST: DummyRequest | None = None

if TYPE_CHECKING:
    # ⚠️ Type-only: no runtime Qt import
    from qtpy.QtWidgets import QApplication


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


@contextmanager
def lazy_qt_app_context(
    *, exec_loop: bool = False, force: bool | None = None
) -> Generator[Optional[QApplication], None, None]:
    """Provide a Qt app context lazily; no-op if GUI is disabled.

    Args:
        exec_loop: Run the Qt event loop (e.g. when showing a non-blocking widget).
        force: None → auto (use is_gui_enabled());
               True → force GUI ON (always create Qt app);
               False → force GUI OFF (no-op).

    Yields:
        The QApplication instance if enabled, else None.

    .. note::

       This context manager is useful for tests that require a Qt application context,
       but should be used with caution to avoid unnecessary Qt imports. For tests
       that are exclusively GUI-based, option `force=True` can be used to ensure
       the Qt application context is always created. For tests that must be executable
       without a GUI, option `force` may be skipped so that operations inside the
       context are only performed if the GUI is enabled.
    """
    enabled = is_gui_enabled() if force is None else force
    if not enabled:
        # No Qt import, block executes as a no-op context
        yield None
        return

    # Lazy import: only when enabled
    # pylint: disable=import-outside-toplevel
    from guidata.qthelpers import qt_app_context

    with qt_app_context(exec_loop=exec_loop) as qt_app:
        yield qt_app


def view_images_side_by_side_if_gui_enabled(*args, **kwargs) -> None:
    """Display images side-by-side with PlotPy if GUI mode enabled.

    Forwards all arguments to :py:func:`sigima.tests.vistools.view_images_side_by_side`.

    Args:
        *args: Named arguments.
        **kwargs: Keyword arguments.
    """
    with lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            from sigima.tests import vistools  # pylint: disable=import-outside-toplevel

            vistools.view_images_side_by_side(*args, **kwargs)


def view_curves_and_images_if_gui_enabled(*args, **kwargs) -> None:
    """Display signals and images with PlotPy if GUI mode enabled.

    Forwards all arguments to :py:func:`sigima.tests.vistools.view_curves_and_images`.

    Args:
        *args: Named arguments.
        **kwargs: Keyword arguments.
    """
    with lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            from sigima.tests import vistools  # pylint: disable=import-outside-toplevel

            vistools.view_curves_and_images(*args, **kwargs)


def view_curves_if_gui_enabled(*args, **kwargs) -> None:
    """Display one or more signals with PlotPy if GUI mode enabled.

    Forwards all arguments to :py:func:`sigima.tests.vistools.view_curves`.

    Args:
        *args: Named arguments.
        **kwargs: Keyword arguments.
    """
    with lazy_qt_app_context() as qt_app:
        if qt_app is not None:
            from sigima.tests import vistools  # pylint: disable=import-outside-toplevel

            vistools.view_curves(*args, **kwargs)
