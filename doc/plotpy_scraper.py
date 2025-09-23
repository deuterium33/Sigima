# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""PlotPy scraper for Sphinx-Gallery.

This module provides a scraper that can capture PlotPy plot outputs and convert them
to images for use in Sphinx-Gallery documentation. The scraper automatically detects
PlotPy plot dialogs and captures screenshots for documentation purposes.

The scraper works by:
1. Detecting when PlotPy dialogs are created during example execution
2. Capturing screenshots of these dialogs
3. Converting the screenshots to appropriate formats for Sphinx-Gallery
4. Saving the images in the expected location for gallery generation

Usage:
    This module is automatically used by Sphinx-Gallery when configured in conf.py.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

try:
    from plotpy.plot import (
        PlotDialog,
        PlotWidget,
        PlotWindow,
        SubplotWidget,
        SyncPlotDialog,
    )

    PLOTPY_AVAILABLE = True
except ImportError:
    PLOTPY_AVAILABLE = False

try:
    from qtpy.QtWidgets import QApplication

    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


def _find_plotpy_figures() -> list[Any]:
    """Find all open PlotPy plot dialogs.

    Returns:
        List of visible PlotPy widgets.
    """
    if not PLOTPY_AVAILABLE:
        return []

    try:
        app = QApplication.instance()
        if app is None:
            return []

        figures = []
        for widget in app.topLevelWidgets():
            if (
                isinstance(
                    widget,
                    (
                        PlotDialog,
                        PlotWidget,
                        PlotWindow,
                        SubplotWidget,
                        SyncPlotDialog,
                    ),
                )
                and widget.isVisible()
            ):
                figures.append(widget)
        return figures
    except (AttributeError, RuntimeError):
        # Qt not initialized or other error
        return []


def _capture_figure(figure: Any, output_path: str | Path) -> bool:
    """Capture a screenshot of a PlotPy figure and save it.

    Args:
        figure: The PlotPy widget to capture.
        output_path: Path where to save the PNG screenshot.

    Returns:
        True if capture was successful, False otherwise.
    """
    if not hasattr(figure, "grab"):
        return False

    try:
        # Make sure the figure is visible and rendered
        figure.show()
        figure.raise_()
        figure.activateWindow()

        # Process events to ensure the figure is fully rendered
        app = QApplication.instance()
        if app:
            app.processEvents()

        # Capture the screenshot
        pixmap = figure.grab()
        if not pixmap.isNull():
            success = pixmap.save(output_path, "PNG")
            return success
    except Exception as e:
        print(f"Warning: Failed to capture PlotPy figure: {e}")
        return False

    return False


def plotpy_scraper(block, block_vars, gallery_conf, **kwargs):
    """Scraper for PlotPy figures in Sphinx-Gallery.

    This function is called by Sphinx-Gallery after executing each code block
    to capture any PlotPy figures that were created.

    Args:
        _block: The code block that was executed (unused).
        _block_vars: Variables from the executed code block (unused).
        gallery_conf: Sphinx-Gallery configuration.
        **_kwargs: Additional arguments (unused).

    Returns:
        RST code to include the captured images.
    """
    if not PLOTPY_AVAILABLE or not QT_AVAILABLE:
        print("PlotPy or Qt not available for scraping")
        return ""

    # Set environment variable to indicate we're building gallery
    os.environ["SPHINX_GALLERY_BUILDING"] = "1"

    # Ensure QApplication exists before trying to find figures
    try:
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            # Create minimal QApplication for gallery building
            app = QApplication([])
            print("Created QApplication for PlotPy scraper")
    except Exception as e:
        print(f"Could not initialize QApplication: {e}")
        return ""

    # Find all PlotPy figures
    figures = _find_plotpy_figures()
    print(f"Found {len(figures)} PlotPy figures")

    if not figures:
        return ""

    # Generate our own image paths since Sphinx-Gallery doesn't pass the iterator
    # We need to create unique image names for this example
    from pathlib import Path

    # Use the gallery configuration to get the correct path
    if gallery_conf and "src_dir" in gallery_conf and "gallery_dirs" in gallery_conf:
        src_dir = Path(gallery_conf["src_dir"])
        gallery_dirs = gallery_conf["gallery_dirs"]

        if gallery_dirs:
            # Handle both string and list cases for gallery_dirs
            if isinstance(gallery_dirs, list):
                gallery_dir = gallery_dirs[0]  # e.g., 'auto_examples'
            else:
                gallery_dir = gallery_dirs  # Already a string
            img_dir = src_dir / gallery_dir / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
        else:
            print("Warning: No gallery_dirs found in configuration")
            return ""
    else:
        print("Warning: Cannot determine proper image directory from gallery config")
        return ""

    if not figures:
        return ""

    timestamp = int(time.time() * 1000)  # milliseconds for uniqueness

    # Capture each figure
    rst_blocks = []
    for i, figure in enumerate(figures):
        try:
            # Generate unique image path
            image_name = f"sphx_glr_plotpy_{timestamp}_{i:03d}.png"
            image_path = img_dir / image_name

            print(
                f"Attempting to capture figure {i + 1}/{len(figures)} to {image_path}"
            )

            # Capture the figure
            success = _capture_figure(figure, str(image_path))

            if success:
                # Generate RST code to include the image (relative to the gallery dir)
                gal_dirs = gallery_conf.get("gallery_dirs", ["auto_examples"])
                if isinstance(gal_dirs, list):
                    rst_gallery_dir = gal_dirs[0] if gal_dirs else "auto_examples"
                else:
                    rst_gallery_dir = gal_dirs
                rst_blocks.append(
                    f"""
.. image:: /{rst_gallery_dir}/images/{image_name}
   :alt: PlotPy figure {i + 1}
   :class: sphx-glr-single-img
"""
                )
                print(f"Successfully captured figure {i + 1}")
            else:
                print(f"Failed to capture figure {i + 1}")

            # Close the figure to prevent accumulation
            if hasattr(figure, "close"):
                figure.close()

        except Exception as e:
            print(f"Warning: Failed to process PlotPy figure {i}: {e}")
            continue

    return "".join(rst_blocks)


def _get_plotpy_version() -> str | None:
    """Get PlotPy version if available.

    Returns:
        PlotPy version string or None if not available.
    """
    if not PLOTPY_AVAILABLE:
        return None

    try:
        import plotpy

        return getattr(plotpy, "__version__", "unknown")
    except ImportError:
        return None


def setup_plotpy_scraper(_app: Any, config: Any) -> None:  # noqa: ARG001
    """Setup function to register the PlotPy scraper with Sphinx.

    Args:
        _app: Sphinx application instance (unused).
        config: Sphinx configuration object.
    """
    if hasattr(config, "sphinx_gallery_conf"):
        scrapers = config.sphinx_gallery_conf.get("image_scrapers", [])
        if plotpy_scraper not in scrapers:
            scrapers.append(plotpy_scraper)
            config.sphinx_gallery_conf["image_scrapers"] = scrapers


def get_plotpy_scraper() -> Any:
    """Return the PlotPy scraper function for use in Sphinx-Gallery configuration.

    Returns:
        The plotpy_scraper function.
    """
    return plotpy_scraper


def get_plotpy_scraper_config() -> dict[str, Any]:
    """Return a configuration dict for PlotPy scraper.

    Returns:
        Configuration dictionary for Sphinx-Gallery.
    """
    config = {
        "image_scrapers": [plotpy_scraper],
        "reset_modules": ("plotpy", "sigima"),  # Reset modules between examples
        "remove_config_comments_from_code": False,
        "expected_failing_examples": [],
    }

    # Add PlotPy version info if available
    version = _get_plotpy_version()
    if version:
        config["plotpy_version"] = version

    return config
