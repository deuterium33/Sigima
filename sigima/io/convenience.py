# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Convenience I/O functions

This module provides convenient wrapper functions for input/output operations with
signals and images. These functions offer a simplified interface to the underlying
I/O system, making common tasks easier to perform.
"""

from __future__ import annotations

from sigima.io.image.base import ImageIORegistry
from sigima.io.signal.base import SignalIORegistry
from sigima.objects.image import ImageObj
from sigima.objects.signal import SignalObj


def read_signals(filename: str) -> list[SignalObj]:
    """Read a list of signals from a file.

    Args:
        filename: File name.

    Returns:
        List of signals.
    """
    return SignalIORegistry.read(filename)


def read_signal(filename: str) -> SignalObj:
    """Read a signal from a file.

    Args:
        filename: File name.

    Returns:
        Signal.
    """
    return read_signals(filename)[0]


def write_signal(filename: str, signal: SignalObj) -> None:
    """Write a signal to a file.

    Args:
        filename: File name.
        signal: Signal.
    """
    SignalIORegistry.write(filename, signal)


def read_images(filename: str) -> list[ImageObj]:
    """Read a list of images from a file.

    Args:
        filename: File name.

    Returns:
        List of images.
    """
    return ImageIORegistry.read(filename)


def read_image(filename: str) -> ImageObj:
    """Read an image from a file.

    Args:
        filename: File name.

    Returns:
        Image.
    """
    return read_images(filename)[0]


def write_image(filename: str, image: ImageObj) -> None:
    """Write an image to a file.

    Args:
        filename: File name.
        image: Image.
    """
    ImageIORegistry.write(filename, image)
