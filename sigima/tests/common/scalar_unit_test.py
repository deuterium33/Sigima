# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for scalar computation functions (GeometryResult transformations).
"""

from __future__ import annotations

import numpy as np

from sigima.objects.scalar import GeometryResult
from sigima.proc.scalar import (
    fliph,
    flipv,
    rotate,
    scale,
    scale_1d,
    translate,
    translate_1d,
    transpose,
)


def create_rectangle(x0=0.0, y0=0.0, w=1.0, h=1.0) -> GeometryResult:
    """Create a simple rectangle GeometryResult."""
    coords = np.array([[x0, y0, x0 + w, y0 + h]], dtype=float)
    return GeometryResult("rect", "rectangle", coords)


def test_rotate() -> None:
    """Test rotation of a rectangle geometry result."""
    rect = create_rectangle()
    rotated = rotate(rect, np.pi / 2)
    assert rotated.coords.shape == rect.coords.shape
    assert not np.allclose(rotated.coords, rect.coords)


def test_fliph() -> None:
    """Test horizontal flip and its reversibility."""
    rect = create_rectangle(1.0, 2.0, 2.0, 3.0)
    flipped = fliph(rect, cx=2.0)
    flipped_back = fliph(flipped, cx=2.0)
    np.testing.assert_allclose(flipped_back.coords, rect.coords, rtol=1e-12)


def test_flipv() -> None:
    """Test vertical flip and its reversibility."""
    rect = create_rectangle(1.0, 2.0, 2.0, 3.0)
    flipped = flipv(rect, cy=3.5)
    flipped_back = flipv(flipped, cy=3.5)
    np.testing.assert_allclose(flipped_back.coords, rect.coords, rtol=1e-12)


def test_translate() -> None:
    """Test translation of a geometry result."""
    rect = create_rectangle()
    translated = translate(rect, 1.5, -2.0)
    expected = rect.coords + np.array([1.5, -2.0, 1.5, -2.0])
    np.testing.assert_allclose(translated.coords, expected, rtol=1e-12)


def test_scale() -> None:
    """Test scaling and inverse scaling of a geometry result."""
    rect = create_rectangle(1.0, 1.0, 2.0, 2.0)
    scaled = scale(rect, 2.0, 0.5, center=(2.0, 2.0))
    unscaled = scale(scaled, 0.5, 2.0, center=(2.0, 2.0))
    np.testing.assert_allclose(unscaled.coords, rect.coords, rtol=1e-12)


def test_transpose() -> None:
    """Test transpose and double-transpose (should restore original)."""
    rect = create_rectangle(1.0, 2.0, 3.0, 4.0)
    transposed = transpose(rect)
    transposed_back = transpose(transposed)
    np.testing.assert_allclose(transposed_back.coords, rect.coords, rtol=1e-12)
