# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for transformations module
"""

from __future__ import annotations

import numpy as np

from sigima.objects.scalar import GeometryResult, KindShape
from sigima.objects.shape import PointCoordinates
from sigima.proc.transformations import GeometryTransformer, transformer


def test_geometry_transformer_singleton() -> None:
    """
    Test that GeometryTransformer follows singleton pattern.
    """
    t1 = GeometryTransformer()
    t2 = GeometryTransformer()
    assert t1 is t2
    assert t1 is transformer


def test_transform_geometry_result_point() -> None:
    """
    Test transformation of GeometryResult with POINT coordinates.
    """
    # Create a GeometryResult with point coordinates
    coords = np.array([[1.0, 2.0], [3.0, 4.0]])
    geometry = GeometryResult(
        title="Test Points",
        kind=KindShape.POINT,
        coords=coords,
        roi_indices=None,
        attrs={},
    )

    # Test rotation
    rotated = transformer.rotate(geometry, np.pi / 2, center=(0, 0))
    expected_coords = np.array([[-2.0, 1.0], [-4.0, 3.0]])
    assert np.allclose(rotated.coords, expected_coords)
    assert rotated.title == geometry.title
    assert rotated.kind == geometry.kind

    # Test translation
    translated = transformer.translate(geometry, 10.0, 20.0)
    expected_coords = np.array([[11.0, 22.0], [13.0, 24.0]])
    assert np.allclose(translated.coords, expected_coords)

    # Original should be unchanged
    assert np.allclose(geometry.coords, coords)


def test_transform_geometry_result_rectangle() -> None:
    """
    Test transformation of GeometryResult with RECTANGLE coordinates.
    """
    # Create a GeometryResult with rectangle coordinates (x0, y0, dx, dy)
    coords = np.array([[0.0, 0.0, 3.0, 1.0], [10.0, 10.0, 2.0, 4.0]])
    geometry = GeometryResult(
        title="Test Rectangles",
        kind=KindShape.RECTANGLE,
        coords=coords,
        roi_indices=None,
        attrs={},
    )

    # Test horizontal flip around x=1
    flipped = transformer.fliph(geometry, cx=1.0)
    expected_coords = np.array([[-1.0, 0.0, 3.0, 1.0], [-10.0, 10.0, 2.0, 4.0]])
    assert np.allclose(flipped.coords, expected_coords)

    # Test transpose
    transposed = transformer.transpose(geometry)
    expected_coords = np.array([[0.0, 0.0, 1.0, 3.0], [10.0, 10.0, 4.0, 2.0]])
    assert np.allclose(transposed.coords, expected_coords)


def test_transform_geometry_result_circle() -> None:
    """
    Test transformation of GeometryResult with CIRCLE coordinates.
    """
    # Create a GeometryResult with circle coordinates
    coords = np.array([[1.0, 2.0, 5.0], [10.0, 20.0, 10.0]])
    geometry = GeometryResult(
        title="Test Circles",
        kind=KindShape.CIRCLE,
        coords=coords,
        roi_indices=None,
        attrs={},
    )

    # Test scaling (only center should be scaled, radius unchanged)
    scaled = transformer.scale(geometry, 2.0, 3.0, center=(0, 0))
    expected_coords = np.array([[2.0, 6.0, 5.0], [20.0, 60.0, 10.0]])
    assert np.allclose(scaled.coords, expected_coords)


def test_geometry_transformer_generic_methods() -> None:
    """
    Test generic transform_geometry method.
    """
    coords = np.array([[1.0, 2.0]])
    geometry = GeometryResult(
        title="Test Point",
        kind=KindShape.POINT,
        coords=coords,
        roi_indices=None,
        attrs={},
    )

    # Test generic method
    rotated = transformer.transform_geometry(
        geometry, "rotate", angle=np.pi / 2, center=(0, 0)
    )
    expected_coords = np.array([[-2.0, 1.0]])
    assert np.allclose(rotated.coords, expected_coords)


def test_unsupported_geometry_kind() -> None:
    """
    Test error handling for unsupported geometry kinds.
    """
    # Create invalid geometry kind (this would need to be added to KindShape enum)
    try:
        # This should raise an error if we had an unsupported kind
        # For now, all defined kinds are supported
        pass
    except ValueError:
        pass


def test_unsupported_operation() -> None:
    """
    Test error handling for unsupported operations.
    """
    coords = np.array([[1.0, 2.0]])
    geometry = GeometryResult(
        title="Test Point",
        kind=KindShape.POINT,
        coords=coords,
        roi_indices=None,
        attrs={},
    )

    try:
        transformer.transform_geometry(geometry, "invalid_operation")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown operation" in str(e)


def test_direct_coordinate_transformation() -> None:
    """
    Test that transformations use the shape coordinate system correctly.
    """
    # Test that our transformer produces the same results as direct shape
    # coordinate usage
    coords = np.array([[2.0, 3.0]])
    geometry = GeometryResult(
        title="Test Point",
        kind=KindShape.POINT,
        coords=coords,
        roi_indices=None,
        attrs={},
    )

    # Transform using transformer
    rotated_geometry = transformer.rotate(geometry, np.pi / 2, center=(1, 1))

    # Transform using shape coordinates directly
    shape_coords = PointCoordinates([2.0, 3.0])
    shape_coords.rotate(np.pi / 2, center=(1, 1))

    # Results should be identical
    assert np.allclose(rotated_geometry.coords[0], shape_coords.data)
