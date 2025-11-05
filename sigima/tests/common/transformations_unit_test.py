# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for transformations module
"""

from __future__ import annotations

import numpy as np
import pytest

from sigima.objects.image.roi import CircularROI, PolygonalROI, RectangularROI
from sigima.objects.scalar import GeometryResult, KindShape
from sigima.objects.shape import PointCoordinates
from sigima.proc.image.transformations import GeometryTransformer, transformer


def test_geometry_transformer_singleton() -> None:
    """
    Test that GeometryTransformer follows singleton pattern.
    """
    t1 = GeometryTransformer()
    t2 = GeometryTransformer()
    assert t1 is t2
    assert t1 is transformer


class TestGeometryResultTransformations:
    """Test class for GeometryResult transformations."""

    def test_transform_point(self) -> None:
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

    def test_transform_rectangle(self) -> None:
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

    def test_transform_circle(self) -> None:
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

    def test_generic_transform_method(self) -> None:
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

    def test_unsupported_operation(self) -> None:
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

        with pytest.raises(ValueError, match="Unknown operation"):
            transformer.transform_geometry(geometry, "invalid_operation")

    def test_direct_coordinate_transformation(self) -> None:
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


class TestSingleROITransformations:
    """Test class for single ROI transformations."""

    def test_transform_rectangular_roi_rotation(self) -> None:
        """
        Test rotation transformation of RectangularROI.
        """
        # Create a rectangular ROI (x0, y0, dx, dy)
        # Rectangle corners: (0, 0) and (4, 2)
        roi = RectangularROI(coords=[0.0, 0.0, 4.0, 2.0], indices=False, title="Test")
        original_coords = roi.coords.copy()

        # Rotate 90 degrees around origin
        # After rotation: (0, 0) -> (0, 0) and (4, 2) -> (-2, 4)
        # New bounding box: x_min=-2, y_min=0, x_max=0, y_max=4
        # So: x0=-2, y0=0, dx=2, dy=4
        transformer.rotate(roi, np.pi / 2, center=(0, 0))

        expected_coords = np.array([-2.0, 0.0, 2.0, 4.0])
        assert np.allclose(roi.coords, expected_coords)
        assert not np.allclose(roi.coords, original_coords)

    def test_transform_rectangular_roi_translation(self) -> None:
        """
        Test translation transformation of RectangularROI.
        """
        roi = RectangularROI(coords=[1.0, 2.0, 3.0, 4.0], indices=False, title="Test")

        # Translate by (10, 20)
        transformer.translate(roi, 10.0, 20.0)

        expected_coords = np.array([11.0, 22.0, 3.0, 4.0])
        assert np.allclose(roi.coords, expected_coords)

    def test_transform_rectangular_roi_fliph(self) -> None:
        """
        Test horizontal flip transformation of RectangularROI.
        """
        # Create a rectangle with x0=1, dx=3, so corners at x=1 and x=4
        roi = RectangularROI(coords=[1.0, 2.0, 3.0, 4.0], indices=False, title="Test")

        # Flip horizontally around x=0
        # Corner at x=1 -> x=-1
        # Corner at x=4 -> x=-4
        # New bounding box: x_min=-4, x_max=-1, so x0=-4, dx=3
        transformer.fliph(roi, cx=0.0)

        expected_coords = np.array([-4.0, 2.0, 3.0, 4.0])
        assert np.allclose(roi.coords, expected_coords)

    def test_transform_rectangular_roi_flipv(self) -> None:
        """
        Test vertical flip transformation of RectangularROI.
        """
        # Create a rectangle with y0=2, dy=4, so corners at y=2 and y=6
        roi = RectangularROI(coords=[1.0, 2.0, 3.0, 4.0], indices=False, title="Test")

        # Flip vertically around y=0
        # Corner at y=2 -> y=-2
        # Corner at y=6 -> y=-6
        # New bounding box: y_min=-6, y_max=-2, so y0=-6, dy=4
        transformer.flipv(roi, cy=0.0)

        expected_coords = np.array([1.0, -6.0, 3.0, 4.0])
        assert np.allclose(roi.coords, expected_coords)

    def test_transform_rectangular_roi_transpose(self) -> None:
        """
        Test transpose transformation of RectangularROI.
        """
        roi = RectangularROI(coords=[1.0, 2.0, 3.0, 4.0], indices=False, title="Test")

        # Transpose (swap x and y)
        transformer.transpose(roi)

        expected_coords = np.array([2.0, 1.0, 4.0, 3.0])
        assert np.allclose(roi.coords, expected_coords)

    def test_transform_rectangular_roi_scale(self) -> None:
        """
        Test scale transformation of RectangularROI.
        """
        roi = RectangularROI(coords=[2.0, 3.0, 4.0, 6.0], indices=False, title="Test")

        # Scale by 2x in x, 3x in y around origin
        transformer.scale(roi, 2.0, 3.0, center=(0, 0))

        expected_coords = np.array([4.0, 9.0, 8.0, 18.0])
        assert np.allclose(roi.coords, expected_coords)

    def test_transform_circular_roi_rotation(self) -> None:
        """
        Test rotation transformation of CircularROI.
        """
        # Create a circular ROI (xc, yc, r)
        roi = CircularROI(coords=[3.0, 4.0, 5.0], indices=False, title="Test")

        # Rotate 90 degrees around origin
        transformer.rotate(roi, np.pi / 2, center=(0, 0))

        # Expected: center rotated, radius unchanged
        expected_coords = np.array([-4.0, 3.0, 5.0])
        assert np.allclose(roi.coords, expected_coords)

    def test_transform_circular_roi_translation(self) -> None:
        """
        Test translation transformation of CircularROI.
        """
        roi = CircularROI(coords=[1.0, 2.0, 3.0], indices=False, title="Test")

        # Translate by (10, 20)
        transformer.translate(roi, 10.0, 20.0)

        expected_coords = np.array([11.0, 22.0, 3.0])
        assert np.allclose(roi.coords, expected_coords)

    def test_transform_circular_roi_scale(self) -> None:
        """
        Test scale transformation of CircularROI.
        """
        roi = CircularROI(coords=[2.0, 3.0, 5.0], indices=False, title="Test")

        # Scale by 2x in x, 3x in y around origin
        transformer.scale(roi, 2.0, 3.0, center=(0, 0))

        # Expected: center scaled, radius unchanged
        expected_coords = np.array([4.0, 9.0, 5.0])
        assert np.allclose(roi.coords, expected_coords)

    def test_transform_polygonal_roi_rotation(self) -> None:
        """
        Test rotation transformation of PolygonalROI.
        """
        # Create a triangular ROI (x1, y1, x2, y2, x3, y3)
        roi = PolygonalROI(
            coords=[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], indices=False, title="Test"
        )

        # Rotate 90 degrees around origin
        transformer.rotate(roi, np.pi / 2, center=(0, 0))

        # Expected: all points rotated
        expected_coords = np.array([0.0, 0.0, 0.0, 1.0, -1.0, 0.0])
        assert np.allclose(roi.coords, expected_coords, atol=1e-10)

    def test_transform_polygonal_roi_translation(self) -> None:
        """
        Test translation transformation of PolygonalROI.
        """
        roi = PolygonalROI(
            coords=[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], indices=False, title="Test"
        )

        # Translate by (5, 10)
        transformer.translate(roi, 5.0, 10.0)

        expected_coords = np.array([5.0, 10.0, 6.0, 10.0, 5.0, 11.0])
        assert np.allclose(roi.coords, expected_coords)

    def test_transform_polygonal_roi_transpose(self) -> None:
        """
        Test transpose transformation of PolygonalROI.
        """
        roi = PolygonalROI(
            coords=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], indices=False, title="Test"
        )

        # Transpose (swap x and y for all points)
        transformer.transpose(roi)

        expected_coords = np.array([2.0, 1.0, 4.0, 3.0, 6.0, 5.0])
        assert np.allclose(roi.coords, expected_coords)

    def test_unsupported_roi_type(self) -> None:
        """
        Test error handling for unsupported ROI types.
        """

        # Create a mock ROI class
        class UnsupportedROI:
            """Mock unsupported ROI class."""

            def __init__(self):
                self.coords = np.array([1.0, 2.0])

        unsupported_roi = UnsupportedROI()

        with pytest.raises(ValueError, match="Unsupported ROI type"):
            transformer.transform_single_roi(unsupported_roi, "rotate", angle=0)

    def test_roi_title_preserved(self) -> None:
        """
        Test that ROI title is preserved after transformation.
        """
        roi = RectangularROI(coords=[1.0, 2.0, 3.0, 4.0], indices=False, title="MyROI")

        transformer.translate(roi, 10.0, 20.0)

        assert roi.title == "MyROI"

    def test_roi_inverse_preserved(self) -> None:
        """
        Test that ROI inverse flag is preserved after transformation.
        """
        roi = CircularROI(
            coords=[1.0, 2.0, 3.0], indices=False, title="Test", inverse=True
        )

        transformer.rotate(roi, np.pi / 4, center=(0, 0))

        assert roi.inverse is True
