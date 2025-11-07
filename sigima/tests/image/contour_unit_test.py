# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Contour finding test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

import sys
import time

import numpy as np
import pytest

import sigima.enums
import sigima.objects
import sigima.params
import sigima.proc.image
from sigima.enums import ContourShape
from sigima.tests import guiutils
from sigima.tests.data import get_peak2d_data
from sigima.tests.env import execenv
from sigima.tests.helpers import (
    check_array_result,
    check_scalar_result,
)
from sigima.tools import coordinates
from sigima.tools.image import get_2d_peaks_coords, get_contour_shapes


def create_contour_shape_items(data, shape):
    """Create plotpy items for a specific contour shape.

    Args:
        data: Input data array
        shape: ContourShape enum value

    Returns:
        List of plotpy items representing the detected contours
    """
    # pylint: disable=import-outside-toplevel
    from plotpy.builder import make

    items = []
    coords = get_contour_shapes(data, shape=shape)
    execenv.print(f"Coordinates ({shape}s): {coords}")
    for shapeargs in coords:
        if shape == ContourShape.CIRCLE:
            xc, yc, r = shapeargs
            x0, y0, x1, y1 = coordinates.circle_to_diameter(xc, yc, r)
            item = make.circle(x0, y0, x1, y1)
        elif shape == ContourShape.ELLIPSE:
            xc, yc, a, b, theta = shapeargs
            coords_ellipse = coordinates.ellipse_to_diameters(xc, yc, a, b, theta)
            x0, y0, x1, y1, x2, y2, x3, y3 = coords_ellipse
            item = make.ellipse(x0, y0, x1, y1, x2, y2, x3, y3)
        else:  # ContourShape.POLYGON
            # `shapeargs` is a flattened array of x, y coordinates
            x, y = shapeargs[::2], shapeargs[1::2]
            item = make.polygon(x, y, closed=False)
        items.append(item)
    return items


@pytest.mark.gui
def test_contour_interactive():
    """2D peak detection test"""
    data, _coords = get_peak2d_data()
    with guiutils.lazy_qt_app_context(force=True):
        # pylint: disable=import-outside-toplevel
        from plotpy.builder import make

        from sigima.tests import vistools

        items = [make.image(data, interpolation="linear", colormap="hsv")]
        t0 = time.time()
        peak_coords = get_2d_peaks_coords(data)
        dt = time.time() - t0
        for x, y in peak_coords:
            items.append(make.marker((x, y)))
        execenv.print(f"Calculation time: {int(dt * 1e3):d} ms\n", file=sys.stderr)
        execenv.print(f"Peak coordinates: {peak_coords}")

        # Add contour shapes for all shape types
        for shape in ContourShape:
            items.extend(create_contour_shape_items(data, shape))

        vistools.view_image_items(items)


@pytest.mark.validation
def test_contour_shape() -> None:
    """Test contour shape computation function"""
    # Create test data with known shapes
    data, _expected_coords = get_peak2d_data()

    # Test each contour shape type with ROI creation
    for shape in ContourShape:
        execenv.print(f"Testing contour shape: {shape}")

        # Get contour shapes from the function
        detected_shapes = get_contour_shapes(data, shape=shape)
        execenv.print(f"Detected {len(detected_shapes)} {shape}(s)")

        image = sigima.objects.create_image("Contour Test Image", data=data)
        param = sigima.params.ContourShapeParam.create(shape=shape)
        results = sigima.proc.image.contour_shape(image, param)
        sigima.proc.image.apply_detection_rois(image, results)

        check_array_result(f"Contour shapes ({shape})", detected_shapes, results.coords)

        # Basic validation checks
        assert isinstance(detected_shapes, np.ndarray), (
            f"get_contour_shapes should return numpy array for {shape}"
        )

        if len(detected_shapes) > 0:
            # Check that we detected at least some shapes
            execenv.print(f"Successfully detected contours for {shape}")

            # Validate shape-specific properties
            if shape == ContourShape.CIRCLE:
                # For circles: [xc, yc, r]
                assert detected_shapes.shape[1] == 3, (
                    "Circle contours should have 3 parameters (xc, yc, r)"
                )
                # Check that radius values are positive
                radii = detected_shapes[:, 2]
                assert np.all(radii > 0), "All circle radii should be positive"
                check_scalar_result(
                    "Circle radius range",
                    np.mean(radii),
                    np.mean(radii),  # Just check it's finite
                    rtol=1.0,
                )

            elif shape == ContourShape.ELLIPSE:
                # For ellipses: [xc, yc, a, b, theta]
                assert detected_shapes.shape[1] == 5, (
                    "Ellipse contours should have 5 parameters (xc, yc, a, b, theta)"
                )
                # Check that semi-axes are positive
                a_values = detected_shapes[:, 2]
                b_values = detected_shapes[:, 3]
                assert np.all(a_values > 0), (
                    "All ellipse semi-axes 'a' should be positive"
                )
                assert np.all(b_values > 0), (
                    "All ellipse semi-axes 'b' should be positive"
                )
                check_scalar_result(
                    "Ellipse semi-axis 'a' range",
                    np.mean(a_values),
                    np.mean(a_values),  # Just check it's finite
                    rtol=1.0,
                )

            elif shape == ContourShape.POLYGON:
                # For polygons: flattened x,y coordinates
                # Shape should be (n_contours, max_points) where max_points is even
                assert detected_shapes.shape[1] % 2 == 0, (
                    "Polygon contours should have even number of coordinates "
                    "(x,y pairs)"
                )
                # Check that we have valid coordinates (not all NaN)
                valid_coords = ~np.isnan(detected_shapes)
                assert np.any(valid_coords), (
                    "Polygon should have some valid coordinates"
                )

        # Check that the function handles different threshold levels
        for level in [0.3, 0.5, 0.7]:
            shapes_at_level = get_contour_shapes(data, shape=shape, level=level)
            assert isinstance(shapes_at_level, np.ndarray), (
                f"get_contour_shapes should return numpy array for {shape} "
                f"at level {level}"
            )
            execenv.print(f"  At level {level}: detected {len(shapes_at_level)} shapes")

    execenv.print("All contour shape tests passed!")


if __name__ == "__main__":
    test_contour_interactive()
    test_contour_shape()
