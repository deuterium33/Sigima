# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Coordinates Algorithms (see parent package :mod:`sigima.tools`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np


def circle_to_diameter(
    xc: float, yc: float, r: float
) -> tuple[float, float, float, float]:
    """Convert circle center and radius to X diameter coordinates

    Args:
        xc: Circle center X coordinate
        yc: Circle center Y coordinate
        r: Circle radius

    Returns:
        tuple: Circle X diameter coordinates
    """
    return xc - r, yc, xc + r, yc


def array_circle_to_diameter(data: np.ndarray) -> np.ndarray:
    """Convert circle center and radius to X diameter coordinates (array version)

    Args:
        data: Circle center and radius, in the form of a 2D array (N, 3)

    Returns:
        Circle X diameter coordinates, in the form of a 2D array (N, 4)
    """
    xc, yc, r = data[:, 0], data[:, 1], data[:, 2]
    x_start = xc - r
    x_end = xc + r
    result = np.column_stack((x_start, yc, x_end, yc)).astype(float)
    return result


def circle_to_center_radius(
    x0: float, y0: float, x1: float, y1: float
) -> tuple[float, float, float]:
    """Convert circle X diameter coordinates to center and radius

    Args:
        x0: Diameter start X coordinate
        y0: Diameter start Y coordinate
        x1: Diameter end X coordinate
        y1: Diameter end Y coordinate

    Returns:
        tuple: Circle center and radius
    """
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    r = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / 2
    return xc, yc, r


def array_circle_to_center_radius(data: np.ndarray) -> np.ndarray:
    """Convert circle X diameter coordinates to center and radius (array version)

    Args:
        data: Circle X diameter coordinates, in the form of a 2D array (N, 4)

    Returns:
        Circle center and radius, in the form of a 2D array (N, 3)
    """
    x0, y0, x1, y1 = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    r = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / 2
    result = np.column_stack((xc, yc, r)).astype(float)
    return result


def ellipse_to_diameters(
    xc: float, yc: float, a: float, b: float, theta: float
) -> tuple[float, float, float, float, float, float, float, float]:
    """Convert ellipse center, axes and angle to X/Y diameters coordinates

    Args:
        xc: Ellipse center X coordinate
        yc: Ellipse center Y coordinate
        a: Ellipse half larger axis
        b: Ellipse half smaller axis
        theta: Ellipse angle

    Returns:
        Ellipse X/Y diameters (major/minor axes) coordinates
    """
    dxa, dya = a * np.cos(theta), a * np.sin(theta)
    dxb, dyb = b * np.sin(theta), b * np.cos(theta)
    x0, y0, x1, y1 = xc - dxa, yc - dya, xc + dxa, yc + dya
    x2, y2, x3, y3 = xc - dxb, yc - dyb, xc + dxb, yc + dyb
    return x0, y0, x1, y1, x2, y2, x3, y3


def array_ellipse_to_diameters(data: np.ndarray) -> np.ndarray:
    """Convert ellipse center, axes and angle to X/Y diameters coordinates
    (array version)

    Args:
        data: Ellipse center, axes and angle, in the form of a 2D array (N, 5)

    Returns:
        Ellipse X/Y diameters (major/minor axes) coordinates,
         in the form of a 2D array (N, 8)
    """
    xc, yc, a, b, theta = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
    dxa, dya = a * np.cos(theta), a * np.sin(theta)
    dxb, dyb = b * np.sin(theta), b * np.cos(theta)
    x0, y0, x1, y1 = xc - dxa, yc - dya, xc + dxa, yc + dya
    x2, y2, x3, y3 = xc - dxb, yc - dyb, xc + dxb, yc + dyb
    result = np.column_stack((x0, y0, x1, y1, x2, y2, x3, y3)).astype(float)
    return result


def ellipse_to_center_axes_angle(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
) -> tuple[float, float, float, float, float]:
    """Convert ellipse X/Y diameters coordinates to center, axes and angle

    Args:
        x0: major axis start X coordinate
        y0: major axis start Y coordinate
        x1: major axis end X coordinate
        y1: major axis end Y coordinate
        x2: minor axis start X coordinate
        y2: minor axis start Y coordinate
        x3: minor axis end X coordinate
        y3: minor axis end Y coordinate

    Returns:
        Ellipse center, axes and angle
    """
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    a = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / 2
    b = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2) / 2
    theta = np.arctan2(y1 - y0, x1 - x0)
    return xc, yc, a, b, theta


def array_ellipse_to_center_axes_angle(data: np.ndarray) -> np.ndarray:
    """Convert ellipse X/Y diameters coordinates to center, axes and angle
    (array version)

    Args:
        data: Ellipse X/Y diameters coordinates, in the form of a 2D array (N, 8)

    Returns:
        Ellipse center, axes and angle, in the form of a 2D array (N, 5)
    """
    x0, y0, x1, y1, x2, y2, x3, y3 = (
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
        data[:, 4],
        data[:, 5],
        data[:, 6],
        data[:, 7],
    )
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    a = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / 2
    b = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2) / 2
    theta = np.arctan2(y1 - y0, x1 - x0)
    result = np.column_stack((xc, yc, a, b, theta)).astype(float)
    return result


def to_polar(
    x: np.ndarray, y: np.ndarray, unit: Literal["rad", "deg"] = "rad"
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian coordinates to polar coordinates.

    Args:
        x: Cartesian x-coordinate.
        y: Cartesian y-coordinate.
        unit: Unit of the angle ('rad' or 'deg').

    Returns:
        Polar coordinates (r, theta) where r is the radius and theta is the angle.
    """
    assert x.shape == y.shape, "x and y must have the same shape"
    assert unit in ["rad", "deg"], "unit must be 'rad' or 'deg'"
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    if unit == "deg":
        theta = np.rad2deg(theta)
    return r, theta


def to_cartesian(
    r: np.ndarray, theta: np.ndarray, unit: Literal["rad", "deg"] = "rad"
) -> tuple[np.ndarray, np.ndarray]:
    """Convert polar coordinates to Cartesian coordinates.

    Args:
        r: Polar radius.
        theta: Polar angle.
        unit: Unit of the angle ('rad' or 'deg').

    Returns:
        Cartesian coordinates (x, y) where x is the x-coordinate and y is the
        y-coordinate.

    .. note::

        Negative radius values are not supported. They will be set to 0.
    """
    assert r.shape == theta.shape, "r and theta must have the same shape"
    assert unit in ["rad", "deg"], "unit must be 'rad' or 'deg'"
    if np.any(r < 0):
        warnings.warn(
            "Negative radius values are not supported. They will be set to 0."
        )
        r = np.maximum(r, 0)
    if unit == "deg":
        theta = np.deg2rad(theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def rotate(alpha: float) -> np.ndarray:
    """Return rotation matrix

    Args:
        alpha: Rotation angle (in radians)

    Returns:
        Rotation matrix
    """
    return create_rotation_matrix(alpha)


def colvector(x: float, y: float) -> np.ndarray:
    """Return vector from coordinates

    Args:
        x: x-coordinate
        y: y-coordinate

    Returns:
        Vector
    """
    return np.array([x, y, 1]).T


def vector_rotation(theta: float, dx: float, dy: float) -> tuple[float, float]:
    """Compute theta-rotation on vector

    Args:
        theta: Rotation angle
        dx: x-coordinate of vector
        dy: y-coordinate of vector

    Returns:
        Tuple of (x, y) coordinates of rotated vector
    """
    return (rotate(theta) @ colvector(dx, dy)).ravel()[:2]


# ======================================================================================
# Generic Coordinate Transformation Tools
# ======================================================================================


def apply_affine_transform(coords: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply an affine transformation matrix to coordinates.

    This is a generic function that applies a 2D affine transformation
    to coordinate arrays.

    Args:
        coords: Coordinate array of shape (N, 2) or (N, 4) or (N, 6), etc.
        matrix: 3x3 affine transformation matrix

    Returns:
        Transformed coordinate array of the same shape as input
    """
    if coords.size == 0:
        return coords

    # Reshape coordinates to handle different formats
    original_shape = coords.shape
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)

    # Determine how many coordinate pairs we have
    n_coords = coords.shape[1] // 2

    # Transform each (x, y) pair
    transformed = coords.copy()
    for i in range(n_coords):
        x_idx = i * 2
        y_idx = i * 2 + 1

        # Extract x, y coordinates
        x = coords[:, x_idx]
        y = coords[:, y_idx]

        # Create homogeneous coordinates
        ones = np.ones_like(x)
        homogeneous = np.vstack([x, y, ones])

        # Apply transformation
        transformed_homogeneous = matrix @ homogeneous

        # Extract transformed coordinates
        transformed[:, x_idx] = transformed_homogeneous[0, :]
        transformed[:, y_idx] = transformed_homogeneous[1, :]

    return transformed.reshape(original_shape)


def create_rotation_matrix(angle: float) -> np.ndarray:
    """Create a 2D rotation matrix.

    Args:
        angle: Rotation angle in radians

    Returns:
        3x3 homogeneous rotation matrix
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=float)


def create_flip_matrix(flip_h: bool = False, flip_v: bool = False) -> np.ndarray:
    """Create a 2D flip matrix.

    Args:
        flip_h: Whether to flip horizontally
        flip_v: Whether to flip vertically

    Returns:
        3x3 homogeneous flip matrix
    """
    sx = -1 if flip_h else 1
    sy = -1 if flip_v else 1
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=float)


def create_transpose_matrix() -> np.ndarray:
    """Create a 2D transpose matrix (swap x and y coordinates).

    Returns:
        3x3 homogeneous transpose matrix
    """
    return np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)


def transform_rectangular_coords(coords: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Transform rectangular coordinates [x0, y0, dx, dy].

    Args:
        coords: Rectangular coordinates array
        matrix: 3x3 transformation matrix

    Returns:
        Transformed rectangular coordinates
    """
    # Convert [x0, y0, dx, dy] to corner points
    x0, y0, dx, dy = coords[0], coords[1], coords[2], coords[3]
    corners = np.array([[x0, y0, x0 + dx, y0 + dy]])

    # Transform the corners
    transformed_corners = apply_affine_transform(corners, matrix)

    # Convert back to [x0, y0, dx, dy] format
    tx0, ty0, tx1, ty1 = transformed_corners[0]
    new_dx = tx1 - tx0
    new_dy = ty1 - ty0

    return np.array([tx0, ty0, new_dx, new_dy])


def transform_circular_coords(coords: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Transform circular coordinates [xc, yc, r].

    For circles, only the center point is transformed, radius remains unchanged.

    Args:
        coords: Circular coordinates array
        matrix: 3x3 transformation matrix

    Returns:
        Transformed circular coordinates
    """
    xc, yc, r = coords[0], coords[1], coords[2]
    center = np.array([[xc, yc]])

    # Transform only the center point
    transformed_center = apply_affine_transform(center, matrix)
    txc, tyc = transformed_center[0]

    return np.array([txc, tyc, r])


def transform_polygonal_coords(coords: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Transform polygonal coordinates [x0, y0, x1, y1, x2, y2, ...].

    Args:
        coords: Polygonal coordinates array
        matrix: 3x3 transformation matrix

    Returns:
        Transformed polygonal coordinates
    """
    # Reshape to (1, N) for apply_affine_transform
    coords_reshaped = coords.reshape(1, -1)
    transformed = apply_affine_transform(coords_reshaped, matrix)
    return transformed.reshape(-1)


# ======================================================================================
# Geometry-specific transformation functions
# ======================================================================================


def rotate_coords(
    coords: np.ndarray, angle: float, coord_type: str = "points"
) -> np.ndarray:
    """Rotate coordinates by the specified angle.

    Args:
        coords: Coordinate array
        angle: Rotation angle in radians
        coord_type: Type of coordinates ("points", "rectangular", "circular",
                   "polygonal")

    Returns:
        Rotated coordinates
    """
    # Use the proven scalar transformation functions for coordinate transformation
    from sigima.objects.scalar import GeometryResult, KindShape
    from sigima.proc import scalar

    if coord_type == "rectangular":
        # Convert [x0, y0, dx, dy] to GeometryResult and transform
        temp_coords = np.array([coords])
        temp_geom = GeometryResult("temp", KindShape.RECTANGLE, coords=temp_coords)
        transformed_geom = scalar.rotate(temp_geom, angle, None)
        return transformed_geom.coords[0]

    elif coord_type == "circular":
        # For circular coordinates [xc, yc, r], only transform the center
        xc, yc, r = coords[0], coords[1], coords[2]
        temp_coords = np.array([[xc, yc]])
        temp_geom = GeometryResult("temp", KindShape.POINT, coords=temp_coords)
        transformed_geom = scalar.rotate(temp_geom, angle, None)
        txc, tyc = transformed_geom.coords[0]
        return np.array([txc, tyc, r])

    elif coord_type == "polygonal":
        # Transform polygon coordinates
        temp_coords = np.array([coords])
        temp_geom = GeometryResult("temp", KindShape.POLYGON, coords=temp_coords)
        transformed_geom = scalar.rotate(temp_geom, angle, None)
        return transformed_geom.coords[0]

    else:  # points
        # Transform point coordinates
        if coords.ndim == 1 and len(coords) == 2:
            temp_coords = np.array([coords])
        else:
            temp_coords = coords.reshape(-1, 2)
        temp_geom = GeometryResult("temp", KindShape.POINT, coords=temp_coords)
        transformed_geom = scalar.rotate(temp_geom, angle, None)
        return transformed_geom.coords.reshape(-1)


def flip_coords(
    coords: np.ndarray,
    flip_h: bool = False,
    flip_v: bool = False,
    coord_type: str = "points",
) -> np.ndarray:
    """Flip coordinates horizontally and/or vertically.

    Args:
        coords: Coordinate array
        flip_h: Whether to flip horizontally
        flip_v: Whether to flip vertically
        coord_type: Type of coordinates ("points", "rectangular", "circular",
                   "polygonal")

    Returns:
        Flipped coordinates
    """
    # Use the proven scalar transformation functions
    from sigima.objects.scalar import GeometryResult, KindShape
    from sigima.proc import scalar

    if coord_type == "rectangular":
        temp_coords = np.array([coords])
        temp_geom = GeometryResult("temp", KindShape.RECTANGLE, coords=temp_coords)

        # Apply transformations in sequence
        if flip_h:
            temp_geom = scalar.fliph(temp_geom, None)
        if flip_v:
            temp_geom = scalar.flipv(temp_geom, None)

        return temp_geom.coords[0]

    elif coord_type == "circular":
        xc, yc, r = coords[0], coords[1], coords[2]
        temp_coords = np.array([[xc, yc]])
        temp_geom = GeometryResult("temp", KindShape.POINT, coords=temp_coords)

        if flip_h:
            temp_geom = scalar.fliph(temp_geom, None)
        if flip_v:
            temp_geom = scalar.flipv(temp_geom, None)

        txc, tyc = temp_geom.coords[0]
        return np.array([txc, tyc, r])

    elif coord_type == "polygonal":
        temp_coords = np.array([coords])
        temp_geom = GeometryResult("temp", KindShape.POLYGON, coords=temp_coords)

        if flip_h:
            temp_geom = scalar.fliph(temp_geom, None)
        if flip_v:
            temp_geom = scalar.flipv(temp_geom, None)

        return temp_geom.coords[0]

    else:  # points
        if coords.ndim == 1 and len(coords) == 2:
            temp_coords = np.array([coords])
        else:
            temp_coords = coords.reshape(-1, 2)
        temp_geom = GeometryResult("temp", KindShape.POINT, coords=temp_coords)

        if flip_h:
            temp_geom = scalar.fliph(temp_geom, None)
        if flip_v:
            temp_geom = scalar.flipv(temp_geom, None)

        return temp_geom.coords.reshape(-1)


def transpose_coords(coords: np.ndarray, coord_type: str = "points") -> np.ndarray:
    """Transpose coordinates (swap x and y).

    Args:
        coords: Coordinate array
        coord_type: Type of coordinates ("points", "rectangular", "circular",
                   "polygonal")

    Returns:
        Transposed coordinates
    """
    # Use the proven scalar transformation functions
    from sigima.objects.scalar import GeometryResult, KindShape
    from sigima.proc import scalar

    if coord_type == "rectangular":
        temp_coords = np.array([coords])
        temp_geom = GeometryResult("temp", KindShape.RECTANGLE, coords=temp_coords)
        transformed_geom = scalar.transpose(temp_geom)
        return transformed_geom.coords[0]

    elif coord_type == "circular":
        xc, yc, r = coords[0], coords[1], coords[2]
        temp_coords = np.array([[xc, yc]])
        temp_geom = GeometryResult("temp", KindShape.POINT, coords=temp_coords)
        transformed_geom = scalar.transpose(temp_geom)
        txc, tyc = transformed_geom.coords[0]
        return np.array([txc, tyc, r])

    elif coord_type == "polygonal":
        temp_coords = np.array([coords])
        temp_geom = GeometryResult("temp", KindShape.POLYGON, coords=temp_coords)
        transformed_geom = scalar.transpose(temp_geom)
        return transformed_geom.coords[0]

    else:  # points
        if coords.ndim == 1 and len(coords) == 2:
            temp_coords = np.array([coords])
        else:
            temp_coords = coords.reshape(-1, 2)
        temp_geom = GeometryResult("temp", KindShape.POINT, coords=temp_coords)
        transformed_geom = scalar.transpose(temp_geom)
        return transformed_geom.coords.reshape(-1)
