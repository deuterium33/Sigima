# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Scalar computation objects (see parent package :mod:`sigima.proc`)

Computation functions for scalar results (TableResult, GeometryResult).
"""

from __future__ import annotations

import numpy as np

from sigima.objects.scalar import GeometryResult

# ======================================================================================
# MARK: Low-level affine tr.
# ======================================================================================


def _apply_affine(geom: GeometryResult, matrix: np.ndarray) -> GeometryResult:
    """Apply a 2D affine transform (3x3 matrix) to all coordinates in GeometryResult.

    Args:
        geom: The geometry result to transform.
        matrix: The 3x3 affine transformation matrix.

    Returns:
        The transformed geometry result.
    """
    coords = geom.coords.copy()

    def _apply_pairs(pairs_cols: list[int]):
        for i in range(0, len(pairs_cols), 2):
            xcol, ycol = pairs_cols[i], pairs_cols[i + 1]
            sub = coords[:, [xcol, ycol]]
            homo = np.c_[sub, np.ones(len(sub))]
            out = homo @ matrix.T
            coords[:, xcol] = out[:, 0]
            coords[:, ycol] = out[:, 1]

    k = geom.kind
    if k in ("point", "marker"):
        _apply_pairs([0, 1])
    elif k in ("segment", "rectangle"):
        _apply_pairs([0, 1, 2, 3])
    elif k == "circle":
        _apply_pairs([0, 1])  # radius unchanged for isometric transforms
    elif k == "ellipse":
        _apply_pairs([0, 1])  # axes/theta unchanged unless scaling non-uniformly
    elif k == "polygon":
        cols = list(range(coords.shape[1]))
        _apply_pairs(cols)

    return GeometryResult(
        title=geom.title,
        kind=geom.kind,
        coords=coords,
        roi_indices=geom.roi_indices,
        attrs=dict(geom.attrs),
    )


# ======================================================================================
# MARK: Helpers / centers
# ======================================================================================


def _geom_center_xy(geom: GeometryResult) -> tuple[float, float]:
    """Compute geometric center (mean of all finite points) in GeometryResult.

    Args:
        geom: The geometry result to compute the center for.

    Returns:
        The (x, y) coordinates of the geometric center.
    """
    pts = geom.coords[:, :2].reshape(-1, 2)
    finite_pts = pts[~np.isnan(pts).any(axis=1)]
    if finite_pts.size == 0:
        return (0.0, 0.0)
    return tuple(np.mean(finite_pts, axis=0))


def _geom_center_x(geom: GeometryResult) -> float:
    """Center X coordinate for 1D transformations.

    Args:
        geom: The geometry result to compute the center for.

    Returns:
        The (x, y) coordinates of the geometric center.
    """
    xs = geom.coords[:, 0]
    xs = xs[~np.isnan(xs)]
    if xs.size == 0:
        return 0.0
    return float(np.mean(xs))


# ======================================================================================
# MARK: 2D geometry tr.
# ======================================================================================


def rotate(
    geom: GeometryResult, angle: float, center: tuple[float, float] | None = None
) -> GeometryResult:
    """Rotate the geometry result by a given angle around a center point.

    Args:
        geom: The geometry result to rotate.
        angle: The angle to rotate by (in radians).
        center: The center point to rotate around (x, y).

    Returns:
        The rotated geometry result.
    """
    if center is None:
        center = _geom_center_xy(geom)
    cx, cy = center
    c, s = np.cos(angle), np.sin(angle)
    t1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], float)
    r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)
    t2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], float)
    return _apply_affine(geom, t2 @ r @ t1)


def fliph(geom: GeometryResult, cx: float | None = None) -> GeometryResult:
    """Flip the geometry result horizontally around a center line.

    Args:
        geom: The geometry result to flip.
        cx: The x-coordinate of the center line to flip around
         (default: center of geometry).

    Returns:
        The flipped geometry result.
    """
    if cx is None:
        cx = _geom_center_xy(geom)[0]
    t1 = np.array([[1, 0, -cx], [0, 1, 0], [0, 0, 1]], float)
    f = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    t2 = np.array([[1, 0, cx], [0, 1, 0], [0, 0, 1]], float)
    return _apply_affine(geom, t2 @ f @ t1)


def flipv(geom: GeometryResult, cy: float | None = None) -> GeometryResult:
    """Flip the geometry result vertically around a center line.

    Args:
        geom: The geometry result to flip.
        cy: The y-coordinate of the center line to flip around
         (default: center of geometry).

    Returns:
        The flipped geometry result.
    """
    if cy is None:
        cy = _geom_center_xy(geom)[1]
    t1 = np.array([[1, 0, 0], [0, 1, -cy], [0, 0, 1]], float)
    f = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], float)
    t2 = np.array([[1, 0, 0], [0, 1, cy], [0, 0, 1]], float)
    return _apply_affine(geom, t2 @ f @ t1)


def transpose(geom: GeometryResult) -> GeometryResult:
    """Transpose the geometry result (swap x and y coordinates).

    Args:
        geom: The geometry result to transpose.

    Returns:
        The transposed geometry result.
    """
    m = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], float)
    return _apply_affine(geom, m)


def translate(geom: GeometryResult, dx: float, dy: float) -> GeometryResult:
    """Translate the geometry result by a given offset.

    Args:
        geom: The geometry result to translate.
        dx: The x-axis offset.
        dy: The y-axis offset.

    Returns:
        The translated geometry result.
    """
    m = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], float)
    return _apply_affine(geom, m)


def scale(
    geom: GeometryResult,
    sx: float,
    sy: float,
    center: tuple[float, float] | None = None,
) -> GeometryResult:
    """Scale the geometry result by a given factor around a center point.

    Args:
        geom: The geometry result to scale.
        sx: The scaling factor in the x direction.
        sy: The scaling factor in the y direction.
        center: The center point to scale around (x, y).

    Returns:
        The scaled geometry result.
    """
    if center is None:
        center = _geom_center_xy(geom)
    cx, cy = center
    t1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], float)
    s = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], float)
    t2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], float)
    return _apply_affine(geom, t2 @ s @ t1)


# ======================================================================================
# MARK: 1D geometry tr.
# ======================================================================================


def scale_1d(
    geom: GeometryResult, factor: float, center_x: float | None = None
) -> GeometryResult:
    """Scale the geometry result by a given factor around a center point.

    Args:
        geom: The geometry result to scale.
        factor: The scaling factor in the x direction.
        center_x: The center point to scale around (x).

    Returns:
        The scaled geometry result.
    """
    if center_x is None:
        center_x = _geom_center_x(geom)
    coords = geom.coords.copy()
    coords[:, 0] = (coords[:, 0] - center_x) * factor + center_x
    return GeometryResult(
        title=geom.title,
        kind=geom.kind,
        coords=coords,
        roi_indices=geom.roi_indices,
        attrs=dict(geom.attrs),
    )


def translate_1d(geom: GeometryResult, dx: float) -> GeometryResult:
    """Translate the geometry result by a given offset.

    Args:
        geom: The geometry result to translate.
        dx: The x-axis offset.

    Returns:
        The translated geometry result.
    """
    coords = geom.coords.copy()
    coords[:, 0] += dx
    return GeometryResult(
        title=geom.title,
        kind=geom.kind,
        coords=coords,
        roi_indices=geom.roi_indices,
        attrs=dict(geom.attrs),
    )


# ======================================================================================
# MARK: Mappings
# ======================================================================================

#: Image geometry mapping between image computation function names (keys) and
# their corresponding geometry update functions (values).
IMAGE_GEOMETRY_UPDATE_MAP = {
    "rotate": (rotate, lambda geom, p, **k: (geom, p.angle * np.pi / 180.0, None)),
    "rotate90": (rotate, lambda geom, **k: (geom, np.pi / 2, None)),
    "rotate270": (rotate, lambda geom, **k: (geom, 3 * np.pi / 2, None)),
    "fliph": (fliph, lambda geom, **k: (geom, None)),
    "flipv": (flipv, lambda geom, **k: (geom, None)),
    "resize": (scale, lambda geom, p, **k: (geom, 1.0 / p.zoom, 1.0 / p.zoom, None)),
    "swap_axes": (transpose, lambda geom, **k: (geom,)),
    "binning": (scale, lambda geom, p, **k: (geom, 1.0 / p.sx, 1.0 / p.sy, None)),
}

#: Signal geometry mapping between signal processing function names (keys) and
# their corresponding geometry update functions (values).
SIGNAL_GEOMETRY_UPDATE_MAP = {
    "reverse_x": (
        scale_1d,
        lambda geom, **kwargs: (geom, -1.0, None),  # None → center auto-computed
    ),
}
