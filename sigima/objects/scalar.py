# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Scalar results
==============

Scalar results are compute-friendly result containers for scalar tables and geometric
outputs.

Overview
--------

This module defines two pure data classes:

- `TableResult`: table of scalar metrics
- `GeometryResult`: geometric outputs (points, segments, circles, ...)

Each result object is a simple data container with no behavior or methods:

- It contains the result of a 1-to-0 processing function
  (e.g. `sigima.proc.signal.fwhm()`), i.e. a computation function that takes a signal
  or image object (`SignalObj` or `ImageObj`) as input and produces a scalar output
  (`TableResult` or `GeometryResult`).

- The result may consist of multiple rows, each corresponding to a different ROI.

.. note::

    No UI/HTML, no DataLab-specific metadata here. Adapters/formatters live in
    DataLab. These classes are JSON-friendly via `to_dict()`/`from_dict()`.

Conventions
-----------

Conventions regarding ROI and geometry are as follows:

- ROI indexing:

  - `NO_ROI = -1` sentinel is used for "full image / no ROI" rows.
  - Per-ROI rows use non-negative indices (0-based).

- Geometry coordinates (physical units):

  - `"point"` / `"marker"`: `[x, y]`
  - `"segment"`: `[x0, y0, x1, y1]`
  - `"rectangle"`: `[x0, y0, width, height]`
  - `"circle"`: `[x0, y0, radius]`
  - `"ellipse"`: `[x0, y0, a, b, theta]`   # theta in radians
  - `"polygon"`: `[x0, y0, x1, y1, ..., xn, yn]`  (rows may be NaN-padded)
"""

from __future__ import annotations

import enum
import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterable, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:
    from sigima.objects import ImageObj, SignalObj

# Sentinel value for "full signal/image / no ROI" rows in result tables
NO_ROI: int = -1


class KindShape(str, enum.Enum):
    """Geometric shape types."""

    POINT = "point"
    SEGMENT = "segment"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    RECTANGLE = "rectangle"
    POLYGON = "polygon"
    MARKER = "marker"

    @classmethod
    def values(cls) -> list[str]:
        """Return all shape type values."""
        return [e.value for e in cls]


@dataclass(frozen=True)
class TableResult:
    """Table of scalar results, optionally per-ROI.

    Args:
        title: Human-readable title for this table of results.
        names: Column names (one per metric).
        labels: Human-readable labels for each column (including units as
         formatted strings).
        data: 2-D array of shape (N, len(names)) with scalar values.
        roi_indices: Optional 1-D array (N,) mapping rows to ROI indices.
         Use NO_ROI (-1) for the "full image / no ROI" row.
        attrs: Optional algorithmic context (e.g. thresholds, method variant).

    Raises:
        ValueError: If dimensions are inconsistent or fields are invalid.

    Notes:
        - No UI/presentation concerns, no persistence schema here.
        - Use DataLab-side adapters to store results in metadata if needed.
    """

    title: str
    names: Sequence[str] = field(default_factory=list)
    labels: Sequence[str] = field(default_factory=list)
    data: np.ndarray = field(default_factory=lambda: np.empty((0, 0), float))
    roi_indices: np.ndarray | None = None
    attrs: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if not isinstance(self.title, str) or not self.title:
            raise ValueError("title must be a non-empty string")
        if not isinstance(self.names, (list, tuple)) or not all(
            isinstance(c, str) for c in self.names
        ):
            raise ValueError("names must be a sequence of strings")
        if not isinstance(self.labels, (list, tuple)) or not all(
            isinstance(c, str) for c in self.labels
        ):
            raise ValueError("labels must be a sequence of strings")
        if not isinstance(self.data, np.ndarray):
            raise ValueError("data must be a numpy array")
        if self.data.ndim != 2 or self.data.shape[1] != len(self.names):
            raise ValueError("data must be (N, ncols) and match names length")
        if self.roi_indices is not None:
            if (
                not isinstance(self.roi_indices, np.ndarray)
                or self.roi_indices.ndim != 1
            ):
                raise ValueError("roi_indices must be a 1-D numpy array if provided")
            if len(self.roi_indices) != len(self.data):
                raise ValueError("roi_indices length must match number of data rows")

    # -------- Factory methods --------

    @classmethod
    def from_rows(
        cls,
        title: str,
        names: Sequence[str],
        labels: Sequence[str],
        rows: np.ndarray,
        roi_indices: np.ndarray | None = None,
        *,
        attrs: dict[str, object] | None = None,
    ) -> TableResult:
        """Create a TableResult from raw data.

        Args:
            title: Human-readable title for this table of results.
            names: Column names (one per metric).
            labels: Human-readable labels for each column (including units as
             formatted strings).
            rows: 2-D array of shape (N, len(names)) with scalar values.
            roi_indices: Optional 1-D array (N,) mapping rows to ROI indices.
             Use NO_ROI (-1) for the "full image / no ROI" row.
            attrs: Optional algorithmic context (e.g. thresholds, method variant).

        Returns:
            A TableResult instance.
        """
        return cls(
            title,
            names,
            labels,
            np.asarray(rows, float),
            None if roi_indices is None else np.asarray(roi_indices, int),
            {} if attrs is None else dict(attrs),
        )

    # -------- JSON-friendly (de)serialization (no DataLab metadata coupling) -----

    def to_dict(self) -> dict:
        """Convert the TableResult to a dictionary."""
        return {
            "schema": 1,
            "title": self.title,
            "names": list(self.names),
            "labels": list(self.labels),
            "data": self.data.tolist(),
            "roi_indices": None
            if self.roi_indices is None
            else self.roi_indices.tolist(),
            "attrs": dict(self.attrs) if self.attrs else {},
        }

    @staticmethod
    def from_dict(d: dict) -> TableResult:
        """Convert a dictionary to a TableResult."""
        return TableResult(
            title=d["title"],
            names=list(d["names"]),
            labels=list(d["labels"]),
            data=np.asarray(d["data"], dtype=float),
            roi_indices=None
            if d.get("roi_indices") is None
            else np.asarray(d["roi_indices"], dtype=int),
            attrs=dict(d.get("attrs", {})),
        )

    # -------- User-oriented methods --------

    def col(self, name: str) -> np.ndarray:
        """Return the column vector by name (raises KeyError if missing).

        Args:
            name: The name of the column to retrieve.

        Returns:
            A 1-D numpy array containing the column data.
        """
        try:
            j = list(self.names).index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        return self.data[:, j]

    def __getitem__(self, name: str) -> np.ndarray:
        """Shorthand for col(name)."""
        return self.col(name)

    def __contains__(self, name: str) -> bool:
        """Check if a column name exists in the table.

        Args:
            name: The name of the column to check.

        Returns:
            True if the column exists, False otherwise.
        """
        return name in self.names

    def __len__(self) -> int:
        """Return the number of names in the table."""
        return len(self.names)

    def value(self, name: str, roi: int | None = None) -> float:
        """Return a single scalar by column name and ROI.

        Args:
            name: The name of the column to retrieve.
            roi: The region of interest (ROI) to filter by (optional).
             Use None for NO_ROI row.

        Returns:
            A single scalar value from the specified column and ROI.
        """
        vec = self.col(name)
        if self.roi_indices is None:
            # single row (common in 'full image' stats)
            if len(vec) != 1:
                raise ValueError(
                    "Ambiguous selection: multiple rows but no ROI indices"
                )
            return float(vec[0])
        target = NO_ROI if roi is None else int(roi)
        mask: np.ndarray = self.roi_indices == target
        if not mask.any():
            raise KeyError(f"No row for ROI={target}")
        if mask.sum() != 1:
            raise ValueError(f"Ambiguous selection: {mask.sum()} rows for ROI={target}")
        return float(vec[mask][0])

    def as_dict(self, roi: int | None = None) -> dict[str, float]:
        """Return a {column -> value} mapping for one row (ROI or full image).

        Args:
            roi: The region of interest (ROI) to filter by (optional).
             Use None for NO_ROI row.

        Returns:
            A dictionary mapping column names to their corresponding values.
        """
        if self.roi_indices is None:
            if self.data.shape[0] != 1:
                raise ValueError(
                    "Ambiguous selection: multiple rows but no ROI indices"
                )
            row = self.data[0]
        else:
            target = NO_ROI if roi is None else int(roi)
            mask: np.ndarray = self.roi_indices == target
            if not mask.any():
                raise KeyError(f"No row for ROI={target}")
            if mask.sum() != 1:
                raise ValueError(
                    f"Ambiguous selection: {mask.sum()} rows for ROI={target}"
                )
            row = self.data[mask][0]
        return {name: float(row[j]) for j, name in enumerate(self.names)}


class TableResultBuilder:
    """Builder for TableResult with fluent interface.

    Args:
        title: The title of the table.
    """

    def __init__(self, title: str) -> None:
        self.title = title
        self.columns: list[tuple[Callable, str, str]] = []

    def add(self, func: Callable, name: str, label: str = "") -> None:
        """Add a column to the table.

        Args:
            func: The function to compute the column values.
            name: The name of the column.
            label: The human-readable label for the column (including units).
             Default is an empty string.
        """
        assert isinstance(name, str) and name, "Column name must be a non-empty string"
        assert isinstance(label, str), "Column label must be a string"
        assert isinstance(func, Callable), "Column function must be callable"
        # Check function signature:
        sig = inspect.signature(func)
        if len(sig.parameters) < 1:
            raise ValueError(
                f"Column function '{name}' must accept at least one argument"
            )
        first_param = list(sig.parameters.values())[0]
        if (
            first_param.annotation is not sig.empty
            and first_param.annotation != "np.ndarray"
        ):
            raise ValueError(f"Column function '{name}' must accept a np.ndarray")
        # Check return type
        if sig.return_annotation is not sig.empty and sig.return_annotation not in (
            "float",
            "int",
        ):
            raise ValueError(f"Column function '{name}' must return a float or int")
        self.columns.append((name, label, func))

    def compute(self, obj: SignalObj | ImageObj) -> TableResult:
        """Extract data from the image or signal object and compute the table.

        Args:
            obj: The image or signal object to extract data from.

        Returns:
            A TableResult object containing the extracted data.
        """
        names = [name for name, _, _ in self.columns]
        labels = [label for _, label, _ in self.columns]
        for label in labels:
            if label != "":
                # Check if formatting works
                try:
                    label.format(obj) % 1.234  # Test formatting with a dummy value
                except KeyError as exc:
                    raise ValueError(f"Label '{label}' is not valid") from exc

        roi_indices = list(obj.iterate_roi_indices())
        if roi_indices[0] is not None:
            roi_indices.insert(0, None)

        rows = []
        roi_idx = []
        for i_roi in roi_indices:
            data = obj.get_data(i_roi)
            rows.append([float(func(data)) for _name, _label, func in self.columns])
            roi_idx.append(NO_ROI if i_roi is None else int(i_roi))

        return TableResult.from_rows(
            title=self.title,
            names=names,
            labels=labels,
            rows=np.asarray(rows, float),
            roi_indices=np.asarray(roi_idx, int),
        )


@dataclass(frozen=True)
class GeometryResult:
    """Geometric outputs, optionally per-ROI.

    Args:
        title: Human-readable title for this geometric output set.
        kind: Shape kind (`KindShape` member or its string value).
        coords: 2-D array (N, K) with coordinates per row. K depends on `kind`
         and may be NaN-padded (e.g., for polygons).
        roi_indices: Optional 1-D array (N,) mapping rows to ROI indices.
         Use NO_ROI (-1) for the "full signal/image / no ROI" row.
        attrs: Optional algorithmic context (e.g. thresholds, method variant).

    Raises:
        ValueError: If dimensions are inconsistent or fields are invalid.

    .. warning::

        Coordinate conventions are as follows:

        - `KindShape.POINT`: `[x, y]`
        - `KindShape.SEGMENT`: `[x0, y0, x1, y1]`
        - `KindShape.RECTANGLE`: `[x0, y0, width, height]`
        - `KindShape.CIRCLE`: `[x0, y0, radius]`
        - `KindShape.ELLIPSE`: `[x0, y0, a, b, theta]`   # theta in radians
        - `KindShape.POLYGON`: `[x0, y0, x1, y1, ..., xn, yn]`  (rows may be NaN-padded)
    """

    title: str
    kind: KindShape
    coords: np.ndarray
    roi_indices: np.ndarray | None = None
    attrs: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        # --- kind validation/coercion (smooth migration) ---
        k = object.__getattribute__(self, "kind")
        if isinstance(k, str):
            try:
                k = KindShape(k)  # coerce "ellipse" -> KindShape.ELLIPSE
            except ValueError as exc:
                raise ValueError(f"Unsupported geometry kind: {k!r}") from exc
            object.__setattr__(self, "kind", k)
        elif not isinstance(k, KindShape):
            raise ValueError("kind must be a KindShape or its string value")
        if not isinstance(self.title, str) or not self.title:
            raise ValueError("title must be a non-empty string")
        if not isinstance(self.coords, np.ndarray) or self.coords.ndim != 2:
            raise ValueError("coords must be a 2-D numpy array")
        if k == KindShape.POINT and self.coords.shape[1] != 2:
            raise ValueError("coords for 'point' must be (N,2)")
        if k == KindShape.SEGMENT and self.coords.shape[1] != 4:
            raise ValueError("coords for 'segment' must be (N,4)")
        if k == KindShape.CIRCLE and self.coords.shape[1] != 3:
            raise ValueError("coords for 'circle' must be (N,3)")
        if k == KindShape.ELLIPSE and self.coords.shape[1] != 5:
            raise ValueError("coords for 'ellipse' must be (N,5)")
        if k == KindShape.RECTANGLE and self.coords.shape[1] != 4:
            raise ValueError("coords for 'rectangle' must be (N,4)")
        if k == KindShape.POLYGON and self.coords.shape[1] % 2 != 0:
            raise ValueError("coords for 'polygon' must be (N,2M) for M vertices")
        if self.roi_indices is not None:
            if (
                not isinstance(self.roi_indices, np.ndarray)
                or self.roi_indices.ndim != 1
            ):
                raise ValueError("roi_indices must be a 1-D numpy array if provided")
            if len(self.roi_indices) != len(self.coords):
                raise ValueError("roi_indices length must match number of coord rows")

    # -------- Factory methods --------

    @classmethod
    def from_coords(
        cls,
        title: str,
        kind: KindShape,
        coords: np.ndarray,
        roi_indices: np.ndarray | None = None,
        *,
        attrs: dict[str, object] | None = None,
    ) -> GeometryResult:
        """Create a GeometryResult from raw data.

        Args:
            title: Human-readable title for this geometric output.
            kind: Shape kind (e.g. "point", "segment").
            coords: 2-D array (N, K) with coordinates per row.
            roi_indices: Optional 1-D array (N,) mapping rows to ROI indices.
            attrs: Optional algorithmic context (e.g. thresholds, method variant).

        Returns:
            A GeometryResult instance.
        """
        return cls(
            title,
            kind,
            np.asarray(coords, float),
            None if roi_indices is None else np.asarray(roi_indices, int),
            {} if attrs is None else dict(attrs),
        )

    # -------- JSON-friendly (de)serialization (no DataLab metadata coupling) -----

    def to_dict(self) -> dict:
        """Convert the GeometryResult to a dictionary."""
        return {
            "schema": 1,
            "title": self.title,
            "kind": self.kind.value,
            "coords": self.coords.tolist(),
            "roi_indices": None
            if self.roi_indices is None
            else self.roi_indices.tolist(),
            "attrs": dict(self.attrs) if self.attrs else {},
        }

    @staticmethod
    def from_dict(d: dict) -> GeometryResult:
        """Convert a dictionary to a GeometryResult."""
        return GeometryResult(
            title=d["title"],
            kind=KindShape(d["kind"]),
            coords=np.asarray(d["coords"], dtype=float),
            roi_indices=None
            if d.get("roi_indices") is None
            else np.asarray(d["roi_indices"], dtype=int),
            attrs=dict(d.get("attrs", {})),
        )

    # -------- User-oriented methods --------

    def __len__(self) -> int:
        """Return the number of coordinates (rows) in the result."""
        return self.coords.shape[0]

    def rows(self, roi: int | None = None) -> np.ndarray:
        """Return coords for all rows (this ROI or full-image row).

        Args:
            roi: Optional ROI index to filter rows.

        Returns:
            2-D array of shape (M, K) with coordinates for the selected rows.
        """
        if self.roi_indices is None:
            return self.coords
        target = NO_ROI if roi is None else int(roi)
        return self.coords[self.roi_indices == target]

    # Optional convenience for common kinds:
    def segments_lengths(self) -> np.ndarray:
        """For kind='segment': return vector of segment lengths."""
        if self.kind is not KindShape.SEGMENT:
            raise ValueError("segments_lengths requires kind='segment'")
        dx = self.coords[:, 2] - self.coords[:, 0]
        dy = self.coords[:, 3] - self.coords[:, 1]
        return np.sqrt(dx * dx + dy * dy)

    def circles_radii(self) -> np.ndarray:
        """For kind='circle': return radii."""
        if self.kind is not KindShape.CIRCLE:
            raise ValueError("circles_radii requires kind='circle'")
        return self.coords[:, 2]

    def ellipse_axes_angles(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For kind='ellipse': return (a, b, theta)."""
        if self.kind is not KindShape.ELLIPSE:
            raise ValueError("ellipse_axes_angles requires kind='ellipse'")
        return self.coords[:, 2], self.coords[:, 3], self.coords[:, 4]


# ===========================
# Small, compute-side helpers
# ===========================


def calc_table_from_data(
    title: str,
    data: np.ndarray,
    labeledfuncs: Mapping[str, Callable[[np.ndarray], float]],
    roi_masks: list[np.ndarray] | None = None,
    attrs: dict[str, object] | None = None,
) -> TableResult:
    """Run scalar metrics on a full array or per-ROI masks and return a TableResult.

    Args:
        title: Result title.
        data: N-D array consumed by metric functions.
        labeledfuncs: Mapping of {label: func}, where func(data_or_masked) -> float.
        roi_masks: Optional list of boolean masks (same shape as data). If provided,
         results are computed per mask; otherwise a single full-image row is returned.
        attrs: Optional algorithmic context.

    Returns:
        TableResult with rows per ROI mask (or one row if `roi_masks` is None).
        `roi_indices` will be the mask indices (0..M-1) or NO_ROI for the
        single row.
    """
    names = list(labeledfuncs.keys())
    funcs = list(labeledfuncs.values())

    if roi_masks:
        rows = []
        roi_idx = []
        for i, m in enumerate(roi_masks):
            sub = data[m] if (isinstance(m, np.ndarray) and m.dtype == bool) else data
            rows.append([float(f(sub)) for f in funcs])
            roi_idx.append(i)
        return TableResult(
            title=title,
            names=names,
            data=np.asarray(rows, dtype=float),
            roi_indices=np.asarray(roi_idx, dtype=int),
            attrs={} if attrs is None else dict(attrs),
        )

    # No ROI: single row with NO_ROI sentinel
    row = [float(f(data)) for f in funcs]
    return TableResult(
        title=title,
        names=names,
        data=np.asarray([row], dtype=float),
        roi_indices=np.asarray([NO_ROI], dtype=int),
        attrs={} if attrs is None else dict(attrs),
    )


# ---- Concatenation & filtering utilities (pure NumPy, no pandas) ----


def concat_tables(title: str, items: Iterable[TableResult]) -> TableResult:
    """Concatenate multiple TableResult objects with identical names.

    Args:
        title: Title for the concatenated result.
        items: Iterable of TableResult objects to concatenate.

    Returns:
        TableResult with concatenated data and updated metadata.
    """
    items = list(items)
    if not items:
        return TableResult(title=title, names=[], data=np.zeros((0, 0), float))
    first = items[0]
    cols = list(first.names)
    for it in items[1:]:
        if list(it.names) != cols:
            raise ValueError(
                "All TableResult objects must share the same names to concatenate"
            )
    data = (
        np.vstack([it.data for it in items])
        if any(len(it.data) for it in items)
        else np.zeros((0, len(cols)))
    )
    if any(it.roi_indices is not None for it in items):
        parts = [
            (
                it.roi_indices
                if it.roi_indices is not None
                else np.full((len(it.data),), NO_ROI, int)
            )
            for it in items
        ]
        roi = np.concatenate(parts) if len(parts) else None
    else:
        roi = None
    return TableResult(title=title, names=cols, data=data, roi_indices=roi)


def filter_table_by_roi(res: TableResult, roi: int | None) -> TableResult:
    """Filter rows by ROI index. If roi is None, keeps NO_ROI rows.

    Args:
        res: The TableResult to filter.
        roi: The ROI index to filter by, or None to keep all.

    Returns:
        A filtered TableResult.
    """
    if res.roi_indices is None:
        # No ROI info: either keep all or none depending on request
        keep_all = roi in (None, NO_ROI)
        data = res.data if keep_all else np.zeros((0, res.data.shape[1]))
        indices = None if keep_all else np.zeros((0,), int)
        return TableResult(
            title=res.title,
            names=list(res.names),
            data=data,
            roi_indices=indices,
            attrs=dict(res.attrs),
        )
    target = NO_ROI if roi is None else int(roi)
    mask = res.roi_indices == target
    return TableResult(
        title=res.title,
        names=list(res.names),
        data=res.data[mask],
        roi_indices=res.roi_indices[mask],
        attrs=dict(res.attrs),
    )


def concat_geometries(
    title: str,
    items: Iterable[GeometryResult],
    *,
    kind: KindShape | None = None,
) -> GeometryResult:
    """Concatenate multiple GeometryResult objects of the same kind.

    Args:
        title: Title for the concatenated result.
        items: Iterable of GeometryResult objects to concatenate.
        kind: Optional kind label for the concatenated result.

    Returns:
        GeometryResult with concatenated data and updated metadata.
    """
    items = list(items)
    if not items:
        return GeometryResult(
            title=title, kind=KindShape.POINT, coords=np.zeros((0, 2), float)
        )
    k = kind if kind is not None else items[0].kind
    for it in items:
        if it.kind != k:
            raise ValueError(
                "All GeometryResult objects must share the same kind to concatenate"
            )
    max_k = max(it.coords.shape[1] for it in items) if items else 0
    # right-pad with NaNs to match width
    padded = []
    for it in items:
        c = it.coords
        if c.shape[1] < max_k:
            pad = np.full((c.shape[0], max_k - c.shape[1]), np.nan, dtype=float)
            c = np.hstack([c, pad])
        padded.append(c)
    coords = np.vstack(padded) if padded else np.zeros((0, max_k))
    if any(it.roi_indices is not None for it in items):
        parts = [
            (
                it.roi_indices
                if it.roi_indices is not None
                else np.full((len(it.coords),), NO_ROI, int)
            )
            for it in items
        ]
        roi = np.concatenate(parts) if len(parts) else None
    else:
        roi = None
    return GeometryResult(title=title, kind=k, coords=coords, roi_indices=roi)


def filter_geometry_by_roi(res: GeometryResult, roi: int | None) -> GeometryResult:
    """Filter shapes by ROI index. If roi is None, keeps NO_ROI rows.

    Args:
        res: The GeometryResult to filter.
        roi: The ROI index to filter by, or None to keep all.

    Returns:
        A filtered GeometryResult.
    """
    if res.roi_indices is None:
        keep_all = roi in (None, NO_ROI)
        coords = res.coords if keep_all else np.zeros((0, res.coords.shape[1]))
        indices = None if keep_all else np.zeros((0,), int)
        return GeometryResult(
            title=res.title,
            kind=res.kind,
            coords=coords,
            roi_indices=indices,
            attrs=dict(res.attrs),
        )
    target = NO_ROI if roi is None else int(roi)
    mask = res.roi_indices == target
    return GeometryResult(
        title=res.title,
        kind=res.kind,
        coords=res.coords[mask],
        roi_indices=res.roi_indices[mask],
        attrs=dict(res.attrs),
    )
