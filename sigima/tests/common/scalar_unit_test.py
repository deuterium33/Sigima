# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for scalar computation functions (GeometryResult transformations).
"""

from __future__ import annotations

import numpy as np
import pytest

from sigima.objects.scalar import (
    NO_ROI,
    GeometryResult,
    KindShape,
    TableResult,
    TableResultBuilder,
    calc_table_from_data,
    concat_geometries,
    concat_tables,
    filter_geometry_by_roi,
    filter_table_by_roi,
)
from sigima.proc.transformations import transformer


def create_rectangle(x0=0.0, y0=0.0, w=1.0, h=1.0) -> GeometryResult:
    """Create a simple rectangle GeometryResult."""
    coords = np.array([[x0, y0, w, h]], dtype=float)
    return GeometryResult("rect", "rectangle", coords)


def test_rotate() -> None:
    """Test rotation of a rectangle geometry result."""
    rect = create_rectangle(0.0, 0.0, 1.0, 2.0)
    rotated = transformer.rotate(rect, np.pi / 2, center=(0.5, 1.0))
    expected_coords = np.array([[-0.5, 0.5, 2.0, 1.0]])
    assert rotated.coords.shape == rect.coords.shape
    assert np.allclose(rotated.coords, expected_coords)


def test_fliph() -> None:
    """Test horizontal flip and its reversibility."""
    rect = create_rectangle(1.0, 2.0, 2.0, 3.0)
    flipped = transformer.fliph(rect, cx=2.0)
    flipped_back = transformer.fliph(flipped, cx=2.0)
    np.testing.assert_allclose(flipped_back.coords, rect.coords, rtol=1e-12)


def test_flipv() -> None:
    """Test vertical flip and its reversibility."""
    rect = create_rectangle(1.0, 2.0, 2.0, 3.0)
    flipped = transformer.flipv(rect, cy=3.5)
    flipped_back = transformer.flipv(flipped, cy=3.5)
    np.testing.assert_allclose(flipped_back.coords, rect.coords, rtol=1e-12)


def test_translate() -> None:
    """Test translation of a geometry result."""
    rect = create_rectangle()
    translated = transformer.translate(rect, 1.5, -2.0)
    expected = rect.coords + np.array([1.5, -2.0, 0.0, 0.0])
    np.testing.assert_allclose(translated.coords, expected, rtol=1e-12)


def test_scale() -> None:
    """Test scaling and inverse scaling of a geometry result."""
    rect = create_rectangle(1.0, 1.0, 2.0, 2.0)
    scaled = transformer.scale(rect, 2.0, 0.5, center=(2.0, 2.0))
    unscaled = transformer.scale(scaled, 0.5, 2.0, center=(2.0, 2.0))
    np.testing.assert_allclose(unscaled.coords, rect.coords, rtol=1e-12)


def test_transpose() -> None:
    """Test transpose and double-transpose (should restore original)."""
    rect = create_rectangle(1.0, 2.0, 3.0, 4.0)
    transposed = transformer.transpose(rect)
    transposed_back = transformer.transpose(transposed)
    np.testing.assert_allclose(transposed_back.coords, rect.coords, rtol=1e-12)


# ============================= TableResult Tests =============================


def test_table_result_init_valid() -> None:
    """Test TableResult initialization with valid data."""
    data = [[1.0, 2.0], [3.0, 4.0]]
    roi_indices = [0, 1]
    table = TableResult(
        title="Test Table",
        names=["col1", "col2"],
        labels=["Column 1", "Column 2"],
        data=data,
        roi_indices=roi_indices,
        attrs={"method": "test"},
    )
    assert table.title == "Test Table"
    assert list(table.names) == ["col1", "col2"]
    assert list(table.labels) == ["Column 1", "Column 2"]
    assert table.data == data
    assert table.roi_indices == roi_indices
    assert table.attrs == {"method": "test"}


def test_table_result_init_invalid_title() -> None:
    """Test TableResult initialization with invalid title."""
    with pytest.raises(ValueError, match="title must be a non-empty string"):
        TableResult(title="", names=["col1"], data=[[1.0]])


def test_table_result_init_invalid_names() -> None:
    """Test TableResult initialization with invalid names."""
    with pytest.raises(ValueError, match="names must be a sequence of strings"):
        TableResult(title="Test", names=[1, 2], data=[[1.0, 2.0]])


def test_table_result_init_invalid_data_shape() -> None:
    """Test TableResult initialization with invalid data shape."""
    with pytest.raises(ValueError, match="data must be a list of lists"):
        TableResult(
            title="Test", names=["col1"], data=[1.0, 2.0]
        )  # 1D list instead of 2D


def test_table_result_init_invalid_data_columns() -> None:
    """Test TableResult initialization with mismatched data columns."""
    with pytest.raises(ValueError, match="data columns must match names length"):
        TableResult(
            title="Test", names=["col1"], data=[[1.0, 2.0]]
        )  # 2 columns, 1 name


def test_table_result_init_invalid_roi_indices() -> None:
    """Test TableResult initialization with invalid ROI indices."""
    with pytest.raises(
        ValueError, match="roi_indices length must match number of data rows"
    ):
        TableResult(
            title="Test",
            names=["col1"],
            data=[[1.0], [2.0]],
            roi_indices=[0],  # 1 ROI index, 2 data rows
        )


def test_table_result_init_invalid_roi_indices_type() -> None:
    """Test TableResult initialization with invalid ROI indices type."""
    with pytest.raises(ValueError, match="roi_indices must be a list if provided"):
        TableResult(
            title="Test",
            names=["col1"],
            data=[[1.0]],
            roi_indices=[[0]],  # 2D list instead of 1D list
        )


def test_table_result_init_invalid_roi_indices_2d() -> None:
    """Test TableResult initialization with 2D ROI indices."""
    with pytest.raises(ValueError, match="roi_indices must be a list if provided"):
        TableResult(
            title="Test",
            names=["col1"],
            data=[[1.0]],
            roi_indices=[[0]],  # 2D list instead of 1D
        )


def test_table_result_from_rows() -> None:
    """Test TableResult.from_rows factory method."""
    rows = [[1.0, 2.0], [3.0, 4.0]]
    roi_indices = [0, 1]
    table = TableResult.from_rows(
        title="Test Table",
        names=["col1", "col2"],
        labels=["Column 1", "Column 2"],
        rows=rows,
        roi_indices=roi_indices,
        attrs={"method": "test"},
    )
    assert table.title == "Test Table"
    np.testing.assert_array_equal(table.data, np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_array_equal(table.roi_indices, np.array([0, 1]))


def test_table_result_to_dict() -> None:
    """Test TableResult serialization to dictionary."""
    data = [[1.0, 2.0]]
    table = TableResult(
        title="Test",
        names=["col1", "col2"],
        labels=["Column 1", "Column 2"],
        data=data,
        roi_indices=[NO_ROI],
        attrs={"method": "test"},
    )
    result_dict = table.to_dict()
    expected = {
        "schema": 1,
        "title": "Test",
        "names": ["col1", "col2"],
        "labels": ["Column 1", "Column 2"],
        "data": [[1.0, 2.0]],
        "roi_indices": [NO_ROI],
        "attrs": {"method": "test"},
    }
    assert result_dict == expected


def test_table_result_from_dict() -> None:
    """Test TableResult deserialization from dictionary."""
    data_dict = {
        "schema": 1,
        "title": "Test",
        "names": ["col1", "col2"],
        "labels": ["Column 1", "Column 2"],
        "data": [[1.0, 2.0]],
        "roi_indices": [NO_ROI],
        "attrs": {"method": "test"},
    }
    table = TableResult.from_dict(data_dict)
    assert table.title == "Test"
    assert list(table.names) == ["col1", "col2"]
    assert table.data == [[1.0, 2.0]]
    np.testing.assert_array_equal(table.roi_indices, np.array([NO_ROI]))
    assert table.attrs == {"method": "test"}


def test_table_result_col() -> None:
    """Test TableResult.col method."""
    data = [[1.0, 2.0], [3.0, 4.0]]
    table = TableResult(title="Test", names=["col1", "col2"], data=data)

    col1 = table.col("col1")
    assert col1 == [1.0, 3.0]

    col2 = table.col("col2")
    assert col2 == [2.0, 4.0]

    # Test KeyError for non-existent column
    with pytest.raises(KeyError):
        table.col("nonexistent")


def test_table_result_getitem() -> None:
    """Test TableResult.__getitem__ method."""
    data = [[1.0, 2.0]]
    table = TableResult(title="Test", names=["col1", "col2"], data=data)

    col1 = table["col1"]
    assert col1 == [1.0]


def test_table_result_contains() -> None:
    """Test TableResult.__contains__ method."""
    table = TableResult(title="Test", names=["col1", "col2"], data=[[1.0, 2.0]])

    assert "col1" in table
    assert "col2" in table
    assert "nonexistent" not in table


def test_table_result_len() -> None:
    """Test TableResult.__len__ method."""
    table = TableResult(title="Test", names=["col1", "col2"], data=[[1.0, 2.0]])
    assert len(table) == 2


def test_table_result_value_single_row() -> None:
    """Test TableResult.value method with single row."""
    data = [[1.0, 2.0]]
    table = TableResult(title="Test", names=["col1", "col2"], data=data)

    assert table.value("col1") == 1.0
    assert table.value("col2") == 2.0


def test_table_result_value_with_roi() -> None:
    """Test TableResult.value method with ROI indices."""
    data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    roi_indices = [NO_ROI, 0, 1]
    table = TableResult(
        title="Test", names=["col1", "col2"], data=data, roi_indices=roi_indices
    )

    assert table.value("col1", roi=None) == 1.0  # NO_ROI row
    assert table.value("col1", roi=0) == 3.0
    assert table.value("col1", roi=1) == 5.0

    # Test KeyError for non-existent ROI
    with pytest.raises(KeyError):
        table.value("col1", roi=99)


def test_table_result_value_ambiguous_selection() -> None:
    """Test TableResult.value method with ambiguous selection."""
    # Multiple rows without ROI indices
    data = [[1.0], [2.0]]
    table = TableResult(title="Test", names=["col1"], data=data)

    with pytest.raises(
        ValueError, match="Ambiguous selection: multiple rows but no ROI indices"
    ):
        table.value("col1")


def test_table_result_value_duplicate_roi() -> None:
    """Test TableResult.value method with duplicate ROI indices."""
    data = [[1.0], [2.0]]
    roi_indices = [0, 0]  # Duplicate ROI index
    table = TableResult(
        title="Test", names=["col1"], data=data, roi_indices=roi_indices
    )

    with pytest.raises(ValueError, match="Ambiguous selection: 2 rows for ROI=0"):
        table.value("col1", roi=0)


def test_table_result_as_dict_ambiguous_selection() -> None:
    """Test TableResult.as_dict method with ambiguous selection."""
    # Multiple rows without ROI indices
    data = [[1.0, 2.0], [3.0, 4.0]]
    table = TableResult(title="Test", names=["col1", "col2"], data=data)

    with pytest.raises(
        ValueError, match="Ambiguous selection: multiple rows but no ROI indices"
    ):
        table.as_dict()


def test_table_result_as_dict_missing_roi() -> None:
    """Test TableResult.as_dict method with missing ROI."""
    data = [[1.0, 2.0]]
    roi_indices = [0]
    table = TableResult(
        title="Test", names=["col1", "col2"], data=data, roi_indices=roi_indices
    )

    with pytest.raises(KeyError, match="No row for ROI=-1"):
        table.as_dict(roi=None)  # NO_ROI not present


def test_table_result_as_dict_duplicate_roi() -> None:
    """Test TableResult.as_dict method with duplicate ROI indices."""
    data = [[1.0, 2.0], [3.0, 4.0]]
    roi_indices = [0, 0]  # Duplicate ROI index
    table = TableResult(
        title="Test", names=["col1", "col2"], data=data, roi_indices=roi_indices
    )

    with pytest.raises(ValueError, match="Ambiguous selection: 2 rows for ROI=0"):
        table.as_dict(roi=0)


def test_table_result_as_dict_single_row() -> None:
    """Test TableResult.as_dict method with single row."""
    data = [[1.0, 2.0]]
    table = TableResult(title="Test", names=["col1", "col2"], data=data)

    result = table.as_dict()
    expected = {"col1": 1.0, "col2": 2.0}
    assert result == expected


def test_table_result_as_dict_with_roi() -> None:
    """Test TableResult.as_dict method with ROI indices."""
    data = [[1.0, 2.0], [3.0, 4.0]]
    roi_indices = [NO_ROI, 0]
    table = TableResult(
        title="Test", names=["col1", "col2"], data=data, roi_indices=roi_indices
    )

    # Test NO_ROI row
    result_no_roi = table.as_dict(roi=None)
    expected_no_roi = {"col1": 1.0, "col2": 2.0}
    assert result_no_roi == expected_no_roi

    # Test specific ROI
    result_roi0 = table.as_dict(roi=0)
    expected_roi0 = {"col1": 3.0, "col2": 4.0}
    assert result_roi0 == expected_roi0


# ============================ TableResultBuilder Tests ===========================


class MockSignalObj:
    """Mock signal object for testing TableResultBuilder."""

    def __init__(self, data):
        self._data = data

    def iterate_roi_indices(self):
        """Return an empty list (no ROIs)."""
        return [None]  # Return [None] to indicate full signal with no ROI

    def get_data(self, roi_index=None):  # pylint: disable=unused-argument
        """Return the data array."""
        return self._data


def test_table_result_builder_init() -> None:
    """Test TableResultBuilder initialization."""
    builder = TableResultBuilder("Test Table")
    assert builder.title == "Test Table"
    assert len(builder.columns) == 0


def test_table_result_builder_add_valid_function() -> None:
    """Test adding valid functions to TableResultBuilder."""
    builder = TableResultBuilder("Test Table")

    # Add a simple function
    def mean_func(data: np.ndarray) -> float:
        return float(np.mean(data))

    builder.add(mean_func, "mean", "Mean Value")
    assert len(builder.columns) == 1
    assert builder.columns[0][0] == "mean"
    assert builder.columns[0][1] == "Mean Value"
    # pylint: disable=comparison-with-callable
    assert builder.columns[0][2] == mean_func


def test_table_result_builder_add_invalid_name() -> None:
    """Test adding function with invalid name."""
    builder = TableResultBuilder("Test Table")

    def dummy_func(data: np.ndarray) -> float:  # pylint: disable=unused-argument
        return 1.0

    # Test empty name
    with pytest.raises(AssertionError):
        builder.add(dummy_func, "", "Label")

    # Test non-string name
    with pytest.raises(AssertionError):
        builder.add(dummy_func, 123, "Label")


def test_table_result_builder_add_invalid_label() -> None:
    """Test adding function with invalid label."""
    builder = TableResultBuilder("Test Table")

    def dummy_func(data: np.ndarray) -> float:  # pylint: disable=unused-argument
        return 1.0

    # Test non-string label
    with pytest.raises(AssertionError):
        builder.add(dummy_func, "name", 123)


def test_table_result_builder_add_invalid_function() -> None:
    """Test adding non-callable object."""
    builder = TableResultBuilder("Test Table")

    with pytest.raises(AssertionError):
        builder.add("not_a_function", "name", "label")


def test_table_result_builder_add_function_no_params() -> None:
    """Test adding function with no parameters."""
    builder = TableResultBuilder("Test Table")

    def no_params_func() -> float:
        return 1.0

    with pytest.raises(ValueError, match="must accept at least one argument"):
        builder.add(no_params_func, "name", "label")


def test_table_result_builder_add_function_wrong_annotation() -> None:
    """Test adding function with wrong parameter annotation."""
    builder = TableResultBuilder("Test Table")

    # pylint: disable=unused-argument
    def wrong_annotation_func(data: str) -> float:  # Should be np.ndarray
        return 1.0

    with pytest.raises(ValueError, match="must accept a np.ndarray"):
        builder.add(wrong_annotation_func, "name", "label")


def test_table_result_builder_add_function_wrong_return_annotation() -> None:
    """Test adding function with wrong return annotation."""
    builder = TableResultBuilder("Test Table")

    # pylint: disable=unused-argument
    def wrong_return_func(data: np.ndarray) -> str:  # Should be float or int
        return "test"

    with pytest.raises(ValueError, match="must return a float or int"):
        builder.add(wrong_return_func, "name", "label")


def test_table_result_builder_compute() -> None:
    """Test TableResultBuilder.compute method."""
    builder = TableResultBuilder("Test Table")

    # Add functions
    def mean_func(data: np.ndarray) -> float:
        return float(np.mean(data))

    def sum_func(data: np.ndarray) -> float:
        return float(np.sum(data))

    builder.add(mean_func, "mean", "")
    builder.add(sum_func, "sum", "")

    # Create mock signal object
    test_data = np.array([1.0, 2.0, 3.0, 4.0])
    mock_signal = MockSignalObj(test_data)

    # Compute table
    result = builder.compute(mock_signal)

    assert result.title == "Test Table"
    assert list(result.names) == ["mean", "sum"]
    assert list(result.labels) == ["", ""]
    assert len(result.data) == 1 and len(result.data[0]) == 2
    assert result.data[0] == [2.5, 10.0]  # mean=2.5, sum=10.0
    np.testing.assert_array_equal(result.roi_indices, [NO_ROI])


def test_table_result_builder_compute_with_roi() -> None:
    """Test TableResultBuilder.compute method with ROI indices."""
    builder = TableResultBuilder("Test Table")

    # Add functions
    def mean_func(data: np.ndarray) -> float:
        return float(np.mean(data))

    builder.add(mean_func, "mean", "")

    # Create mock signal object with ROI
    class MockSignalObjWithROI:
        """Mock signal object with ROI for testing TableResultBuilder."""

        def __init__(self, data):
            self._data = data

        def iterate_roi_indices(self):
            """Return ROI indices (simulate having ROIs)."""
            return [0, 1]  # Two ROIs

        def get_data(self, roi_index=None):
            """Return subset of data based on ROI."""
            if roi_index == 0:
                return self._data[:2]  # First half
            if roi_index == 1:
                return self._data[2:]  # Second half
            return self._data  # Full data

    test_data = np.array([1.0, 2.0, 3.0, 4.0])
    mock_signal = MockSignalObjWithROI(test_data)

    # Compute table
    result = builder.compute(mock_signal)

    assert result.title == "Test Table"
    assert list(result.names) == ["mean"]
    assert len(result.data) == 3 and len(result.data[0]) == 1  # NO_ROI + 2 ROIs
    # NO_ROI: mean([1,2,3,4]) = 2.5, ROI 0: mean([1,2]) = 1.5, ROI 1: mean([3,4]) = 3.5
    assert [row[0] for row in result.data] == [2.5, 1.5, 3.5]
    np.testing.assert_array_equal(result.roi_indices, [NO_ROI, 0, 1])


# ============================= GeometryResult Tests ==============================


def test_geometry_result_init_valid_point() -> None:
    """Test GeometryResult initialization with valid point data."""
    coords = np.array([[1.0, 2.0], [3.0, 4.0]])
    roi_indices = np.array([0, 1])
    geom = GeometryResult(
        title="Test Points",
        kind=KindShape.POINT,
        coords=coords,
        roi_indices=roi_indices,
        attrs={"method": "test"},
    )
    assert geom.title == "Test Points"
    assert geom.kind == KindShape.POINT
    np.testing.assert_array_equal(geom.coords, coords)
    np.testing.assert_array_equal(geom.roi_indices, roi_indices)
    assert geom.attrs == {"method": "test"}


def test_geometry_result_init_string_kind() -> None:
    """Test GeometryResult initialization with string kind (auto-conversion)."""
    coords = np.array([[1.0, 2.0]])
    geom = GeometryResult(title="Test", kind="point", coords=coords)
    assert geom.kind == KindShape.POINT


def test_geometry_result_init_invalid_kind() -> None:
    """Test GeometryResult initialization with invalid kind."""
    coords = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="Unsupported geometry kind"):
        GeometryResult(title="Test", kind="invalid_shape", coords=coords)


def test_geometry_result_init_invalid_title() -> None:
    """Test GeometryResult initialization with invalid title."""
    coords = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="title must be a non-empty string"):
        GeometryResult(title="", kind=KindShape.POINT, coords=coords)


def test_geometry_result_init_invalid_coords_shape() -> None:
    """Test GeometryResult initialization with invalid coords shape."""
    coords = np.array([1.0, 2.0])  # 1D array
    with pytest.raises(ValueError, match="coords must be a 2-D numpy array"):
        GeometryResult(title="Test", kind=KindShape.POINT, coords=coords)


def test_geometry_result_init_point_wrong_columns() -> None:
    """Test GeometryResult initialization with wrong number of columns for point."""
    coords = np.array([[1.0, 2.0, 3.0]])  # 3 columns for point
    with pytest.raises(ValueError, match="coords for 'point' must be \\(N,2\\)"):
        GeometryResult(title="Test", kind=KindShape.POINT, coords=coords)


def test_geometry_result_init_segment_wrong_columns() -> None:
    """Test GeometryResult initialization with wrong number of columns for segment."""
    coords = np.array([[1.0, 2.0]])  # 2 columns for segment
    with pytest.raises(ValueError, match="coords for 'segment' must be \\(N,4\\)"):
        GeometryResult(title="Test", kind=KindShape.SEGMENT, coords=coords)


def test_geometry_result_init_circle_wrong_columns() -> None:
    """Test GeometryResult initialization with wrong number of columns for circle."""
    coords = np.array([[1.0, 2.0]])  # 2 columns for circle
    with pytest.raises(ValueError, match="coords for 'circle' must be \\(N,3\\)"):
        GeometryResult(title="Test", kind=KindShape.CIRCLE, coords=coords)


def test_geometry_result_init_ellipse_wrong_columns() -> None:
    """Test GeometryResult initialization with wrong number of columns for ellipse."""
    coords = np.array([[1.0, 2.0, 3.0]])  # 3 columns for ellipse
    with pytest.raises(ValueError, match="coords for 'ellipse' must be \\(N,5\\)"):
        GeometryResult(title="Test", kind=KindShape.ELLIPSE, coords=coords)


def test_geometry_result_init_rectangle_wrong_columns() -> None:
    """Test GeometryResult initialization with wrong number of columns for rectangle."""
    coords = np.array([[1.0, 2.0]])  # 2 columns for rectangle
    with pytest.raises(ValueError, match="coords for 'rectangle' must be \\(N,4\\)"):
        GeometryResult(title="Test", kind=KindShape.RECTANGLE, coords=coords)


def test_geometry_result_init_polygon_odd_columns() -> None:
    """Test GeometryResult initialization with odd number of columns for polygon."""
    coords = np.array([[1.0, 2.0, 3.0]])  # 3 columns (odd) for polygon
    with pytest.raises(
        ValueError, match="coords for 'polygon' must be \\(N,2M\\) for M vertices"
    ):
        GeometryResult(title="Test", kind=KindShape.POLYGON, coords=coords)


def test_geometry_result_init_mismatched_roi_indices() -> None:
    """Test GeometryResult initialization with mismatched ROI indices."""
    coords = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 rows
    roi_indices = np.array([0])  # 1 ROI index
    with pytest.raises(
        ValueError, match="roi_indices length must match number of coord rows"
    ):
        GeometryResult(
            title="Test", kind=KindShape.POINT, coords=coords, roi_indices=roi_indices
        )


def test_geometry_result_from_coords() -> None:
    """Test GeometryResult.from_coords factory method."""
    coords = [[1.0, 2.0], [3.0, 4.0]]
    roi_indices = [0, 1]
    geom = GeometryResult.from_coords(
        title="Test Points",
        kind=KindShape.POINT,
        coords=coords,
        roi_indices=roi_indices,
        attrs={"method": "test"},
    )
    assert geom.title == "Test Points"
    assert geom.kind == KindShape.POINT
    np.testing.assert_array_equal(geom.coords, np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_array_equal(geom.roi_indices, np.array([0, 1]))
    assert geom.attrs == {"method": "test"}


def test_geometry_result_to_dict() -> None:
    """Test GeometryResult serialization to dictionary."""
    coords = np.array([[1.0, 2.0, 3.0]])
    geom = GeometryResult(
        title="Test Circle",
        kind=KindShape.CIRCLE,
        coords=coords,
        roi_indices=np.array([NO_ROI]),
        attrs={"method": "test"},
    )
    result_dict = geom.to_dict()
    expected = {
        "schema": 1,
        "title": "Test Circle",
        "kind": "circle",
        "coords": [[1.0, 2.0, 3.0]],
        "roi_indices": [NO_ROI],
        "attrs": {"method": "test"},
    }
    assert result_dict == expected


def test_geometry_result_from_dict() -> None:
    """Test GeometryResult deserialization from dictionary."""
    data_dict = {
        "schema": 1,
        "title": "Test Circle",
        "kind": "circle",
        "coords": [[1.0, 2.0, 3.0]],
        "roi_indices": [NO_ROI],
        "attrs": {"method": "test"},
    }
    geom = GeometryResult.from_dict(data_dict)
    assert geom.title == "Test Circle"
    assert geom.kind == KindShape.CIRCLE
    np.testing.assert_array_equal(geom.coords, np.array([[1.0, 2.0, 3.0]]))
    np.testing.assert_array_equal(geom.roi_indices, np.array([NO_ROI]))
    assert geom.attrs == {"method": "test"}


def test_geometry_result_len() -> None:
    """Test GeometryResult.__len__ method."""
    coords = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    geom = GeometryResult(title="Test", kind=KindShape.POINT, coords=coords)
    assert len(geom) == 3


def test_geometry_result_rows_no_roi() -> None:
    """Test GeometryResult.rows method without ROI indices."""
    coords = np.array([[1.0, 2.0], [3.0, 4.0]])
    geom = GeometryResult(title="Test", kind=KindShape.POINT, coords=coords)

    rows = geom.rows()
    np.testing.assert_array_equal(rows, coords)


def test_geometry_result_rows_with_roi() -> None:
    """Test GeometryResult.rows method with ROI indices."""
    coords = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    roi_indices = np.array([NO_ROI, 0, 1])
    geom = GeometryResult(
        title="Test", kind=KindShape.POINT, coords=coords, roi_indices=roi_indices
    )

    # Test getting NO_ROI rows
    no_roi_rows = geom.rows(roi=None)
    np.testing.assert_array_equal(no_roi_rows, np.array([[1.0, 2.0]]))

    # Test getting specific ROI rows
    roi_0_rows = geom.rows(roi=0)
    np.testing.assert_array_equal(roi_0_rows, np.array([[3.0, 4.0]]))

    roi_1_rows = geom.rows(roi=1)
    np.testing.assert_array_equal(roi_1_rows, np.array([[5.0, 6.0]]))


def test_geometry_result_segments_lengths() -> None:
    """Test GeometryResult.segments_lengths method."""
    # Create segments: (0,0)-(3,4) and (1,1)-(5,5)
    coords = np.array([[0.0, 0.0, 3.0, 4.0], [1.0, 1.0, 5.0, 5.0]])
    geom = GeometryResult(title="Test Segments", kind=KindShape.SEGMENT, coords=coords)

    lengths = geom.segments_lengths()
    expected = [5.0, np.sqrt(32)]  # Length of (0,0)-(3,4) is 5, (1,1)-(5,5) is sqrt(32)
    np.testing.assert_allclose(lengths, expected)


def test_geometry_result_segments_lengths_wrong_kind() -> None:
    """Test GeometryResult.segments_lengths with wrong kind."""
    coords = np.array([[1.0, 2.0]])
    geom = GeometryResult(title="Test", kind=KindShape.POINT, coords=coords)

    with pytest.raises(ValueError, match="segments_lengths requires kind='segment'"):
        geom.segments_lengths()


def test_geometry_result_circles_radii() -> None:
    """Test GeometryResult.circles_radii method."""
    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    geom = GeometryResult(title="Test Circles", kind=KindShape.CIRCLE, coords=coords)

    radii = geom.circles_radii()
    expected = [3.0, 6.0]
    np.testing.assert_array_equal(radii, expected)


def test_geometry_result_circles_radii_wrong_kind() -> None:
    """Test GeometryResult.circles_radii with wrong kind."""
    coords = np.array([[1.0, 2.0]])
    geom = GeometryResult(title="Test", kind=KindShape.POINT, coords=coords)

    with pytest.raises(ValueError, match="circles_radii requires kind='circle'"):
        geom.circles_radii()


def test_geometry_result_ellipse_axes_angles() -> None:
    """Test GeometryResult.ellipse_axes_angles method."""
    coords = np.array([[1.0, 2.0, 3.0, 4.0, 0.5], [5.0, 6.0, 7.0, 8.0, 1.0]])
    geom = GeometryResult(title="Test Ellipses", kind=KindShape.ELLIPSE, coords=coords)

    a, b, theta = geom.ellipse_axes_angles()
    np.testing.assert_array_equal(a, [3.0, 7.0])
    np.testing.assert_array_equal(b, [4.0, 8.0])
    np.testing.assert_array_equal(theta, [0.5, 1.0])


def test_geometry_result_ellipse_axes_angles_wrong_kind() -> None:
    """Test GeometryResult.ellipse_axes_angles with wrong kind."""
    coords = np.array([[1.0, 2.0]])
    geom = GeometryResult(title="Test", kind=KindShape.POINT, coords=coords)

    with pytest.raises(ValueError, match="ellipse_axes_angles requires kind='ellipse'"):
        geom.ellipse_axes_angles()


# =============================== Utility Function Tests ===============================


def test_calc_table_from_data_no_roi() -> None:
    """Test calc_table_from_data without ROI masks."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    funcs = {"mean": np.mean, "sum": np.sum, "max": np.max}

    result = calc_table_from_data("Test Calculation", data, funcs)

    assert result.title == "Test Calculation"
    assert list(result.names) == ["mean", "sum", "max"]
    assert len(result.data) == 1 and len(result.data[0]) == 3
    assert result.data[0] == [2.5, 10.0, 4.0]
    np.testing.assert_array_equal(result.roi_indices, [NO_ROI])


def test_calc_table_from_data_with_roi() -> None:
    """Test calc_table_from_data with ROI masks."""
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask1 = np.array([[True, False], [False, True]])  # elements 1 and 4
    mask2 = np.array([[False, True], [True, False]])  # elements 2 and 3
    roi_masks = [mask1, mask2]

    funcs = {"mean": np.mean, "sum": np.sum}

    result = calc_table_from_data("Test ROI Calculation", data, funcs, roi_masks)

    assert result.title == "Test ROI Calculation"
    assert list(result.names) == ["mean", "sum"]
    assert len(result.data) == 2 and len(result.data[0]) == 2
    # Mask1: [1.0, 4.0] -> mean=2.5, sum=5.0
    # Mask2: [2.0, 3.0] -> mean=2.5, sum=5.0
    assert result.data[0] == [2.5, 5.0]
    assert result.data[1] == [2.5, 5.0]
    np.testing.assert_array_equal(result.roi_indices, [0, 1])


def test_concat_tables_empty() -> None:
    """Test concat_tables with empty list."""
    result = concat_tables("Empty Concat", [])
    assert result.title == "Empty Concat"
    assert len(result.names) == 0
    assert len(result.data) == 0


def test_concat_tables_single() -> None:
    """Test concat_tables with single table."""
    table = TableResult(title="Single", names=["col1", "col2"], data=[[1.0, 2.0]])
    result = concat_tables("Concat Single", [table])

    assert result.title == "Concat Single"
    assert list(result.names) == ["col1", "col2"]
    assert result.data == [[1.0, 2.0]]


def test_concat_tables_multiple() -> None:
    """Test concat_tables with multiple tables."""
    table1 = TableResult(
        title="Table1",
        names=["col1", "col2"],
        data=[[1.0, 2.0]],
        roi_indices=[0],
    )
    table2 = TableResult(
        title="Table2",
        names=["col1", "col2"],
        data=[[3.0, 4.0], [5.0, 6.0]],
        roi_indices=[1, 2],
    )

    result = concat_tables("Concatenated", [table1, table2])

    assert result.title == "Concatenated"
    assert list(result.names) == ["col1", "col2"]
    expected_data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert result.data == expected_data
    assert result.roi_indices == [0, 1, 2]


def test_concat_tables_mismatched_names() -> None:
    """Test concat_tables with mismatched column names."""
    table1 = TableResult(title="Table1", names=["col1", "col2"], data=[[1.0, 2.0]])
    table2 = TableResult(title="Table2", names=["col3", "col4"], data=[[3.0, 4.0]])

    with pytest.raises(
        ValueError, match="All TableResult objects must share the same names"
    ):
        concat_tables("Mismatched", [table1, table2])


def test_filter_table_by_roi_no_roi_indices() -> None:
    """Test filter_table_by_roi with table that has no ROI indices."""
    table = TableResult(title="No ROI", names=["col1"], data=[[1.0], [2.0]])

    # Filter for NO_ROI should keep all
    result_no_roi = filter_table_by_roi(table, None)
    assert result_no_roi.data == table.data

    # Filter for specific ROI should keep none
    result_roi = filter_table_by_roi(table, 0)
    assert len(result_roi.data) == 0


def test_filter_table_by_roi_with_roi_indices() -> None:
    """Test filter_table_by_roi with table that has ROI indices."""
    data = [[1.0], [2.0], [3.0]]
    roi_indices = [NO_ROI, 0, 1]
    table = TableResult(
        title="With ROI", names=["col1"], data=data, roi_indices=roi_indices
    )

    # Filter for NO_ROI
    result_no_roi = filter_table_by_roi(table, None)
    assert result_no_roi.data == [[1.0]]
    assert result_no_roi.roi_indices == [NO_ROI]

    # Filter for specific ROI
    result_roi0 = filter_table_by_roi(table, 0)
    assert result_roi0.data == [[2.0]]
    assert result_roi0.roi_indices == [0]


def test_concat_geometries_empty() -> None:
    """Test concat_geometries with empty list."""
    result = concat_geometries("Empty Concat", [])
    assert result.title == "Empty Concat"
    assert result.kind == KindShape.POINT
    assert result.coords.shape == (0, 2)


def test_concat_geometries_single() -> None:
    """Test concat_geometries with single geometry."""
    geom = GeometryResult(
        title="Single", kind=KindShape.POINT, coords=np.array([[1.0, 2.0]])
    )
    result = concat_geometries("Concat Single", [geom])

    assert result.title == "Concat Single"
    assert result.kind == KindShape.POINT
    np.testing.assert_array_equal(result.coords, np.array([[1.0, 2.0]]))


def test_concat_geometries_multiple() -> None:
    """Test concat_geometries with multiple geometries."""
    geom1 = GeometryResult(
        title="Geom1",
        kind=KindShape.POINT,
        coords=np.array([[1.0, 2.0]]),
        roi_indices=np.array([0]),
    )
    geom2 = GeometryResult(
        title="Geom2",
        kind=KindShape.POINT,
        coords=np.array([[3.0, 4.0], [5.0, 6.0]]),
        roi_indices=np.array([1, 2]),
    )

    result = concat_geometries("Concatenated", [geom1, geom2])

    assert result.title == "Concatenated"
    assert result.kind == KindShape.POINT
    expected_coords = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    np.testing.assert_array_equal(result.coords, expected_coords)
    np.testing.assert_array_equal(result.roi_indices, [0, 1, 2])


def test_concat_geometries_different_widths() -> None:
    """Test concat_geometries with different coordinate widths (NaN padding)."""
    geom1 = GeometryResult(
        title="Geom1",
        kind=KindShape.POLYGON,
        coords=np.array([[1.0, 2.0, 3.0, 4.0]]),  # 2 vertices
    )
    geom2 = GeometryResult(
        title="Geom2",
        kind=KindShape.POLYGON,
        coords=np.array([[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]),  # 3 vertices
    )

    result = concat_geometries("Padded", [geom1, geom2], kind=KindShape.POLYGON)

    assert result.title == "Padded"
    assert result.kind == KindShape.POLYGON
    expected_coords = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, np.nan, np.nan],  # Padded with NaN
            [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ]
    )
    np.testing.assert_array_equal(result.coords, expected_coords)


def test_concat_geometries_mismatched_kinds() -> None:
    """Test concat_geometries with mismatched kinds."""
    geom1 = GeometryResult(
        title="Geom1", kind=KindShape.POINT, coords=np.array([[1.0, 2.0]])
    )
    geom2 = GeometryResult(
        title="Geom2", kind=KindShape.CIRCLE, coords=np.array([[3.0, 4.0, 5.0]])
    )

    with pytest.raises(
        ValueError, match="All GeometryResult objects must share the same kind"
    ):
        concat_geometries("Mismatched", [geom1, geom2])


def test_filter_geometry_by_roi_no_roi_indices() -> None:
    """Test filter_geometry_by_roi with geometry that has no ROI indices."""
    geom = GeometryResult(
        title="No ROI", kind=KindShape.POINT, coords=np.array([[1.0, 2.0], [3.0, 4.0]])
    )

    # Filter for NO_ROI should keep all
    result_no_roi = filter_geometry_by_roi(geom, None)
    np.testing.assert_array_equal(result_no_roi.coords, geom.coords)

    # Filter for specific ROI should keep none
    result_roi = filter_geometry_by_roi(geom, 0)
    assert result_roi.coords.shape == (0, 2)


def test_filter_geometry_by_roi_with_roi_indices() -> None:
    """Test filter_geometry_by_roi with geometry that has ROI indices."""
    coords = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    roi_indices = np.array([NO_ROI, 0, 1])
    geom = GeometryResult(
        title="With ROI", kind=KindShape.POINT, coords=coords, roi_indices=roi_indices
    )

    # Filter for NO_ROI
    result_no_roi = filter_geometry_by_roi(geom, None)
    np.testing.assert_array_equal(result_no_roi.coords, np.array([[1.0, 2.0]]))
    np.testing.assert_array_equal(result_no_roi.roi_indices, [NO_ROI])

    # Filter for specific ROI
    result_roi0 = filter_geometry_by_roi(geom, 0)
    np.testing.assert_array_equal(result_roi0.coords, np.array([[3.0, 4.0]]))
    np.testing.assert_array_equal(result_roi0.roi_indices, [0])
