# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for reading and writing coordinated text format files.
"""

import os
import os.path as osp
import tempfile

import numpy as np

import sigima.io
import sigima.objects
from sigima.io.image.formats import CoordinatedTextFileReader
from sigima.tests.env import execenv
from sigima.tests.helpers import (
    WorkdirRestoringTempDir,
    check_array_result,
    get_test_fnames,
)


def test_read_image_basic():
    """Basic test to read a simple coordinated text image file"""
    path = get_test_fnames("coordinated_text/image.txt")[0]
    imgs = CoordinatedTextFileReader.read_images(path)
    assert len(imgs) == 1, f"Expected 1 image, got {len(imgs)}"
    arr = np.asarray(imgs[0].data)
    expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    check_array_result("test read image.txt", arr, expected)


def test_read_image_with_unit():
    """Test to read a coordinated text image file with units in metadata"""
    path = get_test_fnames("coordinated_text/image_with_unit.txt")[0]
    imgs = CoordinatedTextFileReader.read_images(path)
    assert len(imgs) == 1, f"Expected 1 image, got {len(imgs)}"
    img = imgs[0]
    # units should come from metadata (X, Y, Z)
    check_array_result(
        "test read image_with_unit.txt",
        np.asarray(img.data),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )

    assert img.xunit == "mm", (
        f"X unit not read correctly: {img.xunit} given but mm expected"
    )
    assert img.yunit == "nm", (
        f"Y unit not read correctly: {img.yunit} given but nm expected"
    )
    assert img.zunit == "A", (
        f"Z unit not read correctly: {img.zunit} given but A expected"
    )


def test_read_image_with_nan():
    """Test to read a coordinated text image file with NaN values"""
    path = get_test_fnames("coordinated_text/image_with_nan.txt")[0]
    imgs = CoordinatedTextFileReader.read_images(path)
    assert len(imgs) == 1, f"Expected 1 image, got {len(imgs)}"
    arr = np.asarray(imgs[0].data)
    # expected NaN positions from the test file
    assert np.isnan(arr[0, 2]), "expected NaN at position (0,2), got {arr[0,2]}"
    assert np.isnan(arr[1, 0]), "expected NaN at position (1,0), got {arr[1,0]}"
    assert np.isnan(arr[1, 1]), "expected NaN at position (1,1), got {arr[1,1]}"
    # and a valid value
    assert arr[0, 0] == 1, "expected 1 at position (0,0), got {arr[0,0]}"
    assert arr[1, 2] == 6, "expected 6 at position (1,2), got {arr[1,2]}"


def test_read_complex_image_and_error():
    """Test to read a coordinated text complex image file with error image"""
    path = get_test_fnames("coordinated_text/complex_image.txt")[0]
    imgs = CoordinatedTextFileReader.read_images(path)
    # should return main image and error image
    assert len(imgs) == 2, f"Expected 2 images, got {len(imgs)}"
    img, img_err = imgs[0], imgs[1]
    # data should be complex
    assert np.iscomplexobj(np.asarray(img.data)), (
        f"expected complex data, got {np.asarray(img.data).dtype}"
    )
    assert np.iscomplexobj(np.asarray(img_err.data)), (
        f"expected complex data, got {np.asarray(img_err.data).dtype}"
    )
    # check first element values (from first data line)
    first_val = img.data[0, 0]
    expected = complex(3.678795e-01, 3.678795e-01)
    np.testing.assert_allclose(first_val, expected, rtol=1e-7, atol=1e-12)
    first_err = img_err.data[0, 0]
    expected_err = complex(1.839397e-01, -3.678795e-01)
    np.testing.assert_allclose(first_err, expected_err, rtol=1e-7, atol=1e-12)


def test_read_nonuniform_coordinates():
    """Test reading coordinated text file with non-uniform coordinates"""
    # Create a temporary test file with non-uniform coordinates
    test_content = """# Created on 2024-10-10 12:00:00.000000
# By Test Script
# Using matrislib 3.0.0test
# nx : 3
# ny : 2
# X : X-axis (mm)
# Y : Y-axis (mm)
# Z : Z-value (units)
0.000000	0.000000	1.000000
1.500000	0.000000	2.000000
4.000000	0.000000	3.000000
0.000000	3.000000	4.000000
1.500000	3.000000	5.000000
4.000000	3.000000	6.000000
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(test_content)
        temp_filename = f.name

    try:
        imgs = CoordinatedTextFileReader.read_images(temp_filename)
        assert len(imgs) == 1, f"Expected 1 image, got {len(imgs)}"

        img = imgs[0]

        # Should detect non-uniform coordinates
        assert not img.is_uniform_coords, "Should detect non-uniform coordinates"

        # Check coordinate arrays
        expected_x = np.array([0.0, 1.5, 4.0])
        expected_y = np.array([0.0, 3.0])

        np.testing.assert_allclose(img.xcoords, expected_x, rtol=1e-10)
        np.testing.assert_allclose(img.ycoords, expected_y, rtol=1e-10)

        # Check data shape and values
        assert img.data.shape == (2, 3), f"Expected shape (2, 3), got {img.data.shape}"
        expected_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_allclose(img.data, expected_data)

        # Check coordinate conversion functionality
        img.switch_coords_to("uniform")
        assert img.is_uniform_coords, "Should convert to uniform coordinates"

        # After conversion to uniform, switching back creates uniform grid
        img.switch_coords_to("non-uniform")
        assert not img.is_uniform_coords, "Should convert back to non-uniform"

        # The new non-uniform coordinates will be a uniform grid, not the original
        # This is expected behavior since uniform conversion loses original spacing
        assert len(img.xcoords) == len(expected_x), (
            "Should have same number of X coordinates"
        )
        assert len(img.ycoords) == len(expected_y), (
            "Should have same number of Y coordinates"
        )

    finally:
        os.unlink(temp_filename)


def test_nonuniform_coordinates_io() -> None:
    """Test I/O (read and write) for coordinated text format
    with non-uniform coordinates
    """
    execenv.print(f"{test_nonuniform_coordinates_io.__doc__}:")

    # Create a test image with non-uniform coordinates
    title = "Non-uniform Coordinates Test"
    data = np.random.rand(10, 10)
    metadata = {"test_key": "test_value", "coordinate_type": "non-uniform"}
    units = ("μm", "μm", "counts")
    labels = ("X position", "Y position", "Intensity")

    # Create the image object
    orig_image = sigima.objects.create_image(
        title=title,
        data=data,
        metadata=metadata,
        units=units,
        labels=labels,
    )

    # Set non-uniform coordinates
    xcoords = np.linspace(0, 1, 10)
    ycoords = np.linspace(0, 1, 10) ** 2  # Quadratic spacing
    orig_image.set_coords(xcoords=xcoords, ycoords=ycoords)

    # Verify the original image has non-uniform coordinates
    assert not orig_image.is_uniform_coords
    assert np.array_equal(orig_image.xcoords, xcoords)
    assert np.array_equal(orig_image.ycoords, ycoords)

    execenv.print("  ✓ Created non-uniform coordinate image")

    with WorkdirRestoringTempDir() as tmpdir:
        # Test coordinated text CSV format writing (the main focus of this test)
        csv_filename = osp.join(tmpdir, "test_nonuniform_coords.csv")

        # Save to coordinated text CSV format
        sigima.io.write_image(csv_filename, orig_image)
        execenv.print(f"  ✓ Saved to coordinated text CSV: {csv_filename}")

        # Read back from coordinated text CSV
        loaded_csv_image = sigima.io.read_image(csv_filename)
        execenv.print(f"  ✓ Loaded from coordinated text CSV: {csv_filename}")

        # Verify the loaded CSV image
        assert isinstance(loaded_csv_image, sigima.objects.ImageObj)
        assert loaded_csv_image.title == osp.basename(csv_filename)

        # For CSV files, use allclose instead of array_equal due to
        # floating-point precision loss during text serialization
        assert np.allclose(loaded_csv_image.data, orig_image.data, atol=1e-10)
        csv_units = (
            loaded_csv_image.xunit,
            loaded_csv_image.yunit,
            loaded_csv_image.zunit,
        )
        assert csv_units == units
        csv_labels = (
            loaded_csv_image.xlabel,
            loaded_csv_image.ylabel,
            loaded_csv_image.zlabel,
        )
        assert csv_labels == labels

        # Most importantly: verify coordinate system is preserved
        # Use allclose for coordinates too due to text serialization precision
        assert not loaded_csv_image.is_uniform_coords
        assert np.allclose(loaded_csv_image.xcoords, xcoords, atol=1e-10)
        assert np.allclose(loaded_csv_image.ycoords, ycoords, atol=1e-10)

        execenv.print("  ✓ Coordinated text CSV round-trip verification successful")

    execenv.print(f"{test_nonuniform_coordinates_io.__doc__}: OK")


def test_uniform_coordinates_io() -> None:
    """Test I/O (read and write) for coordinated text format
    with uniform coordinates
    """
    execenv.print(f"{test_uniform_coordinates_io.__doc__}:")

    # Create a test image with uniform coordinates
    title = "Uniform Coordinates Test"
    data = np.random.rand(10, 10)
    metadata = {"test_key": "test_value", "coordinate_type": "uniform"}
    units = ("mm", "mm", "intensity")
    labels = ("X position", "Y position", "Signal")

    # Create the image object
    orig_image = sigima.objects.create_image(
        title=title,
        data=data,
        metadata=metadata,
        units=units,
        labels=labels,
    )

    # Set uniform coordinates (dx, dy, x0, y0)
    dx, dy, x0, y0 = 0.5, 0.3, 10.0, 20.0
    orig_image.set_uniform_coords(dx, dy, x0=x0, y0=y0)

    # Verify the original image has uniform coordinates
    assert orig_image.is_uniform_coords
    assert orig_image.dx == dx
    assert orig_image.dy == dy
    assert orig_image.x0 == x0
    assert orig_image.y0 == y0

    execenv.print("  ✓ Created uniform coordinate image")

    with WorkdirRestoringTempDir() as tmpdir:
        # Test text CSV format writing with uniform coordinates
        # Note: uniform coordinates written as plain CSV (not coordinated text)
        csv_filename = osp.join(tmpdir, "test_uniform_coords.csv")

        # Save to CSV format
        sigima.io.write_image(csv_filename, orig_image)
        execenv.print(f"  ✓ Saved to CSV: {csv_filename}")

        # Read back from CSV
        loaded_csv_image = sigima.io.read_image(csv_filename)
        execenv.print(f"  ✓ Loaded from CSV: {csv_filename}")

        # Verify the loaded CSV image
        assert isinstance(loaded_csv_image, sigima.objects.ImageObj)
        assert loaded_csv_image.title == osp.basename(csv_filename)

        # For CSV files, use allclose instead of array_equal due to
        # floating-point precision loss during text serialization
        assert np.allclose(loaded_csv_image.data, orig_image.data, atol=1e-10)

        # Note: plain CSV format does NOT preserve units and labels
        # So we don't check those here

        # Important: verify coordinate system is NOT preserved for plain CSV
        # Plain CSV files lose coordinate info, revert to default uniform coords
        assert loaded_csv_image.is_uniform_coords
        # Default uniform coordinates (dx=1, dy=1, x0=0, y0=0)
        assert loaded_csv_image.dx == 1.0
        assert loaded_csv_image.dy == 1.0
        assert loaded_csv_image.x0 == 0.0
        assert loaded_csv_image.y0 == 0.0

        execenv.print("  ✓ CSV round-trip verification successful")
        execenv.print("  ⚠ Note: Plain CSV does not preserve coordinate information")

    execenv.print(f"{test_uniform_coordinates_io.__doc__}: OK")


def test_write_with_nan_values() -> None:
    """Test writing coordinated text format with NaN values in data"""
    execenv.print(f"{test_write_with_nan_values.__doc__}:")

    # Create test image with NaN values
    data = np.array(
        [[1.0, 2.0, np.nan], [4.0, np.nan, 6.0], [np.nan, 8.0, 9.0]], dtype=float
    )
    title = "NaN Test"
    units = ("mm", "mm", "V")
    labels = ("X", "Y", "Voltage")

    orig_image = sigima.objects.create_image(
        title=title, data=data, units=units, labels=labels
    )

    # Set non-uniform coordinates to trigger coordinated text format
    xcoords = np.array([0.0, 1.5, 4.0])
    ycoords = np.array([0.0, 3.0, 7.0])
    orig_image.set_coords(xcoords=xcoords, ycoords=ycoords)

    execenv.print("  ✓ Created image with NaN values")

    with WorkdirRestoringTempDir() as tmpdir:
        csv_filename = osp.join(tmpdir, "test_nan.csv")

        # Write with NaN values
        sigima.io.write_image(csv_filename, orig_image)
        execenv.print(f"  ✓ Saved to CSV with NaN values: {csv_filename}")

        # Read back
        loaded_image = sigima.io.read_image(csv_filename)
        execenv.print(f"  ✓ Loaded from CSV: {csv_filename}")

        # Verify NaN values are preserved
        assert np.allclose(
            loaded_image.data, orig_image.data, atol=1e-10, equal_nan=True
        )
        # Count NaN values
        orig_nan_count = np.isnan(orig_image.data).sum()
        loaded_nan_count = np.isnan(loaded_image.data).sum()
        assert orig_nan_count == loaded_nan_count == 3
        execenv.print(f"    ✓ NaN values preserved ({loaded_nan_count} NaNs)")

        # Verify coordinates
        assert np.allclose(loaded_image.xcoords, xcoords, atol=1e-10)
        assert np.allclose(loaded_image.ycoords, ycoords, atol=1e-10)

    execenv.print(f"{test_write_with_nan_values.__doc__}: OK")


if __name__ == "__main__":
    test_read_image_basic()
    test_read_image_with_unit()
    test_read_image_with_nan()
    test_read_complex_image_and_error()
    test_read_nonuniform_coordinates()
    test_nonuniform_coordinates_io()
    test_uniform_coordinates_io()
    test_write_with_nan_values()
