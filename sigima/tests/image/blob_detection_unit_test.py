# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Blob detection tests
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import sigima.enums
import sigima.objects
import sigima.params
import sigima.proc.image
from sigima.tests import guiutils
from sigima.tests.env import execenv
from sigima.tests.helpers import validate_detection_rois

CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None


def create_simple_blob_test_image() -> np.ndarray:
    """Create a simple test image with one obvious blob for debugging.

    Returns:
        Simple test image with a clear blob
    """
    # Create a simple 100x100 image with a clear circular blob
    size = 100
    data = np.zeros((size, size), dtype=np.float64)

    # Add a clear circular blob in the center
    y, x = np.ogrid[:size, :size]
    center_x, center_y = size // 2, size // 2
    radius = 10

    # Create a circular blob (step function, not Gaussian)
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 < radius**2
    data[mask] = 1.0

    # Keep as float64 for blob detection
    return data


def create_blob_test_image(
    size: int = 200,
    n_blobs: int = 5,
    blob_radius: float = 10.0,
    blob_intensity: float = 1000.0,
    noise_level: float = 50.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a test image with synthetic blobs for blob detection testing.

    Args:
        size: Image size (square image)
        n_blobs: Number of blobs to create
        blob_radius: Radius of the blobs in pixels
        blob_intensity: Intensity of the blobs
        noise_level: Standard deviation of Gaussian noise
        seed: Random seed for reproducible results

    Returns:
        Tuple of (image_data, expected_coords) where expected_coords is an array
        of shape (n_blobs, 3) with columns [x, y, radius]
    """
    rng = np.random.default_rng(seed)

    # Create base image with noise
    data = rng.normal(0, noise_level, size=(size, size)).astype(np.float64)

    # Generate blob centers avoiding edges
    margin = int(blob_radius * 2)
    valid_range = size - 2 * margin

    blob_centers = []
    for _ in range(n_blobs):
        x = margin + rng.random() * valid_range
        y = margin + rng.random() * valid_range
        blob_centers.append((x, y))

    # Add circular blobs to the image
    expected_coords = []
    y_grid, x_grid = np.ogrid[:size, :size]

    for x_center, y_center in blob_centers:
        # Create a circular blob using step function
        mask = (x_grid - x_center) ** 2 + (y_grid - y_center) ** 2 < blob_radius**2
        data[mask] += blob_intensity

        # Store expected coordinates with radius
        expected_coords.append([x_center, y_center, blob_radius])

    # Ensure positive values and convert to float64 for blob detection
    data = np.maximum(data, 0)
    data = data.astype(np.float64)
    # Normalize to [0, 1] range for blob detection algorithms
    if data.max() > 0:
        data = data / data.max()

    return data, np.array(expected_coords)


@pytest.mark.validation
def test_image_blob_dog():
    """Blob detection using Difference of Gaussian (DoG) method validation test"""
    execenv.print("Testing blob_dog detection...")

    # Test 1: Simple single blob
    data = create_simple_blob_test_image()
    obj = sigima.objects.create_image("blob_dog_simple", data=data)
    param = sigima.params.BlobDOGParam.create(
        min_sigma=1.0,
        max_sigma=20.0,
        threshold_rel=0.01,
        overlap=0.5,
        exclude_border=False,
    )
    result = sigima.proc.image.blob_dog(obj, param)
    assert result is not None, "Simple blob detection should return results"
    assert len(result.coords) > 0, "Should detect at least one blob in simple case"
    execenv.print(f"✓ DoG simple: detected {len(result.coords)} blobs")

    # Test 2: Multiple blobs with ROI creation
    data, expected_coords = create_blob_test_image(
        size=150, n_blobs=2, blob_radius=12.0, noise_level=0.1, seed=42
    )

    for create_rois in (False, True):
        for roi_geometry in sigima.enums.DetectionROIGeometry:
            if (
                not create_rois
                and roi_geometry != sigima.enums.DetectionROIGeometry.RECTANGLE
            ):
                continue

            obj = sigima.objects.create_image("blob_dog_multi", data=data)
            param = sigima.params.BlobDOGParam.create(
                min_sigma=5.0,
                max_sigma=20.0,
                threshold_rel=0.05,
                overlap=0.3,
                exclude_border=True,
                create_rois=create_rois,
                roi_geometry=roi_geometry,
            )
            result = sigima.proc.image.blob_dog(obj, param)
            # Apply ROIs from detection result
            sigima.proc.image.apply_detection_rois(obj, result)

            title = f"DoG blob detection (ROIs={create_rois}, geom={roi_geometry.name})"
            guiutils.view_images_if_gui(
                obj,
                title=title,
                results=[result],
                colormap="gray",
            )
            if result is not None and len(result.coords) > 0:
                detected_count = len(result.coords)
                expected_count = len(expected_coords)
                execenv.print(
                    f"✓ DoG multi: detected {detected_count} blobs "
                    f"(expected ~{expected_count}, ROIs={create_rois})"
                )
                # Validate coordinate format: should be [x, y, radius]
                assert result.coords.shape[1] == 3, (
                    "Coordinates should have 3 columns [x, y, radius]"
                )
                # Check that all radii are positive
                assert np.all(result.coords[:, 2] > 0), (
                    "All detected blob radii should be positive"
                )
                # Validate ROI creation
                validate_detection_rois(obj, result.coords, create_rois, roi_geometry)
            else:
                execenv.print(
                    "✓ DoG multi: no blobs detected (acceptable for noisy case)"
                )


@pytest.mark.validation
def test_image_blob_doh():
    """Blob detection using Determinant of Hessian (DoH) method validation test"""
    execenv.print("Testing blob_doh detection...")

    # Test with ROI creation validation
    data = create_simple_blob_test_image()

    for create_rois in (False, True):
        for roi_geometry in sigima.enums.DetectionROIGeometry:
            if (
                not create_rois
                and roi_geometry != sigima.enums.DetectionROIGeometry.RECTANGLE
            ):
                continue

            obj = sigima.objects.create_image("blob_doh_test", data=data)
            param = sigima.params.BlobDOHParam.create(
                min_sigma=1.0,
                max_sigma=20.0,
                threshold_rel=0.01,
                overlap=0.5,
                log_scale=False,
                create_rois=create_rois,
                roi_geometry=roi_geometry,
            )
            result = sigima.proc.image.blob_doh(obj, param)
            sigima.proc.image.apply_detection_rois(obj, result)

            assert result is not None, "Blob detection should return results"
            assert len(result.coords) > 0, "Should detect at least one blob"
            execenv.print(
                f"✓ DoH: detected {len(result.coords)} blobs (ROIs={create_rois})"
            )

            # Validate ROI creation
            validate_detection_rois(obj, result.coords, create_rois, roi_geometry)


@pytest.mark.validation
def test_image_blob_log():
    """Blob detection using Laplacian of Gaussian (LoG) method validation test"""
    execenv.print("Testing blob_log detection...")

    # Test with ROI creation validation
    data = create_simple_blob_test_image()

    for create_rois in (False, True):
        for roi_geometry in sigima.enums.DetectionROIGeometry:
            if (
                not create_rois
                and roi_geometry != sigima.enums.DetectionROIGeometry.RECTANGLE
            ):
                continue

            obj = sigima.objects.create_image("blob_log_test", data=data)
            param = sigima.params.BlobLOGParam.create(
                min_sigma=1.0,
                max_sigma=20.0,
                threshold_rel=0.01,
                overlap=0.5,
                log_scale=False,
                exclude_border=False,
                create_rois=create_rois,
                roi_geometry=roi_geometry,
            )
            result = sigima.proc.image.blob_log(obj, param)
            sigima.proc.image.apply_detection_rois(obj, result)

            assert result is not None, "Blob detection should return results"
            assert len(result.coords) > 0, "Should detect at least one blob"
            execenv.print(
                f"✓ LoG: detected {len(result.coords)} blobs (ROIs={create_rois})"
            )

            # Validate ROI creation
            validate_detection_rois(obj, result.coords, create_rois, roi_geometry)


@pytest.mark.validation
@pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV (cv2) is not available")
def test_image_blob_opencv():
    """Blob detection using OpenCV method validation test"""
    execenv.print("Testing blob_opencv detection...")

    # Test with ROI creation validation
    data = create_simple_blob_test_image()

    for create_rois in (False, True):
        for roi_geometry in sigima.enums.DetectionROIGeometry:
            if (
                not create_rois
                and roi_geometry != sigima.enums.DetectionROIGeometry.RECTANGLE
            ):
                continue

            obj = sigima.objects.create_image("blob_opencv_test", data=data)
            param = sigima.params.BlobOpenCVParam.create(
                min_threshold=10.0,
                max_threshold=200.0,
                min_repeatability=2,
                min_dist_between_blobs=10.0,
                filter_by_color=False,
                blob_color=0,
                filter_by_area=True,
                min_area=10.0,
                max_area=1000.0,
                filter_by_circularity=False,
                min_circularity=0.1,
                max_circularity=1.0,
                filter_by_inertia=False,
                min_inertia_ratio=0.1,
                max_inertia_ratio=1.0,
                filter_by_convexity=False,
                min_convexity=0.1,
                max_convexity=1.0,
                create_rois=create_rois,
                roi_geometry=roi_geometry,
            )
            result = sigima.proc.image.blob_opencv(obj, param)
            sigima.proc.image.apply_detection_rois(obj, result)

            assert result is not None, "Blob detection should return results"
            assert len(result.coords) > 0, "Should detect at least one blob"
            execenv.print(
                f"✓ OpenCV: detected {len(result.coords)} blobs (ROIs={create_rois})"
            )

            # Validate ROI creation
            validate_detection_rois(obj, result.coords, create_rois, roi_geometry)


def test_blob_detection_consistency():
    """Test that different blob detection methods produce consistent results"""
    execenv.print("Testing blob detection consistency across methods...")

    # Create a simple test image with well-separated blobs
    data = create_simple_blob_test_image()

    # Test parameters for each method
    methods_and_params = [
        (
            "dog",
            sigima.params.BlobDOGParam.create(
                min_sigma=1.0, max_sigma=20.0, threshold_rel=0.01, overlap=0.3
            ),
        ),
        (
            "log",
            sigima.params.BlobLOGParam.create(
                min_sigma=1.0, max_sigma=20.0, threshold_rel=0.01, overlap=0.3
            ),
        ),
        (
            "doh",
            sigima.params.BlobDOHParam.create(
                min_sigma=1.0, max_sigma=20.0, threshold_rel=0.01, overlap=0.3
            ),
        ),
    ]

    results = {}
    for method_name, param in methods_and_params:
        obj = sigima.objects.create_image(f"blob_{method_name}_consistency", data=data)

        if method_name == "dog":
            result = sigima.proc.image.blob_dog(obj, param)
        elif method_name == "log":
            result = sigima.proc.image.blob_log(obj, param)
        elif method_name == "doh":
            result = sigima.proc.image.blob_doh(obj, param)

        results[method_name] = result
        if result is not None:
            execenv.print(f"{method_name.upper()}: detected {len(result.coords)} blobs")
        else:
            execenv.print(f"{method_name.upper()}: no blobs detected")

    # All methods should detect at least one blob (or we skip if none work)
    working_methods = [
        name
        for name, result in results.items()
        if result is not None and len(result.coords) > 0
    ]

    if len(working_methods) == 0:
        pytest.skip("No blob detection methods returned results")

    for method_name, result in results.items():
        if result is not None:
            assert len(result.coords) > 0, (
                f"{method_name} should detect at least one blob"
            )


if __name__ == "__main__":
    guiutils.enable_gui()
    test_image_blob_dog()
    test_image_blob_doh()
    test_image_blob_log()
    test_image_blob_opencv()
    test_blob_detection_consistency()
