# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for convenience I/O functions with SaveToDirectoryParam.

This module tests the `write_signals` and `write_images` functions from
`sigima.io.convenience`, showcasing the usage of `SaveToDirectoryParam`
for saving multiple objects to a directory with various configuration options.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import os
import os.path as osp

import numpy as np

from sigima.io import ImageIORegistry, SignalIORegistry, write_images, write_signals
from sigima.objects import create_image, create_signal
from sigima.params import SaveToDirectoryParam
from sigima.tests.env import execenv
from sigima.tests.helpers import WorkdirRestoringTempDir


def create_test_signals() -> list:
    """Create a list of test signals for testing."""
    signals = []

    # Create signals with different titles and properties
    x = np.linspace(0, 10, 100)

    # Signal 1: Sine wave
    y1 = np.sin(x)
    signal1 = create_signal(
        title="Sine Wave",
        x=x,
        y=y1,
        metadata={"type": "sine", "frequency": "1 Hz"},
        units=("s", "V"),
    )
    signals.append(signal1)

    # Signal 2: Cosine wave
    y2 = np.cos(x * 2)
    signal2 = create_signal(
        title="Cosine Wave",
        x=x,
        y=y2,
        metadata={"type": "cosine", "frequency": "2 Hz"},
        units=("s", "A"),
    )
    signals.append(signal2)

    # Signal 3: Exponential decay
    y3 = np.exp(-x / 3)
    signal3 = create_signal(
        title="Exponential Decay",
        x=x,
        y=y3,
        metadata={"type": "exponential", "time_constant": "3 s"},
        units=("s", "V"),
    )
    signals.append(signal3)

    return signals


def create_test_images() -> list:
    """Create a list of test images for testing."""
    images = []

    # Create images with different properties

    # Image 1: Random noise
    data1 = np.random.rand(50, 50)
    image1 = create_image(
        title="Random Noise",
        data=data1,
        metadata={"type": "noise", "distribution": "uniform"},
        units=("px", "px", "intensity"),
    )
    images.append(image1)

    # Image 2: Gaussian pattern
    x, y = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
    data2 = np.exp(-(x**2 + y**2) / 2)
    image2 = create_image(
        title="Gaussian Pattern",
        data=data2,
        metadata={"type": "gaussian", "sigma": "1.0"},
        units=("mm", "mm", "intensity"),
    )
    images.append(image2)

    # Image 3: Checkerboard pattern
    data3 = np.zeros((50, 50))
    data3[::10, ::10] = 1
    data3[5::10, 5::10] = 1
    image3 = create_image(
        title="Checkerboard",
        data=data3,
        metadata={"type": "pattern", "period": "10 px"},
        units=("px", "px", "binary"),
    )
    images.append(image3)

    return images


def test_write_signals_basic() -> None:
    """Test basic functionality of write_signals with SaveToDirectoryParam."""
    execenv.print(f"{test_write_signals_basic.__doc__}:")

    with WorkdirRestoringTempDir() as tmpdir:
        # Create test signals
        signals = create_test_signals()
        execenv.print(f"  Created {len(signals)} test signals")

        # Configure SaveToDirectoryParam
        param = SaveToDirectoryParam()
        param.directory = tmpdir
        param.basename = "{title}"
        param.extension = ".h5sig"
        param.overwrite = False

        # Write signals to directory
        write_signals(param, signals)
        execenv.print(f"  Wrote signals to {tmpdir}")

        # Verify files were created
        expected_files = [
            "Sine Wave.h5sig",
            "Cosine Wave.h5sig",
            "Exponential Decay.h5sig",
        ]

        for expected_file in expected_files:
            filepath = osp.join(tmpdir, expected_file)
            assert osp.exists(filepath), f"Expected file {expected_file} not found"
            execenv.print(f"  ✓ Created: {expected_file}")

            # Verify we can read the signal back
            loaded_signal = SignalIORegistry.read(filepath)[0]
            assert loaded_signal is not None
            execenv.print(f"  ✓ Verified: {expected_file}")

    execenv.print(f"{test_write_signals_basic.__doc__}: OK")


def test_write_images_basic() -> None:
    """Test basic functionality of write_images with SaveToDirectoryParam."""
    execenv.print(f"{test_write_images_basic.__doc__}:")

    with WorkdirRestoringTempDir() as tmpdir:
        # Create test images
        images = create_test_images()
        execenv.print(f"  Created {len(images)} test images")

        # Configure SaveToDirectoryParam
        param = SaveToDirectoryParam()
        param.directory = tmpdir
        param.basename = "{title}"
        param.extension = ".h5ima"
        param.overwrite = False

        # Write images to directory
        write_images(param, images)
        execenv.print(f"  Wrote images to {tmpdir}")

        # Verify files were created
        expected_files = [
            "Random Noise.h5ima",
            "Gaussian Pattern.h5ima",
            "Checkerboard.h5ima",
        ]

        for expected_file in expected_files:
            filepath = osp.join(tmpdir, expected_file)
            assert osp.exists(filepath), f"Expected file {expected_file} not found"
            execenv.print(f"  ✓ Created: {expected_file}")

            # Verify we can read the image back
            loaded_image = ImageIORegistry.read(filepath)[0]
            assert loaded_image is not None
            execenv.print(f"  ✓ Verified: {expected_file}")

    execenv.print(f"{test_write_images_basic.__doc__}: OK")


def test_savetodir_param_formatting() -> None:
    """Test SaveToDirectoryParam formatting options and basename patterns."""
    execenv.print(f"{test_savetodir_param_formatting.__doc__}:")

    with WorkdirRestoringTempDir() as tmpdir:
        # Create test signals with specific metadata
        signals = create_test_signals()

        # Test different basename patterns
        test_cases = [
            ("{title}", ["Sine Wave.csv", "Cosine Wave.csv", "Exponential Decay.csv"]),
            (
                "{index:03d}_{title}",
                [
                    "001_Sine Wave.csv",
                    "002_Cosine Wave.csv",
                    "003_Exponential Decay.csv",
                ],
            ),
            (
                "signal_{count}_{index}",
                ["signal_3_1.csv", "signal_3_2.csv", "signal_3_3.csv"],
            ),
            (
                "{metadata[type]}_signal",
                ["sine_signal.csv", "cosine_signal.csv", "exponential_signal.csv"],
            ),
        ]

        for basename_pattern, expected_files in test_cases:
            execenv.print(f"  Testing pattern: {basename_pattern}")

            # Configure SaveToDirectoryParam
            param = SaveToDirectoryParam()
            param.directory = tmpdir
            param.basename = basename_pattern
            param.extension = ".csv"
            param.overwrite = True  # Allow overwriting for multiple test cases

            # Build filenames to verify pattern
            filenames = param.build_filenames(signals)
            execenv.print(f"    Generated filenames: {filenames}")

            # Verify expected filenames
            assert filenames == expected_files, (
                f"Expected {expected_files}, got {filenames}"
            )
            execenv.print("    ✓ Pattern matched expected filenames")

    execenv.print(f"{test_savetodir_param_formatting.__doc__}: OK")


def test_savetodir_param_collision_handling() -> None:
    """Test SaveToDirectoryParam collision handling and overwrite behavior."""
    execenv.print(f"{test_savetodir_param_collision_handling.__doc__}:")

    with WorkdirRestoringTempDir() as tmpdir:
        # Create test signals with duplicate titles to force collision
        signals = []
        x = np.linspace(0, 10, 100)

        for i in range(3):
            y = np.sin(x + i)
            signal = create_signal(
                title="Test Signal",  # Same title for all
                x=x,
                y=y,
                metadata={"index": i},
            )
            signals.append(signal)

        # Test collision handling without overwrite
        param = SaveToDirectoryParam()
        param.directory = tmpdir
        param.basename = "{title}"
        param.extension = ".h5sig"
        param.overwrite = False

        # Build filenames - should generate unique names
        filenames = param.build_filenames(signals)
        expected_files = [
            "Test Signal.h5sig",
            "Test Signal_1.h5sig",
            "Test Signal_2.h5sig",
        ]
        assert filenames == expected_files, (
            f"Expected {expected_files}, got {filenames}"
        )
        execenv.print(f"  ✓ Collision handling generated unique filenames: {filenames}")

        # Write signals and verify files exist
        write_signals(param, signals)

        for filename in expected_files:
            filepath = osp.join(tmpdir, filename)
            assert osp.exists(filepath), f"File {filename} was not created"
            execenv.print(f"  ✓ Created: {filename}")

        # Test overwrite behavior
        param.overwrite = True
        write_signals(param, signals[:1])  # Write only first signal

        # With overwrite=True, should only create the first file
        filepath = osp.join(tmpdir, "Test Signal.h5sig")
        assert osp.exists(filepath), "File should exist after overwrite"
        execenv.print("  ✓ Overwrite test passed")

    execenv.print(f"{test_savetodir_param_collision_handling.__doc__}: OK")


def test_savetodir_param_metadata_access() -> None:
    """Test SaveToDirectoryParam accessing metadata fields in basename patterns."""
    execenv.print(f"{test_savetodir_param_metadata_access.__doc__}:")

    with WorkdirRestoringTempDir() as tmpdir:
        # Create signals with rich metadata
        signals = []
        x = np.linspace(0, 10, 100)

        metadata_list = [
            {"experiment": "exp001", "sensor": "A", "temperature": "25C"},
            {"experiment": "exp002", "sensor": "B", "temperature": "30C"},
            {"experiment": "exp003", "sensor": "C", "temperature": "35C"},
        ]

        for i, metadata in enumerate(metadata_list):
            y = np.sin(x + i)
            signal = create_signal(title=f"Signal {i + 1}", x=x, y=y, metadata=metadata)
            signals.append(signal)

        # Test accessing nested metadata in basename pattern
        param = SaveToDirectoryParam()
        param.directory = tmpdir
        param.basename = (
            "{metadata[experiment]}_{metadata[sensor]}_{metadata[temperature]}"
        )
        param.extension = ".csv"
        param.overwrite = False

        # Build and verify filenames
        filenames = param.build_filenames(signals)
        expected_files = ["exp001_A_25C.csv", "exp002_B_30C.csv", "exp003_C_35C.csv"]

        assert filenames == expected_files, (
            f"Expected {expected_files}, got {filenames}"
        )
        execenv.print(f"  ✓ Metadata access in basename patterns: {filenames}")

        # Write signals to verify the full workflow
        write_signals(param, signals)

        for filename in expected_files:
            filepath = osp.join(tmpdir, filename)
            assert osp.exists(filepath), f"File {filename} was not created"
            execenv.print(f"  ✓ Created: {filename}")

    execenv.print(f"{test_savetodir_param_metadata_access.__doc__}: OK")


def test_savetodir_param_units_formatting() -> None:
    """Test SaveToDirectoryParam accessing units in basename patterns."""
    execenv.print(f"{test_savetodir_param_units_formatting.__doc__}:")

    with WorkdirRestoringTempDir() as tmpdir:
        # Create signals with different units
        signals = []
        x = np.linspace(0, 10, 100)

        units_list = [("s", "V"), ("ms", "mV"), ("us", "uV")]

        for i, (xunit, yunit) in enumerate(units_list):
            y = np.sin(x + i)
            signal = create_signal(
                title=f"Signal {i + 1}", x=x, y=y, units=(xunit, yunit)
            )
            signals.append(signal)

        # Test accessing units in basename pattern
        param = SaveToDirectoryParam()
        param.directory = tmpdir
        param.basename = "{title}_{xunit}_{yunit}"
        param.extension = ".csv"
        param.overwrite = False

        # Build and verify filenames
        filenames = param.build_filenames(signals)
        expected_files = [
            "Signal 1_s_V.csv",
            "Signal 2_ms_mV.csv",
            "Signal 3_us_uV.csv",
        ]

        assert filenames == expected_files, (
            f"Expected {expected_files}, got {filenames}"
        )
        execenv.print(f"  ✓ Units access in basename patterns: {filenames}")

        # Write signals to verify the full workflow
        write_signals(param, signals)

        for filename in expected_files:
            filepath = osp.join(tmpdir, filename)
            assert osp.exists(filepath), f"File {filename} was not created"
            execenv.print(f"  ✓ Created: {filename}")

    execenv.print(f"{test_savetodir_param_units_formatting.__doc__}: OK")


def test_savetodir_param_edge_cases() -> None:
    """Test SaveToDirectoryParam edge cases and error handling."""
    execenv.print(f"{test_savetodir_param_edge_cases.__doc__}:")

    with WorkdirRestoringTempDir() as tmpdir:
        # Test with empty signals list
        param = SaveToDirectoryParam()
        param.directory = tmpdir
        param.basename = "{title}"
        param.extension = ".h5sig"
        param.overwrite = False

        # Should handle empty list gracefully
        write_signals(param, [])
        execenv.print("  ✓ Handled empty signals list")

        # Test with signals having special characters in titles
        signals = []
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        special_titles = [
            "Signal/with/slashes",
            "Signal:with:colons",
            "Signal with spaces",
            "Signal_with_underscores",
        ]

        for title in special_titles:
            signal = create_signal(title=title, x=x, y=y)
            signals.append(signal)

        # Test filename generation with special characters
        filenames = param.build_filenames(signals)
        execenv.print(f"  Generated filenames with special chars: {filenames}")

        # All filenames should be valid (no slashes/colons in actual filenames)
        for filename in filenames:
            assert "/" not in filename or "\\" not in filename, (
                f"Invalid filename: {filename}"
            )

        execenv.print("  ✓ Handled special characters in titles")

        # Test with very long directory path
        long_subdir = "a" * 50  # Create a long subdirectory name
        long_dir = osp.join(tmpdir, long_subdir)
        os.makedirs(long_dir, exist_ok=True)

        param.directory = long_dir
        param.basename = "test"
        param.extension = ".h5sig"

        # Should handle long paths
        write_signals(param, signals[:1])
        expected_path = osp.join(long_dir, "test.h5sig")
        assert osp.exists(expected_path), "File should be created in long path"
        execenv.print("  ✓ Handled long directory path")

    execenv.print(f"{test_savetodir_param_edge_cases.__doc__}: OK")


if __name__ == "__main__":
    test_write_signals_basic()
    test_write_images_basic()
    test_savetodir_param_formatting()
    test_savetodir_param_collision_handling()
    test_savetodir_param_metadata_access()
    test_savetodir_param_units_formatting()
    test_savetodir_param_edge_cases()
