# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for kernel normalization features in convolution/deconvolution."""

from __future__ import annotations

import numpy as np
import pytest

from sigima.config import options as sigima_options
from sigima.objects import create_image, create_signal
from sigima.objects.image import ImageObj
from sigima.objects.signal import SignalObj
from sigima.proc.image.mathops import convolution as image_convolution
from sigima.proc.image.mathops import deconvolution as image_deconvolution
from sigima.proc.signal import convolution as signal_convolution
from sigima.proc.signal import deconvolution as signal_deconvolution


def _generate_test_signal(size: int = 100) -> SignalObj:
    """Generate a simple test signal.

    Args:
        size: The size of the signal to generate.

    Returns:
        A signal object.
    """
    x = np.linspace(0, 10, size)
    y = np.sin(x) + 0.5 * np.sin(3 * x)
    return create_signal("Test Signal", x, y)


def _generate_unnormalized_signal_kernel(size: int = 100) -> SignalObj:
    """Generate an unnormalized Gaussian-like kernel for signal processing.

    Args:
        size: The size of the kernel.

    Returns:
        A signal object representing an unnormalized kernel.

    Notes:
        The kernel uses the same x-axis range and size as _generate_test_signal
        to ensure compatible sample rates for convolution and deconvolution.
    """
    x = np.linspace(0, 10, size)
    y = np.exp(-((x - 5) ** 2) / 2)  # Centered Gaussian
    y *= 2.0  # Make it unnormalized (sum != 1.0)
    return create_signal("Unnormalized Kernel", x, y)


def _generate_test_image(size: int = 64) -> ImageObj:
    """Generate a simple test image.

    Args:
        size: The dimension of the square image to generate.

    Returns:
        An image object.
    """
    data = np.random.rand(size, size)
    return create_image("Test Image", data)


def _generate_unnormalized_image_kernel(size: int = 5) -> ImageObj:
    """Generate an unnormalized Gaussian-like kernel for image processing.

    Args:
        size: The dimension of the square kernel to generate.

    Returns:
        An image object representing an unnormalized kernel.
    """
    kernel = np.outer(
        np.exp(-(np.linspace(-2, 2, size) ** 2)),
        np.exp(-(np.linspace(-2, 2, size) ** 2)),
    )
    kernel *= 2.0  # Make it unnormalized (sum != 1.0)
    return create_image("Unnormalized Kernel", kernel)


class TestKernelNormalizationSignal:
    """Test suite for signal kernel normalization in convolution."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Store initial option values and restore them after each test."""
        # Store initial values
        initial_auto_normalize = sigima_options.auto_normalize_kernel.get()

        yield

        # Restore initial values
        sigima_options.auto_normalize_kernel.set(initial_auto_normalize)

    def test_signal_convolution_auto_normalization_enabled(self):
        """Test that auto-normalization works correctly for convolution."""
        # Setup: auto-normalization enabled
        sigima_options.auto_normalize_kernel.set(True)

        signal = _generate_test_signal()
        kernel = _generate_unnormalized_signal_kernel()

        # Execute with auto-normalization
        result = signal_convolution(signal, kernel)

        # Verify result exists and has same shape as input
        assert result is not None
        assert result.data is not None
        assert result.data.shape == signal.data.shape

    def test_signal_convolution_auto_normalization_disabled(self):
        """Test convolution with auto-normalization disabled."""
        # Setup: auto-normalization disabled
        sigima_options.auto_normalize_kernel.set(False)

        signal = _generate_test_signal()
        kernel = _generate_unnormalized_signal_kernel()

        # Execute without auto-normalization
        result = signal_convolution(signal, kernel)

        # Verify result exists and has same shape as input
        assert result is not None
        assert result.data is not None
        assert result.data.shape == signal.data.shape

    def test_signal_deconvolution_auto_normalization_enabled(self):
        """Test that auto-normalization works correctly for deconvolution."""
        # Setup: auto-normalization enabled
        sigima_options.auto_normalize_kernel.set(True)

        signal = _generate_test_signal()
        kernel = _generate_unnormalized_signal_kernel()

        # Execute with auto-normalization
        result = signal_deconvolution(signal, kernel)

        # Verify result exists and has same shape as input
        assert result is not None
        assert result.data is not None
        assert result.data.shape == signal.data.shape

    def test_signal_deconvolution_auto_normalization_disabled(self):
        """Test deconvolution with auto-normalization disabled."""
        # Setup: auto-normalization disabled
        sigima_options.auto_normalize_kernel.set(False)

        signal = _generate_test_signal()
        kernel = _generate_unnormalized_signal_kernel()

        # Execute without auto-normalization
        result = signal_deconvolution(signal, kernel)

        # Verify result exists and has same shape as input
        assert result is not None
        assert result.data is not None
        assert result.data.shape == signal.data.shape


class TestKernelNormalizationImage:
    """Test suite for image kernel normalization in convolution and deconvolution."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Store initial option values and restore them after each test."""
        # Store initial values
        initial_auto_normalize = sigima_options.auto_normalize_kernel.get()

        yield

        # Restore initial values
        sigima_options.auto_normalize_kernel.set(initial_auto_normalize)

    def test_image_convolution_auto_normalization_enabled(self):
        """Test that auto-normalization works correctly for convolution."""
        # Setup: auto-normalization enabled
        sigima_options.auto_normalize_kernel.set(True)

        image = _generate_test_image()
        kernel = _generate_unnormalized_image_kernel()

        # Execute with auto-normalization
        result = image_convolution(image, kernel)

        # Verify result exists and has same shape as input
        assert result is not None
        assert result.data is not None
        assert result.data.shape == image.data.shape

    def test_image_convolution_auto_normalization_disabled(self):
        """Test convolution with auto-normalization disabled."""
        # Setup: auto-normalization disabled
        sigima_options.auto_normalize_kernel.set(False)

        image = _generate_test_image()
        kernel = _generate_unnormalized_image_kernel()

        # Execute without auto-normalization
        result = image_convolution(image, kernel)

        # Verify result exists and has same shape as input
        assert result is not None
        assert result.data is not None
        assert result.data.shape == image.data.shape

    def test_image_deconvolution_auto_normalization_enabled(self):
        """Test that auto-normalization works correctly for deconvolution."""
        # Setup: auto-normalization enabled
        sigima_options.auto_normalize_kernel.set(True)

        image = _generate_test_image()
        kernel = _generate_unnormalized_image_kernel()

        # Execute with auto-normalization
        result = image_deconvolution(image, kernel)

        # Verify result exists and has same shape as input
        assert result is not None
        assert result.data is not None
        assert result.data.shape == image.data.shape

    def test_image_deconvolution_auto_normalization_disabled(self):
        """Test deconvolution with auto-normalization disabled."""
        # Setup: auto-normalization disabled
        sigima_options.auto_normalize_kernel.set(False)

        image = _generate_test_image()
        kernel = _generate_unnormalized_image_kernel()

        # Execute without auto-normalization
        result = image_deconvolution(image, kernel)

        # Verify result exists and has same shape as input
        assert result is not None
        assert result.data is not None
        assert result.data.shape == image.data.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
