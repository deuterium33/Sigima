"""
Convolution and Deconvolution Tutorial
======================================

This tutorial demonstrates the image convolution and deconvolution features
available in Sigima, showing various kernels and their effects on images.
Each section builds upon the previous one to create a comprehensive
understanding of convolution operations.

Usage:
    python convolution_example.py [--screenshot] [--unattended]

    The script supports standard execenv command line options for automated execution.

Created on September 23, 2025

@author: DataLab Platform Developers
"""

# %% Importing necessary modules

import numpy as np
import scipy.signal as sps
from guidata import qapplication

# Sigima objects and functions
from sigima.objects import create_image, create_image_from_param
from sigima.objects.image import Gauss2DParam, Zeros2DParam
from sigima.proc.image.mathops import convolution, deconvolution
from sigima.tests.vistools import view_images_side_by_side

# Create a QApplication instance
app = qapplication()

# %% Step 1: Creating test images and kernels

# Set the fixed image size for this tutorial
size = 128

# Generate a test square image with a rectangle in the center:
data = np.zeros((size, size), dtype=np.float64)
data[size // 5 : 2 * size // 5, size // 7 : 5 * size // 7] = 1.0
original_image = create_image("Original Rectangle", data)

# Generate a Gaussian kernel:
gparam = Gauss2DParam.create(height=31, width=31, sigma=2.0)
gaussian_kernel = create_image_from_param(gparam)
gaussian_kernel.title = "Gaussian Kernel (σ=2.0)"

# Generate an identity kernel (impulse response):
identity_size = 15
identity_kernel = create_image_from_param(
    Zeros2DParam.create(height=identity_size, width=identity_size)
)
identity_kernel.data[identity_size // 2, identity_size // 2] = 1.0
identity_kernel.title = "Identity Kernel"

print("✓ Test images and kernels created successfully!")
print(f"Original image shape: {original_image.data.shape}")
print(f"Gaussian kernel shape: {gaussian_kernel.data.shape}")
print(f"Identity kernel shape: {identity_kernel.data.shape}")

# %% Step 2: Basic convolution with Gaussian kernel

# Perform convolution with Gaussian kernel
convolved_gauss = convolution(original_image, gaussian_kernel)
convolved_gauss.title = "Convolved with Gaussian"

# Compare with scipy implementation
expected_result = sps.convolve(
    original_image.data, gaussian_kernel.data, mode="same", method="auto"
)
print("\n✓ Convolution completed!")
max_diff = np.max(np.abs(convolved_gauss.data - expected_result))
print(f"Max difference from scipy: {max_diff:.2e}")

# Visualize the results
view_images_side_by_side(
    [original_image, gaussian_kernel, convolved_gauss],
    ["Original Image", "Gaussian Kernel (σ=2.0)", "Convolved Result"],
    title="Gaussian Convolution Example",
)

# %% Step 3: Identity convolution (should preserve original image)

# Perform convolution with identity kernel
convolved_identity = convolution(original_image, identity_kernel)
convolved_identity.title = "Convolved with Identity"

# This should be nearly identical to the original
difference = np.max(np.abs(convolved_identity.data - original_image.data))
print("\n✓ Identity convolution completed!")
print(f"Max difference from original: {difference:.2e}")

# Visualize the identity convolution
view_images_side_by_side(
    [original_image, identity_kernel, convolved_identity],
    ["Original Image", "Identity Kernel", "Convolved with Identity"],
    title="Identity Convolution Example",
)

# %% Step 4: Deconvolution with identity kernel

# Start with the convolved image and deconvolve using identity kernel
deconvolved_identity = deconvolution(convolved_identity, identity_kernel)
deconvolved_identity.title = "Deconvolved (Identity)"

# Check how well we recovered the original
recovery_error = np.max(np.abs(deconvolved_identity.data - original_image.data))
print("\n✓ Identity deconvolution completed!")
print(f"Recovery error: {recovery_error:.2e}")

# Visualize the deconvolution process
view_images_side_by_side(
    [original_image, convolved_identity, deconvolved_identity],
    ["Original", "Convolved", "Deconvolved"],
    title="Identity Deconvolution Example",
)

# %% Step 5: Advanced deconvolution with Gaussian kernel

# Create a Gaussian kernel with smaller sigma for better deconvolution:
gparam.sigma = 1.5
deconv_gaussian = create_image_from_param(gparam)
deconv_gaussian.title = "Gaussian Kernel (σ=1.5)"

# Convolve the original image with this kernel:
large_convolved = convolution(original_image, deconv_gaussian)
large_convolved.title = "Convolved Image"

# Attempt deconvolution to recover the original:
large_deconvolved = deconvolution(large_convolved, deconv_gaussian)
large_deconvolved.title = "Deconvolved Result"

print("\n✓ Gaussian deconvolution completed!")
orig_min, orig_max = np.min(original_image.data), np.max(original_image.data)
deconv_min, deconv_max = np.min(large_deconvolved.data), np.max(large_deconvolved.data)
print(f"Original image range: [{orig_min:.3f}, {orig_max:.3f}]")
print(f"Deconvolved image range: [{deconv_min:.3f}, {deconv_max:.3f}]")

# Visualize the full deconvolution process
view_images_side_by_side(
    [original_image, deconv_gaussian, large_convolved, large_deconvolved],
    ["Original", "Gaussian Kernel", "Convolved", "Deconvolved"],
    title="Gaussian Deconvolution Example",
)

# %% Step 6: Exploring different kernel sizes and sigmas

# Create kernels with different sigma parameters:
gparam.sigma = 0.8
small_sigma = create_image_from_param(gparam)
gparam.sigma = 2.0
medium_sigma = create_image_from_param(gparam)
gparam.sigma = 4.0
large_sigma = create_image_from_param(gparam)

# Convolve the original image with each kernel:
conv_small = convolution(original_image, small_sigma)
conv_medium = convolution(original_image, medium_sigma)
conv_large = convolution(original_image, large_sigma)

print("\n✓ Multiple kernel comparison completed!")

# Show the effect of different sigma values
view_images_side_by_side(
    [small_sigma, medium_sigma, large_sigma],
    ["Kernel σ=0.8", "Kernel σ=2.0", "Kernel σ=4.0"],
    title="Gaussian Kernels with Different Sigma Values",
)

view_images_side_by_side(
    [conv_small, conv_medium, conv_large],
    ["Convolved σ=0.8", "Convolved σ=2.0", "Convolved σ=4.0"],
    title="Convolution Results with Different Sigma Values",
)

# %% Step 7: Creating custom kernels

# Edge detection kernel (Sobel-like):
edge_data = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
edge_kernel = create_image("Edge Detection Kernel", edge_data)

# Sharpening kernel:
sharpen_data = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)
sharpen_kernel = create_image("Sharpening Kernel", sharpen_data)

# Apply custom kernels to the original image:
edge_result = convolution(original_image, edge_kernel)
edge_result.title = "Edge Detection"

sharpen_result = convolution(original_image, sharpen_kernel)
sharpen_result.title = "Sharpened"

print("\n✓ Custom kernel convolutions completed!")

# Visualize custom kernel results
view_images_side_by_side(
    [edge_kernel, sharpen_kernel],
    ["Edge Detection Kernel", "Sharpening Kernel"],
    title="Custom Convolution Kernels",
)

view_images_side_by_side(
    [original_image, edge_result, sharpen_result],
    ["Original", "Edge Detection", "Sharpened"],
    title="Custom Kernel Convolution Results",
)

# %% Summary and conclusions

print("\n" + "=" * 60)
print("CONVOLUTION TUTORIAL SUMMARY")
print("=" * 60)
print("✓ Created test images and various kernels")
print("✓ Demonstrated basic Gaussian convolution")
print("✓ Showed identity kernel behavior")
print("✓ Performed deconvolution operations")
print("✓ Explored different kernel parameters")
print("✓ Applied custom edge detection and sharpening kernels")
print("\nKey Takeaways:")
print("• Larger sigma values create more blurring")
print("• Identity kernels preserve the original image")
print("• Deconvolution can recover original features (with limitations)")
print("• Custom kernels enable specialized image processing effects")

# Final comparison showing the complete pipeline
view_images_side_by_side(
    [original_image, gaussian_kernel, convolved_gauss, large_deconvolved],
    ["Original", "Gaussian Kernel", "Convolved", "Deconvolved"],
    title="Complete Convolution-Deconvolution Pipeline",
)
