# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Creating Signals and Images in Sigima
=====================================

This example demonstrates the three main approaches for creating signals and images
in Sigima:

1. **Synthetic data generation**: Using built-in parameter classes to create standard
    signal and image types (Gaussian, sine waves, random distributions, etc.)
2. **Loading from files**: Importing data from various file formats
3. **From NumPy arrays**: Creating objects directly from existing arrays

Each method has its use cases, and Sigima provides a consistent interface for working
with data regardless of its origin.

For visualization, we use helper functions from the ``sigima.tests.vistools`` module.
This allows us to focus on Sigima's functionality rather than visualization details.
"""

# %%
# Importing necessary modules
# ---------------------------
# First of all, we need to import the required modules.

import numpy as np

import sigima
from sigima.tests import helpers, vistools

# %%
# Method 1: Creating signals from synthetic parameters
# ----------------------------------------------------
#
# Sigima provides built-in generators for common signal types. This is the most
# convenient method when you need standard mathematical functions or random
# distributions.
#
# Available signal types include:
#
# - Mathematical functions: Gaussian, Lorentzian, Sinc, Sine, Cosine, etc.
# - Random distributions: Normal, Uniform, Poisson
# - Standard waveforms: Square, Sawtooth, Triangle
# - Special functions: Planck (blackbody), Linear chirp, Step, Exponential

# Create a Gaussian signal using parameters
gaussian_param = sigima.create_signal_parameters(
    sigima.SignalTypes.GAUSS,  # Signal type
    size=500,  # Number of points
    xmin=-10.0,  # Minimum x value
    xmax=10.0,  # Maximum x value
    a=2.0,  # Amplitude
    mu=0.0,  # Center position
    sigma=1.5,  # Width (standard deviation)
)
signal_synthetic = sigima.create_signal_from_param(gaussian_param)
signal_synthetic.title = "Synthetic Gaussian Signal"
signal_synthetic.units = ("µm", "a.u.")
signal_synthetic.labels = ("Position", "Intensity")

# Create a sinusoidal signal
sine_param = sigima.create_signal_parameters(
    sigima.SignalTypes.SINE, size=500, xmin=0.0, xmax=2.0, freq=10.0, a=1.5
)
signal_sine = sigima.create_signal_from_param(sine_param)
signal_sine.title = "Synthetic Sine Wave"
signal_sine.units = ("s", "V")
signal_sine.labels = ("Time", "Voltage")

print("✓ Synthetic signals created")
print(f"  - {signal_synthetic.title}: {signal_synthetic.y.shape[0]} points")
print(f"  - {signal_sine.title}: {signal_sine.y.shape[0]} points")

# Visualize synthetic signals
vistools.view_curves(
    [signal_synthetic, signal_sine],
    title="Method 1: Synthetic Signals",
    object_name="synthetic_signals",
)

# %%
# Method 2: Loading signals from files
# ------------------------------------
#
# Sigima can read signals from various file formats, automatically detecting the
# format and extracting metadata when available.
#
# Supported formats include:
#
# - Text files: CSV, TXT (with automatic delimiter detection)
# - Scientific formats: HDF5 (.h5sig), MAT-Files (.mat), NumPy (.npy)
# - Specialized: MCA spectrum files (.mca), FT-Lab (.sig)

# Load a real spectrum from a text file
# This is a paracetamol (acetaminophen) UV-Vis absorption spectrum
spectrum_file = helpers.get_test_fnames("paracetamol.txt")[0]
signal_from_file = sigima.read_signal(spectrum_file)
signal_from_file.title = "Paracetamol Spectrum (from file)"

# Visualize signal loaded from text file
vistools.view_curves(
    signal_from_file,
    title="Signal from Text File",
    object_name="signal_from_txt",
)

# Load another signal from a CSV file with multiple curves
csv_file = helpers.get_test_fnames("oscilloscope.csv")[0]
signals_from_csv = sigima.read_signals(csv_file)
# CSV files contain multiple signals; we'll show one
signal_from_csv = signals_from_csv[1]

signal_from_csv.title = "Oscilloscope Data (from CSV)"

print("\n✓ Signals loaded from files")
print(f"  - {signal_from_file.title}: {signal_from_file.y.shape[0]} points")
print(f"  - {signal_from_csv.title}: {signal_from_csv.y.shape[0]} points")

# Visualize signal loaded from csv file

vistools.view_curves(
    signal_from_csv,
    title="Signal from CSV File",
    object_name="signal_from_csv",
)

# %%
# It is interesting to remark here that when importing data from files,
# Sigima automatically extracts and preserves metadata when possible.
# This includes:
#
# - **Axis labels and units**: Column headers from CSV files, variable names from
#   MAT-Files, etc.
# - **Acquisition parameters**: DICOM headers, instrument settings, timestamps
# - **Physical coordinates**: Pixel spacing, origin coordinates when stored in the file
#
# The extracted metadata is seamlessly integrated into the signal or image object,
# making it available for processing, analysis, and visualization without manual
# configuration.

# %%
# Method 3: Creating signals from NumPy arrays
# ---------------------------------------------
#
# When you already have data in NumPy arrays (from calculations, other libraries,
# or custom data sources), you can wrap them in Sigima signal objects to benefit
# from metadata handling and processing functions.

# Create custom data: a damped oscillation
t = np.linspace(0, 5, 1000)
damping = np.exp(-0.5 * t)
oscillation = np.sin(2 * np.pi * 3 * t)
y_damped = damping * oscillation

signal_from_array = sigima.create_signal(
    title="Damped Oscillation (from array)",
    x=t,
    y=y_damped,
    units=("s", ""),
    labels=("Time", "Amplitude"),
)

# Create another signal: a noisy linear ramp
x_ramp = np.linspace(0, 100, 200)
rng = np.random.default_rng(42)
y_ramp = 0.5 * x_ramp + rng.normal(0, 2, 200)

signal_noisy_ramp = sigima.create_signal(
    title="Noisy Ramp (from array)",
    x=x_ramp,
    y=y_ramp,
    units=("mm", "°C"),
    labels=("Position", "Temperature"),
)

print("\n✓ Signals created from NumPy arrays")
print(f"  - {signal_from_array.title}: {signal_from_array.y.shape[0]} points")
print(f"  - {signal_noisy_ramp.title}: {signal_noisy_ramp.y.shape[0]} points")

# Visualize signals created from NumPy arrays
vistools.view_curves(
    [signal_from_array, signal_noisy_ramp],
    title="Method 3: Signals from NumPy Arrays",
    object_name="signals_from_arrays",
)

# %%
# Method 1: Creating images from synthetic parameters
# ----------------------------------------------------
#
# Similar to signals, Sigima can generate synthetic images using parameter classes.
#
# Available image types include:
#
# - Distributions: Normal (Gaussian noise), Uniform, Poisson
# - Analytical functions: 2D Gaussian, 2D ramp (bilinear form)
# - Blank images: Zeros

# Create a 2D Gaussian image
gaussian_img_param = sigima.create_image_parameters(
    sigima.ImageTypes.GAUSS,
    height=300,
    width=300,
)
# Customize the Gaussian parameters
gaussian_img_param.x0 = 0.0  # Center x position
gaussian_img_param.y0 = 0.0  # Center y position
gaussian_img_param.sigma = 1.5  # Width
gaussian_img_param.a = 1000.0  # Amplitude

image_synthetic = sigima.create_image_from_param(gaussian_img_param)
image_synthetic.title = "Synthetic 2D Gaussian"
image_synthetic.units = ("µm", "µm", "counts")
image_synthetic.labels = ("X", "Y", "Intensity")

# Create a ramp image (gradient)
ramp_param = sigima.create_image_parameters(
    sigima.ImageTypes.RAMP,
    height=200,
    width=200,
)
ramp_param.x0 = -5.0
ramp_param.y0 = -5.0
ramp_param.dx = 0.5  # X slope
ramp_param.dy = 0.3  # Y slope

image_ramp = sigima.create_image_from_param(ramp_param)
image_ramp.title = "Synthetic 2D Ramp"
image_ramp.units = ("mm", "mm", "a.u.")
image_ramp.labels = ("X", "Y", "Value")

print("\n✓ Synthetic images created")
print(f"  - {image_synthetic.title}: {image_synthetic.data.shape}")
print(f"  - {image_ramp.title}: {image_ramp.data.shape}")

# Visualize synthetic images
vistools.view_images_side_by_side(
    [image_synthetic, image_ramp],
    titles=["Synthetic Gaussian", "Synthetic Ramp"],
)

# %%
# Method 2: Loading images from files
# ------------------------------------
#
# Sigima supports a wide range of image file formats, both common and scientific.
#
# Supported formats include:
#
# - Common formats: BMP, JPEG, PNG, TIFF, JPEG 2000
# - Scientific formats: DICOM, Andor SIF, Spiricon, Dürr NDT
# - Data formats: NumPy (.npy), MATLAB (.mat), HDF5 (.h5img)
# - Text formats: CSV, TXT, ASC (with coordinate support)

# Load an image from a JPEG file
jpeg_file = helpers.get_test_fnames("fiber.jpg")[0]
image_from_jpeg = sigima.read_image(jpeg_file)
image_from_jpeg.title = "Fiber Image (from JPEG)"

# Load an image from a NumPy file
npy_file = helpers.get_test_fnames("flower.npy")[0]
image_from_npy = sigima.read_image(npy_file)
image_from_npy.title = "Test Image (from NumPy)"

print("\n✓ Images loaded from files")
print(f"  - {image_from_jpeg.title}: {image_from_jpeg.data.shape}")
print(f"  - {image_from_npy.title}: {image_from_npy.data.shape}")

# Visualize images loaded from files
vistools.view_images_side_by_side(
    [image_from_jpeg, image_from_npy],
    titles=["From JPEG", "From NumPy File"],
)

# %%
# Method 3: Creating images from NumPy arrays
# --------------------------------------------
#
# Convert existing NumPy arrays into Sigima image objects to add metadata,
# coordinate systems, and enable advanced processing.

# Create a synthetic pattern: interference fringes
size = 256
x = np.linspace(-10, 10, size)
y = np.linspace(-10, 10, size)
X, Y = np.meshgrid(x, y)

# Interference pattern
pattern = np.cos(2 * np.pi * X / 3) * np.cos(2 * np.pi * Y / 3)
pattern = ((pattern + 1) / 2 * 255).astype(np.uint8)

image_from_array = sigima.create_image(
    title="Interference Pattern (from array)",
    data=pattern,
    units=("mm", "mm", "intensity"),
    labels=("X", "Y", "Signal"),
)

# Create another image: radial gradient with noise
r = np.sqrt(X**2 + Y**2)
radial = np.exp(-(r**2) / 20)
rng = np.random.default_rng(123)
radial = radial + rng.normal(0, 0.05, radial.shape)
radial = np.clip(radial, 0, 1)

image_radial = sigima.create_image(
    title="Radial Gradient (from array)",
    data=radial.astype(np.float32),
    units=("µm", "µm", "a.u."),
    labels=("X", "Y", "Amplitude"),
)

print("\n✓ Images created from NumPy arrays")
print(f"  - {image_from_array.title}: {image_from_array.data.shape}")
print(f"  - {image_radial.title}: {image_radial.data.shape}")

# Visualize images created from NumPy arrays
vistools.view_images_side_by_side(
    [image_from_array, image_radial],
    titles=["Interference Pattern", "Radial Gradient"],
)

# %%
# Summary
# -------
#
# This example demonstrated the three main ways to create signals and images in Sigima:
#
# 1. **Synthetic generation**: Fast creation of standard mathematical functions and
#    distributions using parameter classes. Perfect for testing and simulation.
#
# 2. **File loading**: Read data from various scientific and common file formats,
#    with automatic format detection and metadata extraction. Essential for working
#    with experimental data.
#
# 3. **NumPy array conversion**: Wrap existing array data with Sigima's rich metadata
#    and processing capabilities. Ideal for custom workflows and integration with
#    other Python libraries.
#
# All three methods produce equivalent Sigima objects that can be processed, analyzed,
# and visualized using the same set of tools and functions. Choose the method that
# best fits your workflow and data source.
