# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Laser Beam Size Measurement Example
===================================

This example demonstrates comprehensive laser beam analysis techniques following
the laser beam tutorial workflow. It shows how to load multiple laser beam images,
analyze background noise with histograms, apply proper clipping, detect beam
centroids, extract line and radial profiles, compute FWHM measurements, and
track beam size evolution along the propagation axis.

The script demonstrates advanced optical beam characterization workflows commonly
used in laser physics, beam quality assessment, and optical system design.
"""

# %%
# Importing necessary modules
# --------------------------------
# We start by importing all the required modules for image processing
# and visualization. To run this example, ensure you have all the required
# dependencies installed.

import numpy as np

import sigima.io
import sigima.objects
import sigima.proc.image
import sigima.proc.signal
from sigima.params import FWHMParam
from sigima.proc.image.exposure import ClipParam, HistogramParam, clip, histogram
from sigima.proc.image.extraction import LineProfileParam, RadialProfileParam
from sigima.proc.image.measurement import centroid
from sigima.tests.helpers import get_test_fnames
from sigima.tests.vistools import view_curves, view_images_side_by_side

# %%
# Load all laser beam images
# ------------------------------------------------
# We load a series of laser beam images taken at different positions along
# the propagation axis (z-axis). The images are contained in the folder laser_beam and
# named following the pattern TEM00_z_*.jpg, where * is the z position in arbitrary
# units.
print("=" * 60)
print("LASER BEAM SIZE MEASUREMENT TUTORIAL")
print("=" * 60)


def load_laser_beam_images():
    """Load all laser beam test images from the test data directory.

    Returns:
        List of image objects loaded from TEM00_z_*.jpg files
    """
    # Get all TEM00 laser beam image files
    image_files = get_test_fnames("laser_beam/TEM00_z_*.jpg")

    # Sort files by z-position (extract number from filename)
    image_files.sort(key=lambda f: int(f.split("_z_")[1].split(".")[0]))

    # Load images
    images = []
    for filepath in image_files:
        img = sigima.io.read_image(filepath)
        # Extract z position from filename for proper naming
        z_pos = filepath.split("_z_")[1].split(".")[0]
        img.title = f"TEM00_z_{z_pos}"
        images.append(img)

    return images


images = load_laser_beam_images()

print(f"✓ Loaded {len(images)} laser beam images")
print("Image details:")
for i, img in enumerate(images):
    intensity_range = f"{img.data.min()}-{img.data.max()}"
    print(f"  {i + 1}. {img.title}: {img.data.shape}, range {intensity_range}")


# %%
# Visualize the first few images
print("\n✓ Visualizing sample images...")
view_images_side_by_side(
    images[:3],
    titles=["Laser beam image 1", "Laser beam image 2", "Laser beam image 3"],
)

# %%
# Background noise analysis with histogram
# ------------------------------------------------
# To analyze the background noise characteristics of the laser beam images,
# we create a histogram of pixel values from the first image. This helps us
# identify the noise floor and determine an appropriate clipping threshold.


print("\n--- Background Noise Analysis ---")

hist_param = HistogramParam()
hist_param.bins = 100
hist_param.range = (0, images[0].data.max())

hist = histogram(images[0], hist_param)
hist.title = "Pixel value histogram of image 1"

print(f"✓ Generated histogram with {hist_param.bins} bins")
print(f"Histogram range: {hist_param.range[0]} - {hist_param.range[1]}")
print("The histogram shows background noise distribution")

# Visualize histogram
view_curves([hist], title="Pixel Value Histogram - Background Analysis")

# %%
# Based on the histogram analysis, we determine a clipping threshold around 30-35 LSB to
# effectively remove background noise from all images.

# %%
# Background noise removal via clipping
# ------------------------------------------------
# We set the threshold to 35 and we apply this clipping to each image in the dataset.

background_threshold = 35

print(f"Will use clipping threshold of {background_threshold} LSB")
# %%
# In order to perform the clipping, we create a ClipParam object
# and set the minimum value to the background threshold. We then apply
# the clipping to each image in the dataset.

print("\n--- Applying Background Clipping ---")
clip_param = ClipParam()
clip_param.lower = background_threshold  # Remove background noise below 35 LSB

clipped_images = []
for img in images:
    clipped_img = clip(img, clip_param)
    clipped_img.title = f"{img.title}_clipped"
    clipped_images.append(clipped_img)

print(f"✓ Applied clipping of {clip_param.lower} LSB to all {len(images)} images")
print("Background noise below threshold has been removed")

# %%
# We can now visualize some clipped images:
view_images_side_by_side(
    images[:3] + clipped_images[:3],
    titles=["Laser beam image 1", "Laser beam image 2", "Laser beam image 3"]
    + [
        "Background-Clipped Image 1",
        "Background-Clipped Image 2",
        "Background-Clipped Image 3",
    ],
    rows=2,
)

# %%
# Compute centroids for beam center detection
# ------------------------------------------------
# Next, we compute the centroid of each clipped image to determine the beam center. This
# is important for accurate profile extraction and FWHM measurements.

print("\n--- Computing Beam Centroids ---")

centroids = []
for img in clipped_images:
    centroid_result = centroid(img)
    if centroid_result is not None and len(centroid_result.coords) > 0:
        # Extract centroid coordinates (x, y)
        coords = centroid_result.coords[0]
        centroids.append((coords[0], coords[1]))  # (x, y)
        print(f"  ✓ {img.title}: centroid at ({coords[0]:.1f}, {coords[1]:.1f})")
    else:
        centroids.append(None)
        print(f"  ✗ {img.title}: no centroid detected")

successful_centroids = [c for c in centroids if c is not None]
print(f"\n✓ Successfully detected {len(successful_centroids)}/{len(images)} centroids")

# %%
# Extract line profiles through beam centers
# ------------------------------------------------
# We extract horizontal line profiles through the detected centroids of each clipped
# image. This provides insight into the beam intensity distribution along a horizontal
# cross-section.

print("\n--- Extracting Line Profiles ---")

line_profiles = []
for i, (img, centroid_coords) in enumerate(zip(clipped_images, centroids)):
    if centroid_coords is not None:
        # Create line profile parameters for horizontal line through centroid
        line_param = LineProfileParam()
        line_param.direction = "horizontal"
        line_param.row = int(centroid_coords[1])  # Use centroid y-coordinate as row

        # Extract line profile
        profile = sigima.proc.image.line_profile(img, line_param)
        profile.title = f"Line_profile_{img.title}"
        line_profiles.append(profile)
        print(f"  ✓ Extracted line profile for {img.title} at row {line_param.row}")

    else:
        line_profiles.append(None)
        print(f"  ✗ Skipped {img.title}: no centroid available")

successful_profiles = [p for p in line_profiles if p is not None]
print(f"\n✓ Generated {len(successful_profiles)} line profiles")

# Visualize some line profiles
view_curves(
    successful_profiles[:3],
    title="Horizontal Line Profiles (First 3 Images)",
    xlabel="Position (pixels)",
    ylabel="Intensity",
)

# %%
# Extract radial profiles around beam centers
# ------------------------------------------------
# We extract radial profiles centered on the detected centroids of each clipped image.
# This provides a circular intensity distribution useful for FWHM measurements.

print("\n--- Extracting Radial Profiles ---")

radial_profiles = []
for img in clipped_images:
    # Create radial profile parameters using automatic centroid detection
    radial_param = RadialProfileParam()
    radial_param.center = "centroid"  # Use automatic centroid detection

    # Extract radial profile
    try:
        profile = sigima.proc.image.radial_profile(img, radial_param)
        profile.title = f"Radial_profile_{img.title}"
        radial_profiles.append(profile)
        print(f"  ✓ Extracted radial profile for {img.title}")
    except Exception as e:
        radial_profiles.append(None)
        print(f"  ✗ Failed to extract radial profile for {img.title}: {e}")

successful_radial = [p for p in radial_profiles if p is not None]
print(f"\n✓ Generated {len(successful_radial)} radial profiles")

# %%
# We can now visualize some radial profiles
view_curves(
    successful_radial[:3],
    title="Radial Profiles (First 3 Images)",
    xlabel="Radial Distance (pixels)",
    ylabel="Intensity",
)

# %%
# Compute FWHM for radial profiles
# ------------------------------------------------
# We compute the Full Width at Half Maximum (FWHM) for each radial profile to
# quantify the beam size. The FWHM is a standard metric for beam width.

print("\n--- Computing FWHM Measurements ---")

fwhm_values = []
fwhm_param = FWHMParam()
fwhm_param.method = "zero-crossing"  # Standard FWHM method

for profile in radial_profiles:
    if profile is not None:
        try:
            fwhm_result = sigima.proc.signal.fwhm(profile, fwhm_param)
            if fwhm_result is not None and len(fwhm_result.coords) > 0:
                # Extract FWHM value (length of the segment)
                fwhm_length = fwhm_result.coords[0][2]  # Third coordinate is length
                fwhm_values.append(fwhm_length)
                print(f"  ✓ {profile.title}: FWHM = {fwhm_length:.2f} pixels")
            else:
                fwhm_values.append(np.nan)
                print(f"  ✗ {profile.title}: FWHM could not be computed")
        except Exception as e:
            fwhm_values.append(np.nan)
            print(f"  ✗ {profile.title}: FWHM computation failed - {e}")
    else:
        fwhm_values.append(np.nan)
# %%
# That's done, we can now print some FWHM statistics to check our results:
valid_fwhm = [f for f in fwhm_values if not np.isnan(f)]
if len(valid_fwhm) > 0:
    mean_fwhm = np.mean(valid_fwhm)
    std_fwhm = np.std(valid_fwhm)
    print("\n✓ FWHM Statistics:")
    print(f"  Valid measurements: {len(valid_fwhm)}/{len(fwhm_values)}")
    min_fwhm = min(valid_fwhm)
    max_fwhm = max(valid_fwhm)
    print(f"  Beam size range: {min_fwhm:.2f} - {max_fwhm:.2f} pixels")
    print(f"  Average beam size: {mean_fwhm:.2f} ± {std_fwhm:.2f} pixels")
else:
    print("\n✗ No valid FWHM measurements obtained")
# %%
# Everything seems fine, we can now analyze the beam size evolution along the z-axis.

# %%
# Analyze beam size evolution
# ------------------------------------------------
# Having computed the FWHM for each radial profile, we can now analyze how the
# beam size evolves along the propagation axis (z-axis). This is useful to extract some
# meaningful information from all the numbers we have obtained. We create a signal
# representing the beam size evolution and visualize it.

print("\n--- Beam Size Evolution Analysis ---")

# Create a signal showing beam size evolution along z-axis
z_positions = list(range(len(fwhm_values)))
beam_evolution = sigima.objects.create_signal(
    "Beam size evolution",
    np.array(z_positions),
    np.array(fwhm_values),
    units=("image_index", "pixels"),
)

print(f"✓ Created beam evolution signal with {len(z_positions)} data points")

# Visualize beam size evolution
view_curves([beam_evolution], title="Beam Size Evolution vs Z-Position (uncalibrated)")

# %%
# The beam evolution signal currently uses image index as the x-axis, and even if we can
# see the trend, it is not very informative. It would be way better to have the x-axis
# in mm.

# %%
# Apply z-axis calibration
# ------------------------------------------------
# We apply a linear calibration to the x-axis of the beam evolution signal to convert
# image index to physical distance in mm. In Sigima there is the possibility to perform
# a linear axis calibration. We use the formula x' = 5*x + 15, where x is
# the image index and x' is the calibrated distance in mm.

print("\n--- Applying Z-Axis Calibration ---")

# Calibrate x-axis using the formula: x' = 5*x + 15 (convert to mm)
calib_param = sigima.proc.signal.XYCalibrateParam()
calib_param.axis = "x"
calib_param.a = 5.0  # Scale factor
calib_param.b = 15.0  # Offset

beam_evolution_calibrated = sigima.proc.signal.calibration(beam_evolution, calib_param)
beam_evolution_calibrated.title = f"{beam_evolution.title} (z calibrated)"

print(f"✓ Applied calibration: z' = {calib_param.a}*z + {calib_param.b}")
print("Z-axis now represents physical distance in mm")

# Visualize calibrated beam evolution
view_curves(
    [beam_evolution_calibrated],
    title="Beam Size Evolution vs Z-Position (calibrated)",
    xlabel="Z Position (mm)",
    ylabel="Beam Size (pixels)",
)

# %%
# Comprehensive analysis summary
# ------------------------------------------------
# We summarize the key results and findings from the laser beam analysis workflow.

print("\n" + "=" * 60)
print("LASER BEAM ANALYSIS COMPLETE")
print("=" * 60)

print(f"✓ Processed {len(images)} laser beam images")
print(f"✓ Applied background clipping at {background_threshold} LSB threshold")
print(f"✓ Detected centroids in {len(successful_centroids)} images")
print(f"✓ Generated {len(successful_profiles)} horizontal line profiles")
print(f"✓ Generated {len(successful_radial)} radial profiles")
print(f"✓ Computed {len(valid_fwhm)} valid FWHM measurements")
print("✓ Created beam size evolution analysis with calibration")

if len(valid_fwhm) > 0:
    print("\nBeam Characterization Results:")
    min_beam = min(valid_fwhm)
    max_beam = max(valid_fwhm)
    mean_beam = np.mean(valid_fwhm)
    std_beam = np.std(valid_fwhm)
    cv_beam = 100 * std_beam / mean_beam
    print(f"• Beam size range: {min_beam:.2f} - {max_beam:.2f} pixels")
    print(f"• Average beam size: {mean_beam:.2f} ± {std_beam:.2f} pixels")
    print(f"• Coefficient of variation: {cv_beam:.1f}%")
