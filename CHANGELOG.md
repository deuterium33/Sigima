# Release notes #

The `sigima` library is part of the DataLab open-source platform.
See DataLab [roadmap page](https://datalab-platform.com/en/contributing/roadmap.html) for future and past milestones.

## sigima 0.3.0 ##

✨ Core architecture update: scalar result types

* Introduced two new immutable result types: `TableResult` and `GeometryResult`, replacing the legacy `ResultProperties` and `ResultShape` objects.
  * These new result types are computation-oriented and free of application-specific logic (e.g., Qt, metadata), enabling better separation of concerns and future reuse.
  * Added a `TableResultBuilder` utility to incrementally define tabular computations (e.g., statistics on signals or images) and generate a `TableResult` object.
  * All metadata-related behaviors of former result types have been migrated to the DataLab application layer.
  * Removed obsolete or tightly coupled features such as `from_metadata_entry()` and `transform_shapes()` from the Sigima core.
* This refactoring greatly improves modularity, testability, and the clarity of the scalar computation API.

💥 New features and enhancements:

* New common signal/image feature:
  * Added `phase` (argument) feature to extract the phase information from complex signals or images.
  * Added operation to create complex-valued signal/image from real and imaginary parts.
  * Added operation to create complex-valued signal/image from magnitude and phase.
  * Standard deviation of the selected signals or images (this complements the "Average" feature).
  * Generate new signal or image: Poisson noise.
  * Add noise to the selected signals or images.
    * Gaussian, Poisson or uniform noise can be added.

* New ROI features:
  * Improved single ROI title handling, using default title based on the index of the ROI when no title is provided.
  * Added `combine_with` method to ROI objects (`SignalROI` and `ImageROI`) to return a new ROI that combines the current ROI with another one (union) and handling duplicate ROIs.
  * Image ROI transformations:
    * Before this change, image ROI were removed after applying each single computation function.
    * Now, the geometry computation functions preserve the ROI information across transformations: the transformed ROIs are automatically updated in the image object.
  * Image ROI coordinates:
    * Before this change, image ROI coordinates were defined using indices by default.
    * Now, `ROI2DParam` uses physical coordinates by default.
    * Note that ROI may still be defined using indices instead (using `create_image_roi` function).
  * Image ROI grid:
    * New `generate_image_grid_roi` function: create a grid of ROIs from an image, with customizable parameters for grid size, spacing, and naming.
    * This function allows for easy extraction of multiple ROIs from an image in a structured manner.
    * Parameters are handled via the `ROIGridParam` class, which provides a convenient way to specify grid properties:
      * `nx` / `ny`: Number of grid cells in the X/Y direction.
      * `xsize` / `ysize`: Size of each grid cell in pixels.
      * `xtranslation` / `ytranslation`: Translation of the grid in pixels.
      * `xdirection` / `ydirection`: Direction of the grid (increasing/decreasing).

* New image processing features:
  * New "Frequency domain Gaussian filter" feature:
    * This feature allows to filter an image in the frequency domain using a Gaussian filter.
    * It is implemented in the `sigima.proc.image.frequency_domain_gaussian_filter` function.
  * New "Erase" feature:
    * This feature allows to erase an area of the image using the mean value of the image.
    * It is implemented in the `sigima.proc.image.erase` function.
    * The erased area is defined by a region of interest (ROI) parameter set.
    * Example usage:

      ```python
      import numpy as np
      import sigima.objects as sio
      import sigima.proc.image as sipi

      obj = sio.create_image("test_image", data=np.random.rand(1024, 1024))
      p = sio.ROI2DParam.create(x0=600, y0=800, width=300, height=200)
      dst = sipi.erase(obj, p)
      ```

  * By default, pixel binning changes the pixel size.

  * Improved centroid estimation:
    * New `get_centroid_auto` method implements an adaptive strategy that chooses between the Fourier-based centroid and a more robust fallback (scikit-image), based on agreement with a projected profile-based reference.
    * Introduced `get_projected_profile_centroid` function for robust estimation via 1D projections (median or barycentric), offering high accuracy even with truncated or noisy images.
    * These changes improve centroid accuracy and stability in edge cases (e.g. truncated disks or off-center spots), while preserving noise robustness.
    * See [DataLab issue #251](https://github.com/DataLab-Platform/DataLab/issues/251) for more details.

* New signal processing features:
  * New "Brick wall filter" feature:
    * This feature allows to filter a signal in the frequency domain using an ideal ("brick wall") filter.
    * It is implemented in `sigima.proc.signal.frequency_filter`, along the other frequency domain filtering features (`Bessel`, `Butterworth`, etc.).
  * Enhanced zero padding to support prepend and append. Change default strategy to next power of 2.
  * Comprehensive uncertainty propagation implementation:
    * Added mathematically correct uncertainty propagation to ~15 core signal processing functions.
    * Enhanced `Wrap1to1Func` class to handle uncertainty propagation for mathematical functions (`sqrt`, `log10`, `exp`, `clip`, `absolute`, `real`, `imag`).
    * Implemented uncertainty propagation for arithmetic operations (`product_constant`, `division_constant`).
    * Added uncertainty propagation for advanced processing functions (`power`, `normalize`, `derivative`, `integral`, `calibration`).
    * All implementations use proper error propagation formulas with numerical stability handling (NaN/infinity protection).
    * Optimized for memory efficiency by leveraging `dst_1_to_1` automatic uncertainty copying and in-place modifications.
    * Maintains backward compatibility with existing signal processing workflows.

* New 2D ramp image generator:
  * This feature allows to generate a 2D ramp image: z = a(x − x₀) + b(y − y₀) + c
  * It is implemented in the `sigima.objects.Ramp2DParam` parameter class.
  * Example usage:

    ```python
    import sigima.objects as sio
    param = sio.Ramp2DParam.create(width=100, height=100, a=1.0, b=2.0)
    image = sio.create_image_from_param(param)
    ```

* New signal generators: linear chirp, logistic function, Planck function.

* New I/O features:
  * Added HDF5 format for signal and image objects (extensions `.h5sig` and `.h5ima`) that may be opened with any HDF5 viewer.
  * Added support for FT-Lab signal and image format.
  * Added functions to read and write metadata and ROIs in JSON format:
    * `sigima.io.read_metadata` and `sigima.io.write_metadata` for metadata.
    * `sigima.io.read_roi` and `sigima.io.write_roi` for ROIs.

🛠️ Bug fixes:

* Fix how data is managed in signal objects (`SignalObj`):
  * Signal data is stored internally as a 2D array with shape `(2, n)`, where the first row is the x data and the second row is the y data: that is the `xydata` attribute.
  * Because of this, when storing complex Y data, the data type is propagated to the x data, which is not always desired.
  * As a workaround, the `x` property now returns the real part of the x data.
  * Furthermore, the `get_data` method now returns a tuple of numpy arrays instead of a single array, allowing to access both x and y data separately, keeping the original data type.
* Fix ROI conversion between physical and indices coordinates:
  * The conversion between physical coordinates and indices has been corrected (half pixel error was removed).
  * The `indices_to_physical` and `physical_to_indices` methods now raise a `ValueError` if the input does not contain an even number of elements (x, y pairs).

## sigima 0.2.0 ##

⚠️ Major API changes:

* Rename `sigima.computation` to `sigima.proc`.
* Rename `sigima.algorithms` to `sigima.tools`.
* Rename `sigima.obj` to `sigima.objects`.
* Rename `sigima.param` to `sigima.params`.

ℹ️ Various changes:

* Add Sigima SVG logo.

🛠️ Bug fixes:

* Fix API documentation and docstrings.

## sigima 0.1.0 ##

This first version of the library is the result of the externalization of the signal and image processing features from the DataLab main repository.
