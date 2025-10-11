# Release notes #

The `sigima` library is part of the DataLab open-source platform.
See DataLab [roadmap page](https://datalab-platform.com/en/contributing/roadmap.html) for future and past milestones.

## sigima 1.0.0 ##

💥 New features and enhancements:

* **DateTime support for signal data**: Added comprehensive datetime handling for signal X-axis data
  * Automatic detection and conversion of datetime columns when reading CSV files
    * Detects datetime values in the first or second column (handling index columns)
    * Validates datetime format and ensures reasonable date ranges (post-1900)
    * Converts datetime strings to float timestamps for efficient computation
    * Preserves datetime metadata for proper display and export
  * New `SignalObj` methods for datetime manipulation:
    * `set_x_from_datetime()`: Convert datetime objects/strings to signal X data with configurable time units (s, ms, μs, ns, min, h)
    * `get_x_as_datetime()`: Retrieve X values as datetime objects for display or export
    * `is_x_datetime()`: Check if signal contains datetime data
  * Enhanced CSV export to preserve datetime format when writing signals with datetime X-axis
  * New constants module (`sigima.objects.signal.constants`) defining datetime metadata keys and time unit conversion factors
  * Comprehensive unit tests covering datetime conversion, I/O roundtrip, and edge cases
  * Example test data file with real-world temperature/humidity logger data (`datetime.txt`)

* **New client subpackage**: Migrated DataLab client functionality to `sigima.client`
  * Added `sigima.client.remote.SimpleRemoteProxy` for XML-RPC communication with DataLab
  * Added `sigima.client.base.SimpleBaseProxy` as abstract base class for DataLab proxies
  * Included comprehensive unit tests and API documentation
  * Maintains headless design principle (GUI components excluded)
  * Enables remote control of DataLab application from Python scripts and Jupyter notebooks
  * Client functionality is now directly accessible: `from sigima import SimpleRemoteProxy`

* **New image ROI feature**: Added inverse ROI functionality for image ROIs
  * Added `inside` parameter to `BaseSingleImageROI` base class, inherited by all image ROI types (`PolygonalROI`, `RectangularROI`, `CircularROI`)
  * When `inside=True`, ROI represents the region inside the shape (inverted behavior)
  * When `inside=False` (default), ROI represents the region outside the shape (original behavior)
  * Fully integrated with serialization (`to_dict`/`from_dict`) and parameter conversion (`to_param`/`from_param`)
  * Signal ROIs (`SegmentROI`) are unaffected as the concept doesn't apply to 1D intervals
  * Optimal architecture with zero code duplication - all `inside` functionality implemented once in the base class
  * Individual ROI classes no longer need custom constructors, inheriting directly from base class

* New image operation:
  * Convolution.

* New image format support:
  * **Coordinated text image files**: Added support for reading coordinated text files (`.txt` extension), similar to the Matris image format.
    * Supports both real and complex-valued image data with optional error images.
    * Automatically handles NaN values in the data.
    * Reads metadata including units (X, Y, Z) and labels from file headers.

* New image analysis features:
  * Horizontal and vertical projections
    * Compute the horizontal projection profile by summing values along the y-axis (`sigima.proc.image.measurement.horizontal_projection`).
    * Compute the vertical projection profile by summing values along the x-axis (`sigima.proc.image.measurement.vertical_projection`).

* **New curve fitting algorithms**: Complete curve fitting framework with `sigima.tools.signal.fitting` module:
  * **Core fitting functions**: Comprehensive set of curve fitting algorithms for scientific data analysis:
    * `linear_fit`: Linear regression fitting
    * `polynomial_fit`: Polynomial fitting with configurable degree
    * `gaussian_fit`: Gaussian profile fitting for peak analysis
    * `lorentzian_fit`: Lorentzian profile fitting for spectroscopy
    * `voigt_fit`: Voigt profile fitting (convolution of Gaussian and Lorentzian profiles)
    * `exponential_fit`: Single exponential fitting with overflow protection
    * `doubleexponential_fit`: Double exponential fitting with advanced parameter estimation
    * `planckian_fit`: Planckian (blackbody radiation) fitting with correct physics implementation
    * `twohalfgaussian_fit`: Asymmetric peak fitting with separate left/right parameters
    * `multilorentzian_fit`: Multi-peak Lorentzian fitting for complex spectra
    * `sinusoidal_fit`: Sinusoidal fitting with FFT-based frequency estimation
    * `cdf_fit`: Cumulative Distribution Function fitting using error function
    * `sigmoid_fit`: Sigmoid (logistic) function fitting for S-shaped curves
  * **Advanced double exponential fitting**: Enhanced algorithm with:
    * Standard double exponential model: `y = a_left*exp(b_left*x) + a_right*exp(b_right*x) + y0`
    * Multi-start optimization strategy for robust convergence to global minimum
    * Support for both positive and negative exponential rates (growth and decay components)
    * Comprehensive parameter bounds validation to prevent optimization errors
  * **Enhanced asymmetric peak fitting**: Advanced `twohalfgaussian_fit` with:
    * Separate baseline offsets for left and right sides (`y0_left`, `y0_right`)
    * Independent amplitude parameters (`amp_left`, `amp_right`) for better asymmetric modeling
    * Robust baseline estimation using percentile-based methods
  * **Technical features**: All fitting functions include:
    * Automatic initial parameter estimation from data characteristics
    * Proper bounds enforcement ensuring optimization stability
    * Comprehensive error handling and parameter validation
    * Consistent dataclass-based parameter structures
    * Full test coverage with synthetic and experimental data validation

* New common signal/image feature:
  * Added `phase` (argument) feature to extract the phase information from complex signals or images.
  * Added operation to create complex-valued signal/image from real and imaginary parts.
  * Added operation to create complex-valued signal/image from magnitude and phase.
  * Standard deviation of the selected signals or images (this complements the "Average" feature).
  * Generate new signal or image: Poisson noise.
  * Add noise to the selected signals or images.
    * Gaussian, Poisson or uniform noise can be added.
  * New utility functions to generate file basenames.
  * Deconvolution in the frequency domain.

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
  * New "2D resampling" feature:
    * This feature allows to resample 2D images to a new coordinate grid using interpolation.
    * It supports two resampling modes: pixel size and output shape.
    * Multiple interpolation methods are available: linear, cubic, and nearest neighbor.
    * The `fill_value` parameter controls how out-of-bounds pixels are handled, with support for numeric values or NaN.
    * Automatic data type conversion ensures proper NaN handling for integer images.
    * It is implemented in the `sigima.proc.image.resampling` function with parameters defined in `Resampling2DParam`.
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
  * **Pulse analysis algorithms**: Comprehensive pulse feature extraction framework in `sigima.tools.signal.pulse` module:
    * **Core pulse analysis functions**: Complete set of algorithms for step and square pulse characterization:
      * `extract_pulse_features`: Main function for automated pulse feature extraction
      * `heuristically_recognize_shape`: Intelligent signal type detection (step, square, or other)
      * `detect_polarity`: Robust polarity detection using baseline analysis
    * **Advanced timing parameter extraction**: Precise measurement algorithms for:
      * Rise and fall time calculations with configurable start/stop ratios (e.g., 10%-90%)
      * Timing parameters at specific fractions (x10, x50, x90, x100) of signal amplitude
      * Full width at half maximum (FWHM) computation for square pulses
      * Foot duration measurement for pulse characterization
    * **Baseline analysis capabilities**: Statistical methods for:
      * Automatic baseline range detection from signal extremes
      * Robust baseline level estimation using mean values within ranges
      * Start and end baseline characterization for differential analysis
    * **Signal validation and error handling**: Comprehensive input validation with:
      * Data array consistency checks and NaN/infinity detection
      * Signal length validation and range boundary verification
      * Graceful error handling with descriptive exception messages
    * **PulseFeatures dataclass**: Structured result container with all extracted parameters:
      * Amplitude, polarity, and offset measurements
      * Timing parameters (rise_time, fall_time, fwhm, x10, x50, x90, x100)
      * Baseline ranges (xstartmin, xstartmax, xendmin, xendmax)
      * Signal shape classification and foot duration
    * Implementation leverages robust statistical methods and provides both high-level convenience functions and low-level building blocks for custom pulse analysis workflows.
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

* New image "Extent" computed parameters:
  * Added computed parameters for image extent: `xmin`, `xmax`, `ymin`, and `ymax`.
  * These parameters are automatically calculated based on the image origin, pixel spacing, and dimensions.
  * They provide the physical coordinate boundaries of the image for enhanced spatial analysis.

* New I/O features:
  * Added HDF5 format for signal and image objects (extensions `.h5sig` and `.h5ima`) that may be opened with any HDF5 viewer.
  * Added support for FT-Lab signal and image format.
  * Added functions to read and write metadata and ROIs in JSON format:
    * `sigima.io.read_metadata` and `sigima.io.write_metadata` for metadata.
    * `sigima.io.read_roi` and `sigima.io.write_roi` for ROIs.
  * Added convenience I/O functions `write_signals` and `write_images` with `SaveToDirectoryParam` support:
    * These functions enable batch saving of multiple signal or image objects to a directory with configurable naming patterns.
    * `SaveToDirectoryParam` provides control over file basenames (with Python format string support), extensions, directory paths, and overwrite behavior.
    * Automatic filename conflict resolution ensures unique filenames when duplicates would occur.
    * Enhanced workflow efficiency for processing and saving multiple objects in batch operations.

✨ Core architecture update: scalar result types

* Introduced two new immutable result types: `TableResult` and `GeometryResult`, replacing the legacy `ResultProperties` and `ResultShape` objects.
  * These new result types are computation-oriented and free of application-specific logic (e.g., Qt, metadata), enabling better separation of concerns and future reuse.
  * Added a `TableResultBuilder` utility to incrementally define tabular computations (e.g., statistics on signals or images) and generate a `TableResult` object.
  * All metadata-related behaviors of former result types have been migrated to the DataLab application layer.
  * Removed obsolete or tightly coupled features such as `from_metadata_entry()` and `transform_shapes()` from the Sigima core.
* This refactoring greatly improves modularity, testability, and the clarity of the scalar computation API.

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
