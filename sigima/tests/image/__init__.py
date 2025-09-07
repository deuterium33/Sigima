# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for image features
-----------------------------

[1] Implementation note regarding scikit-image methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following note applies to:
- thresholding methods (isodata, li, mean, minimum, otsu, triangle, yen)
- exposure methods (adjust_gamma, adjust_log, adjust_sigmoid, rescale_intensity,
  equalize_hist, equalize_adapthist)
- restoration methods (denoise_tv, denoise_bilateral, denoise_wavelet)
- morphology methods (white_tophat, black_tophat, erosion, dilation, opening, closing)
- edge detection methods (canny, roberts, prewitt, sobel, scharr, farid, laplace)

The thresholding, morphological, and edge detection methods are implemented
in the scikit-image library: those algorithms are considered to be validated,
so we can use them as reference.
As a consequence, the only purpose of the associated validation tests is to check
if the methods are correctly called and if the results are consistent with
the reference implementation.

In other words, we are not testing the correctness of the algorithms, but
the correctness of the interface between the Sigima and the scikit-image
libraries.
"""
