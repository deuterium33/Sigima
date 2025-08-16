.. _api:

API
===

The public Application Programming Interface (API) of Sigima offers a set of functions that can be used to access the DataLab computational backend. This API is designed to be simple and effective, allowing users to perform signal and image processing tasks with ease.

.. list-table::
    :header-rows: 1
    :align: left

    * - Submodule
      - Purpose

    * - :mod:`sigima.tools`
      - Algorithms for data analysis (operating on NumPy arrays) which purpose is to fill in the gaps of common scientific libraries (NumPy, SciPy, scikit-image, etc.), offering consistent tools for computation functions (see :mod:`sigima.proc`)

    * - :mod:`sigima.params`
      - Sets of parameters for configuring computation functions (these parameters are instances of :class:`guidata.dataset.DataSet` objects)

    * - :mod:`sigima.objects`
      - Object model for signals and images (:class:`sigima.objects.SignalObj` and :class:`sigima.objects.ImageObj`), scalar results (:class:`sigima.objects.GeometryResult` and :class:`sigima.objects.TableResult`), and related functions

    * - :mod:`sigima.proc`
      - Computation functions, which operate on signal and image objects (:class:`sigima.objects.SignalObj` or :class:`sigima.objects.ImageObj`) and return signal or image objects, or scalar results (:class:`sigima.objects.GeometryResult` or :class:`sigima.objects.TableResult`).


.. toctree::
   :maxdepth: 2
   :caption: Public features:

   tools
   params
   objects
   proc
