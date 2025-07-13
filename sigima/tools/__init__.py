"""
Tools (:mod:`sigima.tools`)
---------------------------

This package contains functions operating on NumPy arrays that are intended to be
used in Sigima computation functions. These functions are complementary to the
algorithms provided by external libraries such as SciPy, NumPy, and scikit-image.

Even though these functions are primarily designed to be used in the Sigima pipeline,
they can also be used independently. They provide a wide range of features but are
not exhaustive due to the vast number of algorithms already available in the
scientific Python ecosystem.

.. seealso::

    The :mod:`sigima.proc` module contains the Sigima computation functions that
    operate on signal and image objects (i.e. :class:`sigima.objects.SignalObj` and
    :class:`sigima.objects.ImageObj`, defined in the :mod:`sigima.objects` package).

These tools are organized in subpackages according to their purpose. The following
subpackages are available:

- :mod:`sigima.tools.checks`: Input data checks for all tools
- :mod:`sigima.tools.signal`: Signal processing tools
- :mod:`sigima.tools.image`: Image processing tools
- :mod:`sigima.tools.datatypes`: Data type conversion tools
- :mod:`sigima.tools.coordinates`: Coordinate conversion tools

Check functions
^^^^^^^^^^^^^^^

.. automodule:: sigima.tools.checks
   :members:

Signal Processing Tools
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sigima.tools.signal
   :members:

Image Processing Tools
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sigima.tools.image
   :members:

Data Type Conversion Tools
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sigima.tools.datatypes
   :members:

Coordinate Conversion Tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sigima.tools.coordinates
   :members:

"""
