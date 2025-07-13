"""
Computation (:mod:`sigima.proc`)
--------------------------------

This package contains the Sigima computation functions that implement processing
features for signal and image objects. These functions are designed to operate directly
on :class:`sigima.objects.SignalObj` and :class:`sigima.objects.ImageObj` objects,
which are defined in the :mod:`sigima.objects` package.

Even though these functions are primarily designed to be used in the Sigima pipeline,
they can also be used independently.

.. seealso::

    See the :mod:`sigima.tools` package for some algorithms that operate directly on
    NumPy arrays.

Each computation module defines a set of computation objects, that is, functions
that implement processing features and classes that implement the corresponding
parameters (in the form of :py:class:`guidata.dataset.datatypes.Dataset` subclasses).
The computation functions takes a signal or image object
(e.g. :py:class:`sigima.objects.SignalObj`)
and a parameter object (e.g. :py:class:`sigima.params.MovingAverageParam`) as input
and return a signal or image object as output (the result of the computation).
The parameter object is used to configure the computation function
(e.g. the size of the moving average window).

In Sigima overall architecture, the purpose of this package is to provide the
computation functions that are used by the :mod:`sigima.core.gui.processor` module,
based on the algorithms defined in the :mod:`sigima.tools` module and on the
data model defined in the :mod:`sigima.objects` (or :mod:`sigima.core.model`) module.

The computation modules are organized in subpackages according to their purpose.
The following subpackages are available:

- :mod:`sigima.proc.base`: Common processing features
- :mod:`sigima.proc.signal`: Signal processing features
- :mod:`sigima.proc.image`: Image processing features

Common processing features
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sigima.proc.base
   :members:

Signal processing features
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sigima.proc.signal
   :members:

Image processing features
^^^^^^^^^^^^^^^^^^^^^^^^^

Basic image processing
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sigima.proc.image
   :members:

Thresholding
~~~~~~~~~~~~

.. automodule:: sigima.proc.image.threshold
    :members:

Exposure correction
~~~~~~~~~~~~~~~~~~~

.. automodule:: sigima.proc.image.exposure
    :members:

Restoration
~~~~~~~~~~~

.. automodule:: sigima.proc.image.restoration
    :members:

Morphology
~~~~~~~~~~

.. automodule:: sigima.proc.image.morphology
    :members:

Edge detection
~~~~~~~~~~~~~~

.. automodule:: sigima.proc.image.edges

Detection
~~~~~~~~~

.. automodule:: sigima.proc.image.detection
    :members:

Utilities
^^^^^^^^^

This package also provides some utility functions to help with the creation and
introspection of computation functions:

.. autofunction:: sigima.proc.computation_function
.. autofunction:: sigima.proc.is_computation_function
.. autofunction:: sigima.proc.get_computation_metadata
.. autofunction:: sigima.proc.find_computation_functions
"""
