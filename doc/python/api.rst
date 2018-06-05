.. _python-api:

The *rtneuron* package
======================

The entry point of the RTNeuron Python API is a package called **rtneuron**.
This package contains a collections of submodules with the wrapped C++ classes
as well as some free functions helpful for routine tasks such as displaying a
target from a blue config file.

Some examples of how to use the package can be found in the :ref:`gallery`.

This page is the reference documentation for all the classes and functions
provided by the rtneuron package. The documentation is divided in two sections,
the first one presents the wrapping of the C++ library and the second one
describes classes and functions that are only available in the Python package.

Wrapped C++ classes
-------------------

In reality, the C++ wrapping is a subpackage called **_rtneuron** which
contains classes and other submodules for the different C++ namespaces. When
**rtneuron** is imported, it brings into its namespace all the contents of
**_rtneuron** and imports all the required submodules. The C++ namespace
layout is respected whenever possible in these submodules, as in
**rtneuron.sceneops** for example.

rtneuron namespace
~~~~~~~~~~~~~~~~~~
.. automodule:: rtneuron._rtneuron
   :members:
   :undoc-members:

rtneuron.net namespace
~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: rtneuron._rtneuron._net
   :members:
   :undoc-members:

rtneuron.sceneops namespace
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: rtneuron._rtneuron._sceneops
   :members:
   :undoc-members:

Free functions
~~~~~~~~~~~~~~
.. automodule:: rtneuron
   :members:

Helper modules
--------------

rtneuron.util
~~~~~~~~~~~~~
.. automodule:: rtneuron.util
   :members:

.. automodule:: rtneuron.util.camera
   :members:

.. automodule:: rtneuron.util.camera.Paths
   :members:

.. automodule:: rtneuron.util.camera.Ortho
   :members:

rtneuron.sceneops
~~~~~~~~~~~~~~~~~

.. automodule:: rtneuron.sceneops
   :members:

.. automodule:: rtneuron.sceneops.SynapticProjections
   :members:

