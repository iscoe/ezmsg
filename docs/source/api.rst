API
===

An ``ezmsg`` pipeline is created from a few basic components.
``ezmsg`` provides a framework for you to define your own graphs using its building blocks.
Inherit from its base components to define a pipeline that works for your project.

.. automodule:: ezmsg.core

Most ``ezmsg`` classes intended for use in building pipelines are available in ``ezmsg.core``.
It is convention to ``import ezmsg.core as ez`` and then use this shorthand in your code. e.g.,

.. code-block:: python

   class MyUnit(ez.Unit):
       ...

Components
----------

.. autoclass:: Component

.. autoclass:: Collection
   :show-inheritance:
   :members:

.. autoclass:: NetworkDefinition

.. autoclass:: Unit
   :show-inheritance:
   :members:
   :inherited-members:


Unit Function Decorators
^^^^^^^^^^^^^^^^^^^^^^^^

These function decorators can be added to member functions.

.. automethod:: ezmsg.core.subscriber

.. automethod:: ezmsg.core.publisher

.. automethod:: ezmsg.core.main

.. automethod:: unit.thread

.. automethod:: ezmsg.core.task

.. automethod:: ezmsg.core.process

.. automethod:: ezmsg.core.timeit


Component Interaction
---------------------

.. autoclass:: Settings

.. autoclass:: State


Stream
------

Facilitates a flow of ``Messages`` into or out of a ``Component``.

.. autoclass:: InputStream

.. autoclass:: OutputStream


Custom Exceptions
-----------------

.. autoclass:: Complete

.. autoclass:: NormalTermination


Entry Point
-----------

.. automethod:: ezmsg.core.run
