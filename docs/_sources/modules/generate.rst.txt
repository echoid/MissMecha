Generate Module
=======================

This module defines the interface and mechanism functions for simulating missing values.

The main class, ``MissMechaGenerator``, serves as a flexible controller that supports different missing data mechanisms.  
Each mechanism (MCAR, MAR, MNAR) corresponds to a specific pattern of missingness and can be applied independently or in combination.

.. currentmodule:: missmecha.generator



``MissMechaGenerator``
-----------------------

The main interface to generate missingness patterns.

.. autoclass:: MissMechaGenerator
   :members:
   :undoc-members:
   :show-inheritance:


Mechanism Functions
-----------------------

This section provides details on each missing data mechanism supported by ``MissMechaGenerator``.  
You can explore MCAR, MAR, and MNAR function implementations individually below.


.. toctree::
   :maxdepth: 1
   :titlesonly:

   generate_fun/mcar
   generate_fun/mar
   generate_fun/mnar
   generate_fun/mnar


Custom Mechanisms
-----------------------

In addition to the predefined mechanisms (MCAR, MAR, MNAR), 
``MissMechaGenerator`` also supports **custom missingness mechanisms** 
defined entirely by the user.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   generate_fun/custom_mechanisms


This is useful when you want to simulate structured patterns 
(e.g., top-k thresholds, model-based missingness, domain-specific logic).

Usage Options
^^^^^^^^^^^^^^^^^^

You can inject a custom mechanism in two ways:

- **Global mode**: using ``mechanism="custom"`` and passing your class via ``custom_class=...``.
- **Column-wise mode**: via the ``info`` dictionary, per-column control is supported.

Your custom class must implement:

.. code-block:: python

   class MyMasker:
       def fit(self, X, y=None): ...
       def transform(self, X): ...

**Global example:**

.. code-block:: python

   gen = MissMechaGenerator(
       mechanism="custom",
       custom_class=MyMasker,
       missing_rate=0.3
   )

**Column-wise example:**

.. code-block:: python

   info = {
       "col1": {
           "mechanism": "custom",
           "custom_class": MyMasker,
           "rate": 0.2
       }
   }
   gen = MissMechaGenerator(info=info)

.. note::

   Your custom mechanism must implement both ``fit(X, y=None)`` and ``transform(X)``.
   You can also pass additional parameters through ``para`` if needed.

See also: :doc:`generate_fun/custom_mechanisms`
