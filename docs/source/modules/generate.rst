Generate Module
===============

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


