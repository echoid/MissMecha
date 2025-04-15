Impute Module
=============

This module provides baseline imputers to support quick experimentation or evaluation of imputation strategies.  
The main tool is ``SimpleSmartImputer``, which detects column types and fills missing values accordingly.

.. currentmodule:: missmecha.impute

Function Overview
-----------------
.. autosummary::
   :nosignatures:

   SimpleSmartImputer

Module Reference
===============

``SimpleSmartImputer``
----------------------

This simple yet adaptive imputer chooses different strategies based on column types:
- Numerical columns are imputed using the **mean**
- Categorical columns are imputed using the **mode**

It supports optional control over column type detection and verbosity.

.. autoclass:: SimpleSmartImputer
   :members:
   :undoc-members:
   :show-inheritance:


   