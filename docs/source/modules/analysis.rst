Analysis Module
=======================

This module provides utilities to analyze the structure and quality of missing data and its imputations.

.. currentmodule:: missmecha.analysis

Function Overview
----------------------------------
.. autosummary::
   :nosignatures:

   compute_missing_rate
   evaluate_imputation
   MCARTest




Module Reference
=======================

.. Missing Rate Summary
.. --------------------

``compute_missing_rate``
----------------------------------------

Summarize the extent and structure of missing data in a DataFrame or NumPy array.

.. autofunction:: compute_missing_rate


.. Imputation Evaluation
.. ---------------------

``evaluate_imputation``
----------------------------------------

Evaluate imputation quality by comparing filled values to the ground truth at missing positions.

.. autofunction:: evaluate_imputation


.. MCAR Testing
.. ------------

``MCARTest``
--------------------

This class supports two approaches to test the MCAR assumption:

- **Little's MCAR Test**: a global test for whether the missingness is completely at random.
- **Pairwise T-Tests**: individual tests that compare observed vs. missing groups.

.. autoclass:: MCARTest
   :members:
   :undoc-members:
   :show-inheritance:


