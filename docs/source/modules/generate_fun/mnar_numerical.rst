Numerical MNAR Types
=====================

This section introduces a suite of Missing Not At Random (MNAR) mechanisms specifically designed for **numerical** data.

Unlike MCAR and MAR, MNAR assumes that the probability of missingness depends on the **unobserved** values themselves,  
making it the most complex and realistic class of missingness mechanisms to simulate.

All mechanisms below support numerical arrays (NumPy or pandas), and many allow column-wise or full-matrix control.

.. note::

   Some types (e.g., ``MNARType5`` and ``MNARType6``) are designed to operate **on each column independently** (single-column masking).  
   Others (e.g., ``MNARType1`` to ``MNARType4``) apply **global strategies** across multiple columns.

----

Overview of MNAR Mechanisms
----------------------------

.. list-table:: Summary of MNAR Types
   :widths: 15 20 60
   :header-rows: 1

   * - Type
     - Scope
     - Description
   * - ``MNARType1``
     - Global
     - Quantile-based masking using thresholds on both masked and observed columns.
   * - ``MNARType2``
     - Global
     - Logistic missingness using observed features to determine probabilities.
   * - ``MNARType3``
     - Global
     - Self-masking with logistic sampling, where each feature masks itself.
   * - ``MNARType4``
     - Global
     - Applies missingness above/below specific quantiles, with support for "upper", "lower", or "both" cuts.
   * - ``MNARType5``
     - Single-column
     - Applies self-masking to each feature independently using fitted logistic intercepts.
   * - ``MNARType6``
     - Single-column
     - Column-wise masking below percentile thresholds; supports both NumPy and pandas formats.


.. currentmodule:: missmecha.generate.mnar

``MNARType1``
-----------------------------

.. autoclass:: MNARType1
   :members:
   :undoc-members:
   :show-inheritance:

----

``MNARType2``
-----------------------------

.. autoclass:: MNARType2
   :members:
   :undoc-members:
   :show-inheritance:

----

``MNARType3``
-----------------------------

.. autoclass:: MNARType3
   :members:
   :undoc-members:
   :show-inheritance:

----

``MNARType4``
-----------------------------

.. autoclass:: MNARType4
   :members:
   :undoc-members:
   :show-inheritance:

----

``MNARType5``
-----------------------------

.. autoclass:: MNARType5
   :members:
   :undoc-members:
   :show-inheritance:

----

``MNARType6``
-----------------------------

.. autoclass:: MNARType6
   :members:
   :undoc-members:
   :show-inheritance:


