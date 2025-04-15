Numerical MAR Types
====================

This section introduces functions for simulating **Missing At Random (MAR)** mechanisms in numerical data.

MAR assumes that the probability of missingness depends only on the **observed** variables.  
Formally:

.. math::

   P(M \mid X) = P(M \mid X_{\text{obs}})

This allows for more realistic simulation than MCAR, while remaining statistically manageable for imputation or analysis.

All functions below are designed for **continuous** or **ordinal** numerical data, and cover a wide range of feature dependencies.

.. note::

   All MAR types in MissMecha support tabular numerical data.  
   Some types support full-matrix masking, while others work on a per-column basis.

----

Overview of MAR Mechanisms
----------------------------

.. list-table:: Summary of MAR Types
   :widths: 15 20 60
   :header-rows: 1

   * - Type
     - Dependency
     - Description
   * - ``MARType1``
     - Logistic model
     - Introduces missingness using a fitted logistic regression over observed features.
   * - ``MARType2``
     - Mutual information
     - Selects masking columns based on mutual information with synthetic labels.
   * - ``MARType3``
     - Point-biserial
     - Computes correlations with a (synthetic or real) binary label to guide missingness.
   * - ``MARType4``
     - Correlation ranking
     - Identifies least-correlated columns and masks based on their top correlated partners.
   * - ``MARType5``
     - Ranking
     - Uses value rankings in a controlling column to assign missingness to others.
   * - ``MARType6``
     - Binary grouping
     - Splits rows into high/low groups based on median and applies missingness with skewed probability.
   * - ``MARType7``
     - Top-value rule
     - Selects rows with top values in a controlling column and masks all others.
   * - ``MARType8``
     - Extreme-value
     - Masks rows with both highest and lowest values in a selected column.

.. currentmodule:: missmecha.generate.mar

``MARType1``
------------

.. autoclass:: MARType1
   :members:
   :undoc-members:
   :show-inheritance:

----

``MARType2``
------------

.. autoclass:: MARType2
   :members:
   :undoc-members:
   :show-inheritance:

----

``MARType3``
------------

.. autoclass:: MARType3
   :members:
   :undoc-members:
   :show-inheritance:

----

``MARType4``
------------

.. autoclass:: MARType4
   :members:
   :undoc-members:
   :show-inheritance:

----

``MARType5``
------------

.. autoclass:: MARType5
   :members:
   :undoc-members:
   :show-inheritance:

----

``MARType6``
------------

.. autoclass:: MARType6
   :members:
   :undoc-members:
   :show-inheritance:

----

``MARType7``
------------

.. autoclass:: MARType7
   :members:
   :undoc-members:
   :show-inheritance:

----

``MARType8``
------------

.. autoclass:: MARType8
   :members:
   :undoc-members:
   :show-inheritance:

----


   