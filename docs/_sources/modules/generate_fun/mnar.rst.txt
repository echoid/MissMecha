MNAR Functions
=======================

Missing Not At Random (MNAR) refers to a missing data mechanism where the probability of missingness depends on the **unobserved** data.

Formally, let :math:`X` be the data matrix and :math:`M` the missingness indicator. Then under MNAR:

.. math::

   P(M \mid X) = P(M \mid X_{\text{obs}}, X_{\text{miss}})

This implies that the probability of missingness may depend directly on the values that are missing — making the mechanism inherently more complex and harder to address using standard imputation techniques.

In MissMecha, we provide multiple MNAR variants for both **numerical** and **categorical** data. These functions help simulate challenging scenarios for evaluating robustness of imputation and modeling pipelines.

.. note::

   MNAR mechanisms support all data types, including **numerical**, **categorical**, and **time series** inputs (when treated as tabular data).  
   Some types (e.g., ``MNARType5`` and ``MNARType6``) are designed to operate on **single columns only**.

----

**Numerical MNAR Types**

:doc:`Numerical Types <mnar_numerical>`  
Applies to continuous or ordinal variables. Includes self-masking, ranking-based, and quantile-based missingness strategies.

**Categorical MNAR Types**

:doc:`Categorical Types <mnar_categorical>`  
Applies to discrete or binary variables. Supports class-conditional or value-dependent missingness patterns.

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   mnar_numerical
   mnar_categorical
