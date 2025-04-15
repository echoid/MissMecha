MAR Functions
=============

Missing At Random (MAR) refers to a missing data mechanism where the probability of missingness depends only on the **observed** data.

Formally, let :math:`X` be the data matrix and :math:`M` the missingness indicator. Then under MAR:

.. math::

   P(M \mid X) = P(M \mid X_{\text{obs}})

This implies that the probability of missingness may depend on observed variables, but **not** on the values that are missing themselves.

In MissMecha, we provide multiple MAR variants for both **numerical** and **categorical** data. These functions are especially useful for simulating realistic missingness patterns in applied data science tasks.

.. note::

   MAR mechanisms support all data types, including **numerical**, **categorical**, and **time series** inputs (when treated as tabular data).

----



**Numerical MAR Types**

:doc:`Numerical Types <mar_numerical>`  
Suitable for continuous or ordinal variables. Includes logistic regression masking, correlation-based ranking, and extreme-value-based strategies.

**Categorical MAR Types**

:doc:`Categorical Types <mar_categorical>`  
Designed for discrete or binary variables. Supports group-based missingness and masking driven by feature associations.


.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   mar_numerical
   mar_categorical
