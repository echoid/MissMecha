MAR Functions
=============

This section introduces functions for simulating Missing At Random (MAR) mechanisms.

MAR assumes that the probability of missingness in a variable depends only on the observed data. 
In practice, this type of missingness is common and can be leveraged to design more realistic simulations.

We provide two sets of MAR functions:

- **Numerical MAR Types**: Designed for continuous or ordinal numerical variables, including logistic-based and extreme-value-based masking.
- **Categorical MAR Types**: Designed for discrete or binary variables, supporting group-wise or correlation-based masking.

You can explore each category below:

.. toctree::
   :maxdepth: 2
   :titlesonly:

   mar_numerical
   mar_categorical
