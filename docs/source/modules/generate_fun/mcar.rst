MCAR Functions
==============

Missing Completely At Random (MCAR) refers to a missing data mechanism in which the probability of missingness is completely independent of both observed and unobserved data.

More formally, given a data matrix :math:`X` and a corresponding missingness indicator matrix :math:`M`, the MCAR assumption implies:

.. math::

   P(M \mid X) = P(M)

This means that missing values are distributed entirely at random, regardless of the data values themselves. MCAR is the most stringent and rarest assumption, but also the easiest to handle analytically.

In MissMecha, we implement several MCAR variants that can be applied to numerical, categorical, or time series data.

.. note::

   MCAR mechanisms in this module support all data types, including **numerical**, **categorical**, and **time series** inputs.



``MCARType1``
---------

.. autoclass:: missmecha.generate.mcar.MCARType1
   :members:
   :undoc-members:
   :show-inheritance:

----

``MCARType2``
---------

.. autoclass:: missmecha.generate.mcar.MCARType2
   :members:
   :undoc-members:
   :show-inheritance:

----

``MCARType3``
---------

.. autoclass:: missmecha.generate.mcar.MCARType3
   :members:
   :undoc-members:
   :show-inheritance:

----

