Theory
========================

Understanding the mechanism behind missing data is essential for realistic simulation and effective imputation strategies. In statistics, missingness is typically classified into three types: **MCAR**, **MAR**, and **MNAR**.

.. contents::
   :local:
   :depth: 2
   :hidden:

Introduction
------------

Let :math:`\boldsymbol{X} \in \mathbb{R}^{n \times k}` be a complete data matrix and :math:`\boldsymbol{M} \in \{0,1\}^{n \times k}` be a missingness mask, where :math:`M_{ij} = 1` denotes an observed value and :math:`M_{ij} = 0` a missing one. The conditional distribution of missingness is:

.. math::

   f(\boldsymbol{M} \mid \boldsymbol{X}, \Psi)

where :math:`\Psi` denotes the parameters governing the missing process. Depending on how :math:`f` depends on :math:`\boldsymbol{X}`, we obtain different mechanisms.

.. note::

   MissMecha supports simulation under **MCAR**, **MAR**, and **MNAR** for both numerical and categorical data.

Missing Completely At Random (MCAR)
-----------------------------------

Under MCAR, the missingness does **not depend** on either the observed or missing values:

.. math::

   f(\boldsymbol{M} \mid \boldsymbol{X}, \Psi) = f(\boldsymbol{M} \mid \Psi)

This means that the pattern of missing data is entirely random.

**Example**: Suppose a storage glitch randomly deletes cells from a spreadsheet. The probability of missingness is unrelated to the data itself.

Missing At Random (MAR)
------------------------

Under MAR, the missingness depends on **observed values only**:

.. math::

   f(\boldsymbol{M} \mid \boldsymbol{X}, \Psi) = f(\boldsymbol{M} \mid \boldsymbol{X}^{\text{obs}}, \Psi)

This allows us to condition on known variables when modeling the missingness process.

**Example**: If all missing salaries belong to female employees, and gender is fully observed, then the missingness can be explained by the gender column.

Missing Not At Random (MNAR)
----------------------------

Under MNAR, the missingness depends on the **missing values themselves**, even after conditioning on observed data:

.. math::

   f(\boldsymbol{M} \mid \boldsymbol{X}, \Psi) \text{ depends on } \boldsymbol{X}^{\text{miss}}

This makes modeling and imputation more difficult, as missing values carry information about why they are missing.

**Example**: Individuals with very high income might avoid reporting their income. In this case, missingness is driven by the unobserved value itself.

Illustrative Table
------------------

The table below (from ~\cite{Missing_Mechanisms}) illustrates how different mechanisms affect a toy dataset:

.. list-table:: Missing Mechanism Comparison (Job Ratings by IQ)
   :header-rows: 1
   :widths: 15 15 15 15 15

   * - IQ
     - Ratings
     - MCAR
     - MAR
     - MNAR
   * - 78
     - 9
     - ?
     - ?
     - 9
   * - 84
     - 13
     - 13
     - ?
     - 13
   * - 85
     - 8
     - 8
     - ?
     - ?
   * - 105
     - 11
     - ?
     - 11
     - 11
   * - 118
     - 16
     - 16
     - 16
     - 16

References
----------

- Little, R. J. A., & Rubin, D. B. (2002). *Statistical Analysis with Missing Data* (2nd ed.). Wiley-Interscience.

- Enders, C. K. (2010). *Applied Missing Data Analysis.* The Guilford Press.

- Gomer, B., & Yuan, K. H. (2021). Subtypes of the Missing Not At Random Missing Data Mechanism. *Psychological Methods, 26*(5), 559–598. https://doi.org/10.1037/met0000377
