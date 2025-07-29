Installation
=======================

MissMecha is available on the Python Package Index (PyPI). You can install it via pip:

.. code-block:: bash

   pip install missmecha-py

This will install the latest stable release.

.. note::

   This package is **MissMecha**, a Python package for simulating and evaluating missing data mechanisms.  
   It is **not** related to the R package `MissMech`, which provides statistical tests for MCAR and multivariate normality.  
   For details on the R package, see: https://cran.r-project.org/package=MissMech

Install from Source
--------------------

To install the latest development version from GitHub:

.. code-block:: bash

   git clone https://github.com/echoid/MissMecha.git
   pip install ./MissMecha

Dependencies
------------

MissMecha requires:

- Python 3.8 or above
- numpy
- scikit-learn
- scipy
- matplotlib (optional, for evaluation plots)

You can install all dependencies at once using:

.. code-block:: bash

   pip install -r requirements.txt

Citation
--------

If you use MissMecha in your research, teaching, or projects, please consider citing:

Zhou, Y. (2025). *MissMecha: An All-in-One Python Package for Studying Missing Data Mechanisms*. Demo paper. [Conference TBD].

BibTeX:

.. code-block:: bibtex

   @misc{zhou2025missmecha,
     author       = {Youran Zhou},
     title        = {MissMecha: An All-in-One Python Package for Studying Missing Data Mechanisms},
     year         = {2025},
     howpublished = {\url{https://pypi.org/project/missmecha-py}},
     note         = {Demo track submission}
   }

