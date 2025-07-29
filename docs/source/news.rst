What's New
===========================

This page highlights recent updates, new features, and improvements to the MissMecha package.

Latest Release: v0.1.2 (2025-07-30)
--------------------------------------

✨ **New Features**
^^^^^^^^^^^^^^^^^^^^

- Added support for **custom missing data mechanisms** via ``MissMechaGenerator(mechanism="custom", custom_class=...)``.
- Now supports **column-wise customization** of custom mechanisms through the ``info`` dictionary.
- Improved interface compatibility with ``fit``/``transform`` convention for user-defined classes.

**Fixes**
^^^^^^^^^^^^^^

- **MNARType1** now adapts to ``missing_rate`` by automatically calculating appropriate lower/upper quantiles when parameters are not specified.

  A huge thanks to **@mahshidkhatiri** for raising [Issue #2](https://github.com/echoid/MissMecha/issues/2) and helping us improve the robustness of MNAR mechanisms! 

**Documentation**
^^^^^^^^^^^^^^^^^^^^^^

- A new tutorial page for **Custom Mechanisms** is now available: :doc:`modules/generate_fun/custom_mechanisms`.
- Improved docstring and examples for ``MissMechaGenerator``.

----

Previous Versions
--------------------

**v0.1.1**

- Initial public release of MissMecha on PyPI
- Support for MCAR, MAR, and MNAR simulation
- Evaluation tools (MCAR test, imputation error)
- Visual tools for missingness patterns

----

Stay tuned for more updates!
----
