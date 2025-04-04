Package ``pyampute``
====================

With ``pyampute``, we provide :class:`~pyampute.ampute.MultivariateAmputation`: a transformer for generating missing values in complete datasets. This is useful for evaluating the effect of missing values on your outcome, mostly in experimental settings, but also as a preprocessing step in developing models. 

Additionally, we provide functionality for inspecting incomplete datasets: :class:`~pyampute.exploration.md_patterns.mdPatterns` for displaying missing data patterns and :class:`~pyampute.exploration.mcar_statistical_tests.MCARTest` for performing a statistical hypothesis test for a MCAR mechanis.

.. toctree::
   :maxdepth: 4

   pyampute.ampute
   pyampute.exploration


