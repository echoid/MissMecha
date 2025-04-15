Project Overview
================

MissMecha is a Python package for simulating missing data mechanisms in a principled, unified, and reproducible way.

Born out of frustration with the fragmented landscape of missing data simulation tools, MissMecha is the first Python package to systematically support structured missingness across **numerical**, **categorical**, and **time series** data types.

It offers:
- Support for **MCAR**, **MAR**, and **MNAR** mechanisms
- A rich set of **subtypes** per mechanism, with interpretable logic
- **Column-wise** and **global** missingness simulation
- **Evaluation modules** to benchmark imputation methods

Motivation
----------

In real-world data science tasks, missingness is rarely random or uniform. However, most existing libraries simulate only idealized patterns—if at all—often limited to numerical data.

MissMecha addresses this gap by offering a consistent, extensible API for:
- Designing realistic missing data scenarios
- Testing imputation under controlled conditions
- Bridging statistical theory with practical ML workflows

Supported Modules
-----------------

- `missmecha.generator` – missingness simulation engine
- `missmecha.analysis` – MCAR testing, imputation evaluation
- `missmecha.visual` – visualization tools for missing patterns
- `missmecha.impute` – simple baselines for imputation

Where to Start
--------------

- New to missing data? Learn the basics in :doc:`theory`
- Want to simulate? Start with :doc:`modules/generate`
- Explore the rest of the API in :doc:`modules`
