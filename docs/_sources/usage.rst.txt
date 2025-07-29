Project Overview
================

MissMecha is a Python toolkit for simulating missing data mechanisms in a principled, unified, and reproducible way.

Frustrated by the fragmented landscape of missing data tools, we built MissMecha as the first Python library to systematically support structured missingness across **numerical**, **categorical**, and **time series** data.

MissMecha offers:

- Support for **MCAR**, **MAR**, and **MNAR** mechanisms
- A wide range of **subtypes** per mechanism, each with interpretable logic
- Flexible **global** and **column-wise** missingness simulation
- Built-in **analysis modules** for evaluating and benchmarking imputation methods

Motivation
----------

In real-world machine learning and data science tasks, missingness is rarely random or uniform.  
Yet most existing tools either oversimplify missing patterns or focus narrowly on numerical features.

**MissMecha** closes this gap by providing a **consistent, extensible** framework for:

- Designing realistic and customizable missing data scenarios
- Evaluating imputation strategies under controlled conditions
- Bridging statistical theory with practical machine learning workflows

Where to Start
--------------

- New to missing data? Learn the basics in :doc:`theory`
- Ready to simulate? Start with :doc:`modules/generate`
- Explore the full API in :doc:`modules`
