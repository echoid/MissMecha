# MissMecha

<p align="center">
  <strong>A Python toolkit for simulating, visualizing, imputing, and evaluating missing data mechanisms.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/missmecha-py/"><img alt="PyPI" src="https://img.shields.io/pypi/v/missmecha-py?color=2f6f73"></a>
  <a href="https://pypi.org/project/missmecha-py/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/missmecha-py?color=496ddb"></a>
  <a href="https://echoid.github.io/MissMecha/"><img alt="Documentation" src="https://img.shields.io/badge/docs-online-6f42c1"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"></a>
  <a href="https://vimeo.com/1079046393"><img alt="Demo" src="https://img.shields.io/badge/demo-video-ff69b4"></a>
</p>

**MissMecha** provides a unified, reproducible interface for studying missingness in heterogeneous data. It supports controlled simulation of **MCAR**, **MAR**, and **MNAR** mechanisms, visual diagnosis of missing patterns, baseline imputation, and evaluation utilities for benchmarking imputation workflows.

MissMecha is designed for researchers, educators, and machine learning practitioners who need to create realistic incomplete datasets instead of relying on one-size-fits-all random masking.

> Documentation: <https://echoid.github.io/MissMecha/>
>
> PyPI package: <https://pypi.org/project/missmecha-py/>

---

## Why MissMecha?

Real-world missing data is rarely uniform. A value may be missing because of a random collection error, because another observed feature explains non-response, or because the unobserved value itself influences whether it is reported. MissMecha turns these statistical ideas into a practical Python package.

Use MissMecha to:

- simulate **MCAR**, **MAR**, and **MNAR** missingness with interpretable subtypes;
- configure missingness globally or per column with a scikit-learn style API;
- support numerical, categorical, ordinal, time series, and heterogeneous datasets;
- inspect missingness patterns through matrix and heatmap visualizations;
- evaluate imputations with RMSE, MAE, accuracy, AvgERR, and MCAR tests;
- plug in user-defined missing mechanisms through a simple `fit` / `transform` interface.

---

## Package Overview

<p align="center">
  <img src="https://raw.githubusercontent.com/echoid/MissMecha/main/docs/_static/readme/missmecha_structure2.png" alt="MissMecha package structure" width="900">
</p>

MissMecha is organized around four main modules:

| Module | Purpose | Main tools |
| --- | --- | --- |
| `generator` | Create incomplete datasets under predefined or custom missingness mechanisms. | `MissMechaGenerator` |
| `visual` | Inspect missingness structure with matrix-style plots and heatmaps. | `plot_missing_matrix`, `plot_missing_heatmap` |
| `impute` | Provide a simple adaptive baseline for numerical and categorical imputation. | `SimpleSmartImputer` |
| `analysis` | Summarize missing rates, test MCAR assumptions, and evaluate imputations. | `compute_missing_rate`, `evaluate_imputation`, `MCARTest` |

---

## Installation

Install the latest release from PyPI:

```bash
pip install missmecha-py
```

Or install the development version from source:

```bash
git clone https://github.com/echoid/MissMecha.git
pip install ./MissMecha
```

MissMecha requires Python 3.8 or above and depends on `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, and `seaborn`.

> Note: this package is **MissMecha**, a Python toolkit for missing data mechanisms. It is not related to the R package `MissMech`.

---

## Quick Start

Generate missing values globally across a dataset:

```python
import numpy as np
import pandas as pd
from missmecha import MissMechaGenerator

X = pd.DataFrame(
    np.random.rand(100, 5),
    columns=[f"x{i}" for i in range(5)],
)

generator = MissMechaGenerator(
    mechanism="mar",
    mechanism_type=1,
    missing_rate=0.3,
    seed=42,
)

X_missing = generator.fit_transform(X)
mask = generator.get_bool_mask()
```

Configure different mechanisms for different columns:

```python
from missmecha import MissMechaGenerator

generator = MissMechaGenerator(
    info={
        "x0": {"mechanism": "mcar", "type": 1, "rate": 0.3},
        "x1": {"mechanism": "mnar", "type": 2, "rate": 0.4},
    },
    seed=42,
)

X_missing = generator.fit_transform(X)
```

Evaluate a completed dataset against the original values:

```python
from missmecha.analysis import evaluate_imputation
from missmecha.impute import SimpleSmartImputer

imputer = SimpleSmartImputer()
X_imputed = imputer.fit_transform(X_missing)

scores = evaluate_imputation(
    original_df=X,
    imputed_df=X_imputed,
    mask_array=mask,
    method="rmse",
)
print(scores)
```

---

## Supported Mechanisms

MissMecha currently includes a growing family of mechanism subtypes:

| Mechanism | Description | Examples |
| --- | --- | --- |
| **MCAR** | Missingness is independent of observed and unobserved values. | uniform masking, fixed selection, column-balanced masking |
| **MAR** | Missingness depends on observed variables. | logic models, mutual information based masking, point-biserial relationships |
| **MNAR** | Missingness depends on the target value itself or hidden structure. | quantile masking, self-masking, percentile rules |
| **Custom** | User-defined mechanisms with the same generator interface. | threshold rules, model-based missingness, domain-specific maskers |

For statistical background, see the [Theory](https://echoid.github.io/MissMecha/theory.html) page.

---

## Documentation And Examples

- Full documentation: <https://echoid.github.io/MissMecha/>
- Installation guide: <https://echoid.github.io/MissMecha/installation.html>
- API modules: <https://echoid.github.io/MissMecha/modules.html>
- Examples and notebooks: <https://echoid.github.io/MissMecha/examples.html>
- Custom mechanism demo: <https://echoid.github.io/MissMecha/notebooks/MissMecha-Demo-custom_mechanism.html>
- Five-minute video demo: <https://vimeo.com/1079046393>

---

## Research And Teaching Use Cases

MissMecha can be used to:

- benchmark imputation methods under controlled missingness assumptions;
- construct synthetic incomplete datasets for reproducible experiments;
- teach MCAR, MAR, and MNAR mechanisms with executable examples;
- compare mechanism-specific effects on downstream machine learning models;
- prototype custom missingness processes for domain-specific studies.

---

## Citation

If you use MissMecha in research, teaching, or software projects, please cite:

```bibtex
@misc{zhou2025missmecha,
  author       = {Youran Zhou},
  title        = {MissMecha: An All-in-One Python Package for Studying Missing Data Mechanisms},
  year         = {2025},
  howpublished = {\url{https://pypi.org/project/missmecha-py}},
  note         = {Demo track submission}
}
```

---

## Author

MissMecha is developed and maintained by **Youran Zhou**, PhD candidate at Deakin University, Australia.

- GitHub: <https://github.com/echoid>
- LinkedIn: <https://www.linkedin.com/in/youran-zhou/>

---

## License

MissMecha is released under the [MIT License](LICENSE).
