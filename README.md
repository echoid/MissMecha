# MissMecha

**MissMecha** is a Python package for the **systematic simulation, visualization, and evaluation** of missing data mechanisms.  
It provides a **unified, principled interface** to generate, inspect, and analyze missingness — supporting research, benchmarking, and education.

 **Documentation:** [https://echoid.github.io/MissMecha/](https://echoid.github.io/MissMecha/)  
 **Demo Video:** [MissMecha: Flexible Missing Data Simulation (5-min Demo)](https://vimeo.com/1079046393)

---

## Highlights

- **All About Missing Mechanisms**
  - Simulate **MCAR**, **MAR**, and **MNAR** with flexible configuration
  - Currently supports:
    - `3×` MCAR strategies
    - `8×` MAR strategies
    - `6×` MNAR strategies
    - Experimental support for **categorical** and **time series** missingness

- **Missingness Pattern Visualization**
  - Visual tools to **inspect missing patterns** and detect possible mechanism types (MCAR, MAR, MNAR)

- **Flexible Generator Interface**
  - Column-wise or global simulation
  - **Scikit-learn style API** (`fit`, `transform`)
  - Customize missing rates, dependencies, or simulate label-dependent missingness

- **Evaluation Toolkit**
  - Evaluate imputation with **RMSE**, **MAE**, **accuracy**, or hybrid **AvgERR** metric
  - Built-in statistical test: **Little’s MCAR test**

- **SimpleSmartImputer**
  - Lightweight, automatic imputer that detects column types
  - Mean for numerical columns, Mode for categorical columns, with verbose reporting

---

## Motivation

Working with missing data often means dealing with **fragmented, inconsistent tools**.

**MissMecha** solves this by offering a **unified, reproducible, and flexible** framework for simulating and analyzing missingness — covering the full range of MCAR, MAR, and MNAR patterns.

> Whether you're exploring datasets, designing controlled experiments, or teaching statistics —  
> **MissMecha** brings structure and clarity to missing data problems.

---

## Quick Preview

```python
from missmecha import MissMechaGenerator
import numpy as np

X = np.random.rand(100, 5)

generator = MissMechaGenerator(
    mechanism="mar", mechanism_type=1, missing_rate=0.3
)
X_missing = generator.fit_transform(X)
```

Or configure different mechanisms for each column:

```python
generator = MissMechaGenerator(
    info={
        0: {"mechanism": "mcar", "type": 1, "rate": 0.3},
        1: {"mechanism": "mnar", "type": 2, "rate": 0.4}
    }
)
X_missing = generator.fit_transform(X)
```

> **Watch a 5-minute live demo here:** [MissMecha: Flexible Missing Data Simulation (Vimeo)](https://vimeo.com/1079046393)

---

## Documentation & Demos

- Explore full API and tutorials: [https://echoid.github.io/MissMecha/](https://echoid.github.io/MissMecha/)

---

## Installation

```bash
pip install missmecha-py
```

Available on [PyPI](https://pypi.org/project/missmecha-py/) under the package name `missmecha-py`.

---

## ✍️ Author

Developed by **Youran Zhou**, PhD Candidate @ Deakin University  

---

## License

MIT License

