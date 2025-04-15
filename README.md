# 🧩 MissMecha

**MissMecha** is a Python package dedicated to the systematic simulation, visualization, and evaluation of missing data mechanisms. Our goal is to provide a unified interface for generating, inspecting, and analyzing missingness — supporting research, benchmarking, and education.

---

https://echoid.github.io/MissMecha/


## Highlights

- 🔍 **All About Missing Mechanisms**
  - Simulate **MCAR**, **MAR**, and **MNAR** in flexible formats
  - Currently supports:
    - **3× MCAR** strategies
    - **8× MAR** strategies
    - **6× MNAR** strategies
    - Experimental support for **categorical** and **time series** missingness

- **Missingness Pattern Visualization**
  - Visual tools to **observe missing patterns**
  - Helps identify potential mechanism types (MCAR vs MAR vs MNAR)

- **Flexible Generator Interface**
  - Column-wise or global missingness simulation
  - Sklearn-style `fit` / `transform` methods
  - Supports custom rates, dependencies, labels, and logic

- **Evaluation Toolkit**
  - Quantitative metrics including **MSE**, **MAE**, **RMSE**, and **AvgERR**
  - Built-in support for **Little’s MCAR test**

- **SimpleSmartImputer**
  - Lightweight imputer that automatically detects **numerical** and **categorical** columns
  - Applies **mean** or **mode** imputation with verbose diagnostics

---

## Motivation

Working with missing data often involves disparate implementations and inconsistent assumptions across studies.  
**MissMecha** brings together widely used missing data mechanisms into a single, structured, and reproducible Python framework.

> Whether you're designing benchmark experiments or exploring real-world data — MissMecha lets you simulate and analyze missingness with clarity and control.

---

## ⚡ Quick Preview

```python
from missmecha import MissMechaGenerator

generator = MissMechaGenerator(
    info={
        0: {"mechanism": "mar", "type": 1, "rate": 0.3}
    }
)
generator.fit(X)
X_missing = generator.transform(X)
```

---

## Documentation & Notebooks

- Full documentation: [Link coming soon]
- Demo notebooks:
  - `demo_generate.ipynb`
  - `demo_analysis.ipynb`
  - `demo_visual.ipynb`

---

## Installation

```bash
pip install missmecha-py  # Coming soon on PyPI
```

---

## Author

Developed by **Youran Zhou**, PhD Candidate @ Deakin University  
With support from the open-source research community ❤️

---

## 📄 License

MIT License
