# MissMecha: A Missing Data Analysis Toolkit

**MissMecha** is a modular Python toolkit for analyzing, simulating, and visualizing missing data in real-world tabular and time series datasets. It is designed to support both researchers and practitioners in:

- Understanding the structure and mechanism of missingness
- Simulating controlled missing data patterns
- Evaluating imputation and downstream model performance

---

## Features

- **Visualizations**: Matrix view, value-based coloring, and nullity correlation heatmap
- **Generation**: Injects missingness according to MCAR, MAR, MNAR assumptions
- **Evaluation**: Flexible metrics for imputation and prediction under missing data
- **Mixed-type Support**: Handles categorical, numerical, and time-series data

---

## Module Overview

| Module         | Description |
|----------------|-------------|
| `visual`       | Visualization of missing data patterns |
| `generate`     | Simulation of missing data under various mechanisms |
| `analysis`     | Statistical and structural inspection of missingness |
| `evaluation`   | Imputation quality and model performance metrics |

---

## Installation

```bash
pip install missmecha
