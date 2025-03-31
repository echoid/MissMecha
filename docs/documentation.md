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
| `analysis`     | Statistical and structural inspection of missingness,Imputation quality and model performance metrics  |

---


## `info` Dictionary Format for Column-wise Missing Generation

The `info` dictionary allows you to define **column-specific missing data settings**, including:
- Missing mechanism (`mcar`, `mar`, `mnar`)
- Mechanism type ID
- Missing rate
- Dependency on other columns
- Optional mechanism-specific parameters

This supports **realistic and flexible missingness simulation** where each column can follow a different mechanism.

---

### ✅ Example Structure

```python
info = {
    "income": {
        "missing_type": "mnar",            # Mechanism: MCAR / MAR / MNAR
        "type": 1,                          # Mechanism variant ID
        "missing_rate": 0.3,               # Percentage of missing values
        "dependent_columns": ["income"],   # For MNAR, can depend on itself
        "missing_para": {"threshold": 40000}  # Optional parameters
    },
    "age": {
        "missing_type": "mar",
        "type": 2,
        "missing_rate": 0.2,
        "dependent_columns": ["gender"],
        "missing_para": {"group_bias": "female"}
    },
    "gender": {
        "missing_type": "mcar",
        "type": 1,
        "missing_rate": 0.1,
        "dependent_columns": []
    }
}
```

### 🧩 Field Descriptions

Each entry in the `info` dictionary defines the missing configuration for one column.

| **Key**            | **Type**        | **Required** | **Example**                | **Description**                                                                 |
|--------------------|-----------------|--------------|----------------------------|---------------------------------------------------------------------------------|
| `missing_type`     | `str`           | ✅            | `"mar"`                    | The type of missingness mechanism: one of `"mcar"`, `"mar"`, or `"mnar"`       |
| `type`             | `int`           | ✅            | `1`                        | Mechanism variant ID; refers to predefined mechanism function (e.g., `MCAR_TYPES[1]`) |
| `missing_rate`     | `float`         | ✅            | `0.2`                      | The proportion of values to be made missing in this column                     |
| `dependent_columns`| `list[str]`     | ✅ (can be empty) | `["age", "gender"]`    | List of variables this column's missingness depends on (useful for MAR/MNAR)  |
| `missing_para`     | `dict`          | ❌            | `{"threshold": 40000}`     | Optional parameters passed to the mechanism function                           |

> 🔧 Note: You can leave `dependent_columns` as an empty list `[]` if not applicable (e.g., for MCAR).



