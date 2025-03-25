import pandas as pd 
import numpy as np

def missing_rate(df, print_summary=True):
    """
    Compute and present detailed missingness statistics for each column,
    along with overall missing rate.

    Returns a dictionary with:
      - report: column-level details
      - overall_missing_rate: overall percentage as float
    """
    total_rows = len(df)
    n_missing = df.isnull().sum()
    missing_rate_pct = (n_missing / total_rows * 100).round(2)
    n_unique = df.nunique(dropna=True)
    dtype = df.dtypes

    report = pd.DataFrame({
        "n_missing": n_missing,
        "missing_rate (%)": missing_rate_pct,
        "n_unique": n_unique,
        "dtype": dtype,
        "n_total": total_rows
    })
    report.index.name = "column"
    report = report.sort_values("missing_rate (%)", ascending=False)

    # Convert overall rate to native float
    total_cells = df.size
    total_missing = df.isnull().sum().sum()
    overall_rate = float(round((total_missing / total_cells) * 100, 2))

    if print_summary:
        print(f"Overall missing rate: {overall_rate:.2f}% ({total_missing} / {total_cells} values are missed)")

    return {
        "report": report,
        "overall_missing_rate": overall_rate
    }
