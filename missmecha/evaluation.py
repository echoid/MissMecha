import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score


def imp_eval(
    ground_truth: pd.DataFrame,
    filled_df: pd.DataFrame,
    incomplete_df: pd.DataFrame,
    method: str,  # required
    status: dict = None  # optional
):
    """
    Evaluate imputation performance by comparing filled values to ground truth at missing positions.

    Parameters
    ----------
    ground_truth : pd.DataFrame
        Fully observed reference dataset.
    filled_df : pd.DataFrame
        The dataset after imputation.
    incomplete_df : pd.DataFrame
        Dataset with original missing values (to locate evaluation positions).
    method : str
        'rmse', 'mae', or 'avgerr' — determines how numerical columns are evaluated.
    status : dict, optional
        A dictionary mapping column names to variable types:
        'num' for numerical, 'cat' or 'disc' for categorical.
        If not provided, all variables are treated as numerical.

    Returns
    -------
    result : dict
        {
            "column_scores": {col_name: score, ...},
            "overall_score": average_score
        }
    """
    assert method in ["rmse", "mae"], "method must be one of: 'rmse', 'mae', 'avgerr'"

    mask = incomplete_df.isnull()
    column_scores = {}

    for col in incomplete_df.columns:
        col_type = status[col] if status and col in status else "num"
        y_true = ground_truth.loc[mask[col], col]
        y_pred = filled_df.loc[mask[col], col]

        if y_true.empty:
            column_scores[col] = np.nan
            continue

        if col_type == "num":
            if method == "rmse":
                score = mean_squared_error(y_true, y_pred) ** 0.5
            elif method == "mse":
                score = mean_absolute_error(y_true, y_pred)
            elif method == "mae":
                score = mean_absolute_error(y_true, y_pred)

        elif col_type in ["cat", "disc"]:
            score = accuracy_score(y_true.astype(str), y_pred.astype(str))
        else:
            raise ValueError(f"Unsupported variable type: '{col_type}'")

        column_scores[col] = score
    valid = [v for v in column_scores.values() if not np.isnan(v)]
    overall_score = np.mean(valid) if valid else np.nan

    return {
        "column_scores": column_scores,
        "overall_score": overall_score
    }
