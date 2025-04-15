import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

def compute_missing_rate(data, print_summary=True, plot=False):
    """
    Compute and summarize missingness statistics for each column.

    This function calculates the number and percentage of missing values 
    for each column in a dataset, and optionally provides a summary table and barplot.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        The dataset to analyze for missingness. If ndarray, it will be converted to DataFrame.
    print_summary : bool, default=True
        If True, prints the overall missing rate and top variables by missing rate.
    plot : bool, default=False
        If True, displays a barplot of missing rates per column.

    Returns
    -------
    result : dict
        A dictionary with:
        - 'report' : pandas.DataFrame with per-column missing statistics.
        - 'overall_missing_rate' : float, overall percentage of missing entries.

    Examples
    --------
    >>> from missmecha.analysis import compute_missing_rate
    >>> df = pd.read_csv("data.csv")
    >>> stats = compute_missing_rate(df, print_summary=True, plot=True)
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=[f"col{i}" for i in range(data.shape[1])])

    total_rows, total_cells = data.shape[0], data.size
    n_missing = data.isnull().sum()
    missing_rate_pct = (n_missing / total_rows * 100).round(2)
    n_unique = data.nunique(dropna=True)
    dtype = data.dtypes.astype(str)

    report = pd.DataFrame({
        "n_missing": n_missing,
        "missing_rate (%)": missing_rate_pct,
        "n_unique": n_unique,
        "dtype": dtype,
        "n_total": total_rows
    }).sort_values("missing_rate (%)", ascending=False)

    report.index.name = "column"
    overall_rate = round((n_missing.sum() / total_cells) * 100, 2)

    if print_summary:
        print(f"Overall missing rate: {overall_rate:.2f}%")
        print(f"{n_missing.sum()} / {total_cells} total values are missing.\n")
        print("Top variables by missing rate:")
        display(report.head(5))

    if plot:
        plt.figure(figsize=(8, max(4, len(report) * 0.3)))
        sns.barplot(x=report["missing_rate (%)"], y=report.index, palette="coolwarm")
        plt.xlabel("Missing Rate (%)")
        plt.title("Missing Rate by Column")
        plt.tight_layout()
        plt.show()

    return {
        "report": report,
        "overall_missing_rate": overall_rate
    }






def evaluate_imputation(ground_truth, filled_df, incomplete_df, method, status=None):
    """
    Evaluate imputation quality by comparing imputed values to ground truth.

    This function calculates per-column and overall performance scores 
    by evaluating the imputed values at originally missing positions.

    Parameters
    ----------
    ground_truth : pandas.DataFrame
        Fully observed reference dataset (i.e., the original complete data).
    filled_df : pandas.DataFrame
        The dataset after imputation has been applied.
    incomplete_df : pandas.DataFrame
        The incomplete dataset used to identify where values were originally missing.
    method : str
        Evaluation method: one of {'rmse', 'mae', 'accuracy'}.
        - 'rmse': Root Mean Squared Error for numeric columns.
        - 'mae': Mean Absolute Error for numeric columns.
        - 'accuracy': Classification accuracy for categorical columns.
    status : dict, optional
        Optional dictionary mapping column names to types:
        - 'num' for numerical, 'cat' or 'disc' for categorical.
        If not provided, all columns are assumed to be numeric.

    Returns
    -------
    result : dict
        {
            'column_scores' : dict mapping column names to scores,
            'overall_score' : float, average of all column scores
        }

    Raises
    ------
    ValueError
        If an unknown evaluation method or column type is encountered.

    Examples
    --------
    >>> from missmecha.analysis import evaluate_imputation
    >>> result = evaluate_imputation(X_true, X_filled, X_miss, method="rmse", status={"age": "num", "gender": "cat"})
    >>> print(result["overall_score"])
    """
    assert method in {"rmse", "mae", "accuracy"}, "method must be 'rmse', 'mae', or 'accuracy'"

    mask = incomplete_df.isnull()
    column_scores = {}

    for col in incomplete_df.columns:
        y_true = ground_truth.loc[mask[col], col]
        y_pred = filled_df.loc[mask[col], col]

        if y_true.empty:
            column_scores[col] = np.nan
            continue

        col_type = status.get(col, "num") if status else "num"

        if col_type == "num":
            if method == "rmse":
                score = mean_squared_error(y_true, y_pred, squared=False)
            elif method == "mae":
                score = mean_absolute_error(y_true, y_pred)
            else:
                raise ValueError(f"Method '{method}' not supported for numeric columns.")
        elif col_type in {"cat", "disc"}:
            if method != "accuracy":
                raise ValueError(f"Use method='accuracy' for categorical columns.")
            score = accuracy_score(y_true.astype(str), y_pred.astype(str))
        else:
            raise ValueError(f"Unsupported column type: {col_type}")

        column_scores[col] = score

    valid_scores = [v for v in column_scores.values() if not np.isnan(v)]
    overall_score = np.mean(valid_scores) if valid_scores else np.nan

    return {
        "column_scores": column_scores,
        "overall_score": overall_score
    }



import numpy as np
import pandas as pd
from scipy.stats import chi2, ttest_ind
from typing import Union


class MCARTest:
    """
    A class to perform MCAR (Missing Completely At Random) tests.

    Supports Little's MCAR test (global test for all variables)
    and pairwise MCAR t-tests (for individual variables).
    """

    def __init__(self, method: str = "little"):
        """
        Parameters
        ----------
        method : {'little', 'ttest'}, default='little'
            The MCAR testing method to use.
            - 'little': Use Little's MCAR test (global p-value).
            - 'ttest': Perform pairwise t-tests for each variable.
        """
        if method not in ["little", "ttest"]:
            raise ValueError("method must be 'little' or 'ttest'")
        self.method = method

    def __call__(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[float, pd.DataFrame]:
        """
        Run the selected MCAR test on the input data.

        Parameters
        ----------
        data : np.ndarray or pd.DataFrame
            Input dataset with missing values.

        Returns
        -------
        result : float or pd.DataFrame
            - A p-value (float) if method='little'.
            - A p-value matrix (pd.DataFrame) if method='ttest'.
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=[f"col{i}" for i in range(data.shape[1])])
        elif not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame or a NumPy array.")

        if self.method == "little":
            return self.little_mcar_test(data)
        elif self.method == "ttest":
            return self.mcar_t_tests(data)

    @staticmethod
    def little_mcar_test(X: pd.DataFrame) -> float:
        """
        Perform Little's MCAR test on a DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataset.

        Returns
        -------
        pvalue : float
            P-value of the test.
        """
        dataset = X.copy()
        vars = dataset.columns
        n_var = dataset.shape[1]

        gmean = dataset.mean()
        gcov = dataset.cov()

        r = dataset.isnull().astype(int)
        mdp = np.dot(r, [2**i for i in range(n_var)])
        sorted_mdp = sorted(np.unique(mdp))
        mdp_codes = [sorted_mdp.index(code) for code in mdp]
        dataset["mdp"] = mdp_codes

        pj = 0
        d2 = 0
        for i in range(len(sorted_mdp)):
            subset = dataset[dataset["mdp"] == i][vars]
            valid_vars = subset.columns[~subset.isnull().any()]
            pj += len(valid_vars)
            means = subset[valid_vars].mean() - gmean[valid_vars]
            cov = gcov.loc[valid_vars, valid_vars]
            mj = len(subset)

            if cov.shape[0] == 0:
                continue

            parta = np.dot(means.T, np.linalg.solve(cov, np.eye(cov.shape[0])))
            d2 += mj * np.dot(parta, means)

        df = pj - n_var
        pvalue = 1 - chi2.cdf(d2, df)
        MCARTest.report(pvalue, method="Little's MCAR Test")
        return pvalue

    @staticmethod
    def mcar_t_tests(X: pd.DataFrame) -> pd.DataFrame:
        """
        Perform pairwise MCAR t-tests between missing and observed groups.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataset.

        Returns
        -------
        p_matrix : pd.DataFrame
            Matrix of p-values (var vs var).
        """
        vars = X.columns
        p_matrix = pd.DataFrame(np.nan, index=vars, columns=vars)

        for var in vars:
            for tvar in vars:
                group1 = X.loc[X[var].isnull(), tvar].dropna()
                group2 = X.loc[X[var].notnull(), tvar].dropna()

                if len(group1) > 1 and len(group2) > 1:
                    p = ttest_ind(group1, group2, equal_var=False).pvalue
                    p_matrix.loc[var, tvar] = p

        return p_matrix

    @staticmethod
    def report(pvalue: float, alpha: float = 0.05, method: str = "Little's MCAR Test") -> None:
        """
        Print a summary report of the MCAR test.

        Parameters
        ----------
        pvalue : float
            The p-value from the MCAR test.
        alpha : float, default=0.05
            Significance level.
        method : str, default="Little's MCAR Test"
            Method name shown in report.
        """
        print(f"Method: {method}")
        print(f"Test Statistic p-value: {pvalue:.6f}")

        if pvalue < alpha:
            print(f"Decision: Reject the null hypothesis (α = {alpha})")
            print("→ The data is unlikely to be Missing Completely At Random (MCAR).")
        else:
            print(f"Decision: Fail to reject the null hypothesis (α = {alpha})")
            print("→ There is insufficient evidence to reject MCAR.")
