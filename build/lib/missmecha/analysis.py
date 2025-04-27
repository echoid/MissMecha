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





from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import numpy as np


def evaluate_imputation(original_df, imputed_df, mask_array, method="rmse", cat_cols=None):
    """
    Evaluate imputation quality by comparing imputed values to ground truth.

    This function computes per-column and overall evaluation scores based on the positions 
    that were originally missing. It supports mixed-type data by applying different metrics
    for categorical and numerical columns. Returns both original and scaled (0-1) versions
    of the evaluation metrics.

    Parameters
    ----------
    original_df : pd.DataFrame
        The fully observed reference dataset (i.e., ground truth).
    imputed_df : pd.DataFrame
        The dataset after imputation has been applied.
    mask_array : np.ndarray or pd.DataFrame of bool
        Boolean array where True = originally observed, False = originally missing.
        Usually obtained from MissMechaGenerator.bool_mask.
    method : str, default="rmse"
        Evaluation method to use for numeric columns.
        One of {'rmse', 'mae', 'accuracy'}.
    cat_cols : list of str, optional
        Column names that should be treated as categorical. These will always use accuracy.
        - If not provided, all columns use the method specified by `method`.

    Returns
    -------
    result : dict
        Dictionary with two sub-dictionaries:
        - 'original': Contains raw evaluation scores
            - 'column_scores': mapping from column name to evaluation score
            - 'overall_score': average of valid column scores (float)
        - 'scaled': Contains normalized scores (0-1 range)
            - 'column_scores': mapping from column name to scaled evaluation score
            - 'overall_score': average of valid scaled column scores (float)
        For categorical columns, the scaled score equals the original accuracy score.

    Raises
    ------
    ValueError
        If an unsupported method or column type is used.

    Notes
    -----
    - If `cat_cols` is None: all columns use the selected `method`.
    - If `cat_cols` is provided:
        - columns in `cat_cols` use accuracy
        - all other columns use `method`, which must be 'rmse' or 'mae'
    - Includes formatted print output.

    Examples
    --------
    >>> from missmecha.analysis import evaluate_imputation
    >>> result = evaluate_imputation(X_true, X_filled, mask, method="rmse")

    >>> result = evaluate_imputation(
    ...     original_df=X_true,
    ...     imputed_df=X_filled,
    ...     mask_array=mask,
    ...     method="mae",
    ...     cat_cols=["gender", "job_type"]
    ... )
    >>> print(result["overall_score"])
    0.872
    """
    
    def safe_compare(y_true, y_pred):
        # 统一转为字符串，但处理数值的字符串形式（如 "5.0" -> "5"）
        y_true_str = [str(int(x)) if str(x).replace(".", "").isdigit() else str(x) for x in y_true]
        y_pred_str = [str(int(x)) if str(x).replace(".", "").isdigit() else str(x) for x in y_pred]
        return accuracy_score(y_true_str, y_pred_str)

    if method not in {"rmse", "mae", "accuracy"}:
        raise ValueError("Method must be one of 'rmse', 'mae', or 'accuracy'.")

    if isinstance(mask_array, np.ndarray):
        mask_df = pd.DataFrame(mask_array, columns=original_df.columns, index=original_df.index)
    else:
        mask_df = mask_array.copy()

    cat_cols = set(cat_cols or [])
    column_scores = {}
    score_pool = []

    # 初始化结果存储
    results = {
        "original": {"column_scores": {}, "overall_score": None},
        "scaled": {"column_scores": {}, "overall_score": None},
    }
    
    # 计算原始误差
    for col in original_df.columns:
        y_true = original_df.loc[~mask_df[col], col]
        y_pred = imputed_df.loc[~mask_df[col], col]

        if y_true.empty:
            results["original"]["column_scores"][col] = np.nan
            results["scaled"]["column_scores"][col] = np.nan
            continue

        if col not in (cat_cols or []):
            # 数值列：计算原始误差
            if method == "rmse":
                raw_score = mean_squared_error(y_true, y_pred, squared=False)
            elif method == "mae":
                raw_score = mean_absolute_error(y_true, y_pred)
            
            # 缩放误差到 [0,1]（基于列的最大可能误差）
            col_range = original_df[col].max() - original_df[col].min()
            scaled_score = raw_score / col_range if col_range > 0 else 0.0
        else:
            # 分类列：准确率已经是 [0,1]，无需缩放
            raw_score = safe_compare(y_true, y_pred)
            scaled_score = raw_score

        results["original"]["column_scores"][col] = raw_score
        results["scaled"]["column_scores"][col] = scaled_score

    # 计算原始和缩放的 Overall 分数
    valid_original_scores = [s for s in results["original"]["column_scores"].values() if not np.isnan(s)]
    valid_scaled_scores = [s for s in results["scaled"]["column_scores"].values() if not np.isnan(s)]
    
    results["original"]["overall_score"] = np.mean(valid_original_scores) if valid_original_scores else np.nan
    results["scaled"]["overall_score"] = np.mean(valid_scaled_scores) if valid_scaled_scores else np.nan
    
    if cat_cols:
        method = "AvgErr"
    else:
        method = method.upper()

    # Pretty print
    print("-" * 50)
    print(f"{'Column':<12}{method:>15}{'Scaled (0-1)':>15}")
    print("-" * 50)
    for col in original_df.columns:
        original_str = f"{results['original']['column_scores'][col]:>15.3f}" if not np.isnan(results['original']['column_scores'][col]) else f"{'N/A':>15}"
        scaled_str = f"{results['scaled']['column_scores'][col]:>15.3f}" if not np.isnan(results['scaled']['column_scores'][col]) else f"{'N/A':>15}"
        print(f"{col:<12}{original_str}{scaled_str}")
    print("-" * 50)
    print(f"{'Overall':<12}{results['original']['overall_score']:>15.3f}{results['scaled']['overall_score']:>15.3f}")


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
