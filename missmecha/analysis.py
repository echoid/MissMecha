import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import matplotlib.colors as mcolors
from . import util


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

def compute_missing_rate(data, print_summary=True, plot=False):
    """
    Compute and present detailed missingness statistics for each column,
    along with overall missing rate. Supports both DataFrame and ndarray.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Input dataset.
    print_summary : bool, default=True
        Whether to print overall missingness summary.
    plot : bool, default=False
        Whether to show a barplot of missing rate per column.

    Returns
    -------
    result : dict
        {
            "report": pd.DataFrame of column-wise missing info,
            "overall_missing_rate": float (total % of missing cells)
        }
    """

    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=[f"col{i}" for i in range(data.shape[1])])

    total_rows = len(data)
    total_cells = data.size
    n_missing = data.isnull().sum()
    missing_rate_pct = (n_missing / total_rows * 100).round(2)
    total_missing = n_missing.sum()

    n_unique = data.nunique(dropna=True)
    dtype = data.dtypes

    report = pd.DataFrame({
        "n_missing": n_missing,
        "missing_rate (%)": missing_rate_pct,
        "n_unique": n_unique,
        "dtype": dtype.astype(str),
        "n_total": total_rows
    }).sort_values("missing_rate (%)", ascending=False)

    report.index.name = "column"
    overall_rate = float(round((total_missing / total_cells) * 100, 2))

    if print_summary:
        print(f"Overall missing rate: {overall_rate:.2f}%")
        print(f"{total_missing} / {total_cells} total values are missing.")
        print("\nTop variables by missing rate:")
        display(report.head(5))

    if plot:
        plt.figure(figsize=(7, 4))
        sns.barplot(
            x=report["missing_rate (%)"],
            y=report.index,
            palette="coolwarm"
        )
        plt.xlabel("Missing Rate (%)")
        plt.title("Missing Rate by Column")
        plt.tight_layout()
        plt.show()

    return {
        "report": report,
        "overall_missing_rate": overall_rate
    }





def evaluate_imputation(
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





def get_auto_figsize(n_rows, n_cols, base_width=1.2, base_height=0.3, max_size=(20, 12)):
    """
    Compute dynamic figsize based on DataFrame shape.
    
    base_width: how wide each column should be (in inches)
    base_height: how tall each row should be (in inches)
    max_size: cap the maximum figsize to avoid excessive size
    """
    width = min(max_size[0], max(6, n_cols * base_width))
    height = min(max_size[1], max(4, n_rows * base_height))
    return (width, height)

#def matrix(df, figsize=(20, 12), cmap="RdBu", color=True, fontsize=14, label_rotation=45, show_colorbar=False,ts = False):
def plot_missing_matrix(df, figsize=None,cmap="RdBu", color=True, fontsize=14, label_rotation=45, show_colorbar=False,ts = False):
    """
    Visualizes missing data in a DataFrame as a heatmap.

    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame to visualize.
    figsize : tuple
        Size of the output figure.
    cmap : str
        Colormap to use when `color=True`.
    color : bool
        Whether to use colormap or a fixed color for present values.
    fontsize : int
        Font size for labels.
    label_rotation : int
        Rotation angle for x-axis labels.
    show_colorbar : bool
        Whether to display a colorbar for colormap mode.

    Returns:
    -------
    ax : matplotlib.axes.Axes
        Main plot axis object.
    """


    height, width = df.shape
    missing_rates = df.isnull().sum() / height * 100
    if figsize is None:
        figsize = get_auto_figsize(height, width)
    # Build RGB matrix
    if not color:
        fixed_color = (0.25, 0.25, 0.25)
        g = np.full((height, width, 3), 1.0)
        g[df.notnull().values] = fixed_color
    else:
        data_array = util.type_convert(df)
        for col in range(width):
            col_data = data_array[:, col]
            valid_mask = ~np.isnan(col_data)
            if valid_mask.any():
                min_val, max_val = np.nanmin(col_data), np.nanmax(col_data)
                if min_val != max_val:
                    data_array[valid_mask, col] = (col_data[valid_mask] - min_val) / (max_val - min_val) + 1
                else:
                    data_array[valid_mask, col] = 1
        norm = mcolors.Normalize(vmin=0, vmax=1.5)
        cmap = plt.get_cmap(cmap)
        g = np.full((height, width, 3), 1.0)
        for col in range(width):
            col_data = data_array[:, col]
            valid_mask = ~np.isnan(col_data)
            g[valid_mask, col] = cmap(norm(col_data[valid_mask]))[:, :3]

   # === Plot ===
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(g, interpolation="none", aspect="auto")
    ax.grid(False)

    # Remove all default x-axis ticks/labels from base ax
    ax.set_xticks([])
    ax.set_xticklabels([])

    # --- Top: Column Names ---
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(range(width))
    ax_top.set_xticklabels(df.columns, rotation=label_rotation, ha="left", fontsize=fontsize)
    ax_top.xaxis.set_ticks_position("top")
    ax_top.xaxis.set_label_position("top")

    # --- Bottom: Missing Rates ---
    ax_bottom = ax.twiny()
    ax_bottom.set_xlim(ax.get_xlim())
    ax_bottom.set_xticks(range(width))
    ax_bottom.set_xticklabels([f"{rate:.1f}%" for rate in missing_rates],
                               rotation=label_rotation, ha="right", fontsize=fontsize - 2)
    ax_bottom.xaxis.set_ticks_position("bottom")
    ax_bottom.xaxis.set_label_position("bottom")

    # Y-axis row labels
    if not ts:
        ax.set_yticks([0, height - 1])
        ax.set_yticklabels([1, height], fontsize=fontsize)
    else:
        # Show a fixed maximum number of y-axis labels (e.g., 50)
        max_labels = 50
        step = max(1, height // max_labels)
        ticks = list(range(0, height, step))
        if height - 1 not in ticks:
            ticks.append(height - 1)  # Ensure last row is labeled

        ax.set_yticks(ticks)
        ax.set_yticklabels([df.index[i] for i in ticks], fontsize=fontsize)

    # Optional: colorbar
    if color and show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.02)
        cbar.set_label("Normalized Values", fontsize=fontsize)

    plt.tight_layout()
    plt.show()
    #return ax





# def heatmap(
#     df,
#     figsize=(20, 12),
#     cmap="RdBu",
#     color=True,
#     fontsize=14,
#     label_rotation=45,
#     show_colorbar=False,
#     show_annotations=True,
#     method="pearson",
#     random_state=42
# ):
#     """
#     Visualizes nullity correlation (correlation between missingness) as a full square heatmap.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Input DataFrame containing missing values.
#     cmap : str
#         Colormap for the heatmap.
#     color : bool
#         Placeholder for API consistency; unused in this function.
#     fontsize : int
#         Font size for tick labels and annotations.
#     label_rotation : int
#         Rotation angle for x-axis tick labels.
#     show_colorbar : bool
#         Whether to display the colorbar.
#     show_annotations : bool
#         Whether to annotate the heatmap with correlation values.
#     method : str
#         Correlation method: 'pearson', 'kendall', or 'spearman'.
#     random_state : int
#         Random seed used when sampling.

#     Returns
#     -------
#     ax : matplotlib.axes.Axes
#         The matplotlib axis object.
#     """



#     # Step 2: Convert values (but preserve structure)
#     converted = util.type_convert(df)
#     df_converted = pd.DataFrame(converted, columns=df.columns, index=df.index)

#     # Step 3: Remove columns without missingness variation
#     missing_vars = df_converted.isnull().var() > 0
#     df_used = df_converted.loc[:, missing_vars]

#     if df_used.shape[1] == 0:
#         raise ValueError("No missing values found in the dataset.")

#     # Step 4: Compute nullity correlation matrix using specified method
#     corr_mat = df_used.isnull().corr(method=method)


#     # Step 6: Plot full heatmap
#     fig, ax = plt.subplots(figsize=figsize)
#     sns.heatmap(
#         corr_mat,
#         cmap=cmap,
#         vmin=-1,
#         vmax=1,
#         square=True,
#         cbar=show_colorbar,
#         annot=show_annotations,
#         fmt=".2f" if show_annotations else "",
#         annot_kws={"size": fontsize - 2},
#         ax=ax
#     )

#     # Format ticks
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=label_rotation, ha='right', fontsize=fontsize)
#     ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fontsize)

#     plt.tight_layout()
#     plt.show()

#     return ax

# missmecha/analysis/mcar_test.py

import numpy as np
import pandas as pd
from scipy.stats import chi2, ttest_ind
from math import pow
from typing import Union

from typing import Union
import numpy as np
import pandas as pd
from scipy.stats import chi2, ttest_ind

class MCARTest:
    def __init__(self, method: str = "little"):
        self.method = method

    def __call__(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[float, pd.DataFrame]:
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=[f"col{i}" for i in range(data.shape[1])])
        elif not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame or a NumPy array.")

        if self.method == "little":
            return self.little_mcar_test(data)
        elif self.method == "ttest":
            return self.mcar_t_tests(data)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    @staticmethod
    def little_mcar_test(X: pd.DataFrame) -> float:
        dataset = X.copy()
        vars = dataset.columns
        n_var = dataset.shape[1]

        gmean = dataset.mean()
        gcov = dataset.cov()

        r = 1 * dataset.isnull()
        mdp = np.dot(r, list(map(lambda x: pow(2, x), range(n_var))))
        sorted_mdp = sorted(np.unique(mdp))
        n_pat = len(sorted_mdp)
        correct_mdp = list(map(lambda x: sorted_mdp.index(x), mdp))
        dataset["mdp"] = pd.Series(correct_mdp, index=dataset.index)

        pj = 0
        d2 = 0
        for i in range(n_pat):
            dataset_temp = dataset.loc[dataset["mdp"] == i, vars]
            select_vars = ~dataset_temp.isnull().any()
            pj += np.sum(select_vars)
            select_vars = vars[select_vars]
            means = dataset_temp[select_vars].mean() - gmean[select_vars]
            select_cov = gcov.loc[select_vars, select_vars]
            mj = len(dataset_temp)
            parta = np.dot(
                means.T, np.linalg.solve(select_cov, np.identity(select_cov.shape[0]))
            )
            d2 += mj * (np.dot(parta, means))

        df = pj - n_var
        pvalue = 1 - chi2.cdf(d2, df)

        report_mcar_test(pvalue)
        return pvalue

    @staticmethod
    def mcar_t_tests(X: pd.DataFrame) -> pd.DataFrame:
        dataset = X.copy()
        vars = dataset.columns
        mcar_matrix = pd.DataFrame(
            data=np.zeros((len(vars), len(vars))), columns=vars, index=vars
        )

        for var in vars:
            for tvar in vars:
                part_one = dataset.loc[dataset[var].isnull(), tvar].dropna()
                part_two = dataset.loc[~dataset[var].isnull(), tvar].dropna()
                if len(part_one) > 0 and len(part_two) > 0:
                    mcar_matrix.loc[var, tvar] = ttest_ind(part_one, part_two, equal_var=False).pvalue
                else:
                    mcar_matrix.loc[var, tvar] = np.nan

        return mcar_matrix
def report_mcar_test(pvalue, alpha=0.05, method="Little's MCAR Test"):
    """
    Generate a formal report of the MCAR hypothesis test results.

    Parameters
    ----------
    pvalue : float
        The p-value obtained from the MCAR test.

    alpha : float, default=0.05
        Significance level for hypothesis testing.

    method : str, default="Little's MCAR Test"
        The name of the test used to assess MCAR.

    Returns
    -------
    None
    """
    print(f"Method: {method}")
    print(f"Test Statistic p-value: {pvalue:.6f}")

    if pvalue < alpha:
        print(f"Decision: Reject the null hypothesis at significance level α = {alpha}")
        print("The data is unlikely to be Missing Completely At Random (MCAR).")
    else:
        print(f"Decision: Fail to reject the null hypothesis at significance level α = {alpha}")
        print("Interpretation: There is insufficient evidence to suggest the data deviates from MCAR.")



def plot_missing_heatmap(df, figsize=(20, 12), fontsize=14, label_rotation=45, cmap='RdBu', method = "pearson"):
    """
    Visualizes nullity correlation in the DataFrame using a heatmap.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data to analyze missing patterns.
    figsize : tuple
        Figure size.
    fontsize : int
        Font size for axis labels and annotations.
    label_rotation : int
        Rotation angle for x-axis labels.
    cmap : str
        Colormap used for the heatmap.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object for the heatmap.
    """
    # Step 1: Sample if too large
    if df.shape[0] > 1000:
        df = df.sample(n=1000, random_state=42)
    # Convert types but preserve columns/index
    converted_array = util.type_convert(df)
    df_converted = pd.DataFrame(converted_array, columns=df.columns, index=df.index)

    # Remove fully observed or fully missing columns
    missing_vars = df_converted.isnull().var(axis=0) > 0
    df_used = df_converted.loc[:, missing_vars]

    if df_used.shape[1] == 0:
        raise ValueError("No missing values found in the dataset.")

    # Compute nullity correlation
    corr_mat = df_used.isnull().corr(method=method)

    mask = np.ones_like(corr_mat, dtype=bool)

    # Plot heatmap
    plt.figure(figsize=figsize)
    ax = sns.heatmap(corr_mat, cmap=cmap, vmin=-1, vmax=1,
                     cbar=True, annot=True, fmt=".2f", annot_kws={"size": fontsize - 2})

    ax.set_xticklabels(ax.get_xticklabels(), rotation=label_rotation, ha='right', fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fontsize)

    plt.show()
