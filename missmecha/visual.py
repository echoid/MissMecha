import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from . import util

"""
Visualization module for missing data patterns.

This module provides plotting functions to help understand the structure 
and distribution of missingness in datasets. It includes matrix plots, 
heatmaps, and column-wise summaries.

Functions:
----------
- plot_missing_matrix : Visualize missingness patterns with row/column alignment.
- plot_missing_heatmap : Show missing value correlation heatmap.

These tools are useful during EDA or when diagnosing missing data mechanisms.

Example usage::

    from missmecha import visual
    visual.plot_missing_matrix(X)
"""
def _get_auto_figsize(n_rows, n_cols, base_width=1.2, base_height=0.3, max_size=(20, 12)):
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
def plot_missing_matrix(df, figsize=None, cmap="RdBu", color=True, fontsize=14, label_rotation=45, show_colorbar=False, ts=False):
    """
    Plot a matrix visualization of missing data patterns.

    This function displays the presence and absence of data in a DataFrame as a matrix-style heatmap. 
    It supports either binary missing indicators or color-mapped actual values for observed entries.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame whose missingness pattern will be visualized.
    figsize : tuple of int, optional
        Custom figure size. If None, the size is determined automatically based on the data shape.
    cmap : str, optional
        Colormap to use when `color=True`. Defaults to "RdBu".
    color : bool, optional
        Whether to color the present values using a colormap. If False, uses a fixed gray color.
    fontsize : int, optional
        Font size for column and axis labels. Default is 14.
    label_rotation : int, optional
        Rotation angle for x-axis tick labels (column names and missing rates). Default is 45 degrees.
    show_colorbar : bool, optional
        Whether to show a colorbar for normalized values. Only works when `color=True`.
    ts : bool, optional
        If True, display y-axis using index labels (e.g., for time series data). If False, just label row positions.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object of the main plot.

    Notes
    -----
    - The top axis shows column names; the bottom axis shows missing rates.
    - Fully observed or fully missing columns are retained.
    - If `color=True`, valid (non-missing) values are normalized and mapped using `cmap`.
    - If the dataset is very large, consider pre-sampling before calling this function.

    Examples
    --------
    >>> from missmecha.visual import plot_missing_matrix
    >>> import pandas as pd
    >>> df = pd.read_csv("data.csv")
    >>> plot_missing_matrix(df, color=False)
    
    --------
    """



    height, width = df.shape
    missing_rates = df.isnull().sum() / height * 100
    if figsize is None:
        figsize = _get_auto_figsize(height, width)
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


def plot_missing_heatmap(df, figsize=(20, 12), fontsize=14, label_rotation=45, cmap='RdBu', method = "pearson"):
    """
    Plot a heatmap of pairwise nullity correlations.

    This function visualizes the pairwise correlation between missing value patterns 
    across columns in the input DataFrame. The heatmap helps identify dependencies 
    between missingness in different variables and can guide further missing data analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset to visualize. Each column should represent a feature.
    figsize : tuple of int, optional
        Figure size in inches (width, height). Default is (20, 12).
    fontsize : int, optional
        Font size for axis labels and annotations. Default is 14.
    label_rotation : int, optional
        Rotation angle (in degrees) for x-axis tick labels. Default is 45.
    cmap : str, optional
        Colormap for the heatmap (e.g., 'RdBu', 'viridis'). Default is 'RdBu'.
    method : {'pearson', 'kendall', 'spearman'}, optional
        Correlation method to compute pairwise nullity relationships. Default is 'pearson'.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object containing the plotted heatmap.

    Raises
    ------
    ValueError
        If the input DataFrame does not contain any missing values.

    Notes
    -----
    - Fully observed or fully missing columns are excluded from the plot.
    - If the dataset has more than 1000 rows, a random sample of 1000 rows is used.
    - The heatmap represents correlation between binary indicators of missingness (True/False).

    Examples
    --------
    >>> from missmecha.visual import plot_missing_heatmap
    >>> import pandas as pd
    >>> df = pd.read_csv("my_data.csv")
    >>> plot_missing_heatmap(df)
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

