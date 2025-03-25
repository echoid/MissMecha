import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
from .. import util
import seaborn as sns

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
def matrix(df, figsize=None,cmap="RdBu", color=True, fontsize=14, label_rotation=45, show_colorbar=False,ts = False):
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



def heatmap(df, figsize=(20, 12), fontsize=14, label_rotation=45, cmap='RdBu', method = "pearson"):
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

