import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
from .. import util
import seaborn as sns



def matrix(df,figsize=(20, 12), cmap="RdBu", color=True, fontsize=14, label_rotation=45):
    """
    Visualizes missing data in a DataFrame as a heatmap with either a single color or a column-wise colormap.

    :param df: Pandas DataFrame to visualize.
    :param figsize: Tuple, size of the figure (default=(12, 6)).
    :param cmap: String, colormap for present values (default="viridis").
    :param color: Tuple, single RGB color for present values (default=None, which means use cmap).
    :param fontsize: Integer, font size for labels (default=14).
    :param label_rotation: Integer, rotation angle for column labels (default=45).

    :return: None (Displays the plot)
    """
    
    height, width = df.shape

    # Compute missing value percentage for each column
    missing_rates = df.isnull().sum() / height * 100  
    # If color is provided, override colormap behavior
    if not color:
        color = (0.25, 0.25, 0.25)
        # Single color mode (fixed color for all present values)
        g = np.full((height, width, 3), 1.0)  # Initialize with white (for missing)
        g[df.notnull().values] = color  # Apply fixed color to present values
    else:
        #data_array = df.to_numpy(dtype=float) 
        data_array = util.type_convert(df)
        # Normalize column-wise between 0 and 1 (ignoring NaNs)
        for col in range(width):
            col_data = data_array[:, col]
            valid_mask = ~np.isnan(col_data)
            if valid_mask.any():  # Avoid division by zero for completely missing columns
                min_val, max_val = np.nanmin(col_data), np.nanmax(col_data)
                if min_val != max_val:  # Only scale if there's variation
                    data_array[valid_mask, col] = (col_data[valid_mask] - min_val) / (max_val - min_val)+1
                else:
                    data_array[valid_mask, col] = 1

        # Create a colormap normalization
        norm = mcolors.Normalize(vmin=0, vmax=1.5)
        cmap = plt.get_cmap(cmap)

        # Create an RGB array with normalized colors (missing values stay white)
        g = np.full((height, width, 3), 1.0)  # Initialize with white (for missing)
        for col in range(width):
            col_data = data_array[:, col]
            valid_mask = ~np.isnan(col_data)
            g[valid_mask, col] = cmap(norm(col_data[valid_mask]))[:, :3]  # Apply colormap to valid data

    # Set up figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(g, interpolation="none")

    # Customize the plot appearance
    ax.set_aspect("auto")
    ax.grid(False)
    ax.xaxis.tick_top()

    # Set column labels
    ax.set_xticks(range(width))
    ax.set_xticklabels(df.columns, rotation=label_rotation, ha="left", fontsize=fontsize)

    # Set y-axis ticks to show row numbers
    ax.set_yticks([0, df.shape[0] - 1])
    ax.set_yticklabels([1, df.shape[0]], fontsize=fontsize, rotation=0)

    # Add vertical grid lines between columns
    for x in range(1, width):
        ax.axvline(x - 0.5, linestyle=":", color="gray")

    # Add missing rate (%) as a label above each column
    for i, rate in enumerate(missing_rates):
        ax.text(i, height, f"{rate:.1f}%", ha="center", va="top", fontsize=fontsize-2, color="black",rotation=label_rotation)

    plt.show()
    return ax






import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def heatmap(df, figsize=(20, 12), fontsize=14, label_rotation=45, 
            cmap='RdBu'):
    """
    Visualizes nullity correlation in the DataFrame using a heatmap.
    
    :param df: Pandas DataFrame to analyze missing values.
    :param figsize: Tuple, figure size (default: (20, 12)).
    :param fontsize: Font size for labels (default: 16).
    :param label_rotation: Rotation angle for x-axis labels (default: 45).
    :param cmap: Colormap for the heatmap (default: 'RdBu').
    
    :return: Matplotlib Axes object.
    """
    
    # Remove columns with no missing values (completely filled or empty)
    missing_vars = df.isnull().var(axis=0) > 0
    df = df.loc[:, missing_vars]
    
    if df.shape[1] == 0:
        raise ValueError("No missing values found in the dataset.")
    
    # Compute correlation matrix for missing values
    corr_mat = df.isnull().corr()

    # Create mask for upper triangle to avoid redundancy
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))

    # Create heatmap
    plt.figure(figsize=figsize)
    ax = sns.heatmap(corr_mat, mask=mask, cmap=cmap, vmin=-1, vmax=1, 
                     cbar=True, annot=True, fmt=".2f", annot_kws={"size": fontsize - 2})
    
    # Adjust labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=label_rotation, ha='right', fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fontsize)
    
    # Formatting annotations
    for text in ax.texts:
        val = float(text.get_text())
        text.set_text(f"{val:.2f}")
    
    plt.show()
    return ax
