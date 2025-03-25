
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import warnings

import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
warnings.filterwarnings("ignore", message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.")

def type_convert(df):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 
    """
    detects and converts:
    - Datetime columns with proper format detection.
    - Categorical columns to numerical codes (ignoring NaNs).
    - Numeric columns to float (while preserving NaNs).
    :param df: Pandas DataFrame
    :return: Converted Numpy array
    """
    for col in df.columns:
        try:     
            df[col] = df[col].to_numpy(dtype=float)
        except:
            try:             
                df[col] = pd.to_datetime(df[col])  # Fallback
            except:
                df[col].to_numpy(dtype=object)
                df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan)
    df = df.to_numpy(dtype=float)
    return df


import numpy as np

def apply_missing_rate(data, missing_rate):
    # Flatten the data to simplify the process
    flat_data = data.flatten()

    # Count the existing missing values
    total_elements = flat_data.size
    current_missing_count = np.sum(np.isnan(flat_data))

    # Calculate the target number of missing values
    target_missing_count = int(missing_rate * total_elements)

    # Calculate how many more values need to be removed
    additional_missing_count = target_missing_count - current_missing_count

    if additional_missing_count <= 0:
        # If the current missing rate is already higher than or equal to the target, return the original data
        return data

    # Identify indices that are not already missing
    available_indices = np.where(~np.isnan(flat_data))[0]

    # Randomly select indices to remove additional data
    indices_to_remove = np.random.choice(available_indices, additional_missing_count, replace=False)

    # Set the selected indices to np.nan to represent missing data
    flat_data[indices_to_remove] = np.nan

    # Reshape the flat data back to the original shape
    return flat_data.reshape(data.shape)