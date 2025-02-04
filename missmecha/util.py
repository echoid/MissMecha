
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