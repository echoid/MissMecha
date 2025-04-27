
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import warnings

import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
warnings.filterwarnings("ignore", message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.")

INIT_ARG_HINTS = {
    "MNARType1": "Expected parameter fields: 'up_percentile', 'obs_percentile'.",
    "MNARType2": "Expected parameter fields: 'para', 'exclude_inputs'.",
    "MNARType4": "Expected parameter fields: 'q', 'p', 'cut'.",
    "MARType1": "Expected parameter fields: 'para'.",
    "MARType2": "No additional parameters required.",
    "MARType3": "No additional parameters required.",
    "MARType4": "No additional parameters required.",
    "MARType5": "No additional parameters required.",
    "MARType6": "No additional parameters required.",
    "MARType7": "No additional parameters required.",
    "MARType8": "No additional parameters required.",

    "MNARType3": "No additional parameters required.",
    "MNARType5": "No additional parameters required.",
    "MNARType6": "No additional parameters required.",

}
import inspect
import warnings

def safe_init(cls, kwargs):
    # 获取 __init__ 支持的参数名
    sig = inspect.signature(cls.__init__)
    accepted_args = set(sig.parameters.keys()) - {"self"}

    # 检查多余的参数
    extra_keys = [k for k in kwargs if k not in accepted_args]
    if extra_keys:
        warnings.warn(
            f"[{cls.__name__}] received unrecognized parameter(s): {extra_keys}. "
            f"They will be ignored.",
            UserWarning
        )

    # 正常初始化（多余的 key 会在下一步抛错）
    try:
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_args}
        return cls(**filtered_kwargs)
    except TypeError as e:
        raise TypeError(
            f"[{cls.__name__}] failed to initialize with parameters: {list(kwargs.keys())}\n"
            f"Original error: {e}\n"
            f"Hint: Please check the 'parameter' field and match expected init arguments."
        )


def verify_missing_rate(rate, var_name="missing_rate"):
    """
    Verify that the missing rate is between 0 and 1 (inclusive).

    Parameters
    ----------
    rate : float
        The missing rate to check.
    var_name : str, optional
        Variable name to show in the error message.

    Raises
    ------
    ValueError
        If the rate is not in the range [0, 1].
    """
    if not isinstance(rate, (float, int)):
        raise TypeError(f"{var_name} must be a float or int.")
    if not (0 <= rate <= 1):
        raise ValueError(f"{var_name} must be between 0 and 1 (got {rate}).")


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




