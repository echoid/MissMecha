import numpy as np
import pandas as pd
from .gen_util.mcar import MCAR_TYPES
from .gen_util.mar import MAR_TYPES
# from .gen_util.mnar import MNAR_TYPES
import numpy as np
import pandas as pd


def generate_missing(data, missing_type="mcar", type=1, missing_rate=0.1, info=None, seed=1):
    """
    Generate missing values in a dataset using a specified missing mechanism.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Input dataset.
    missing_type : str, optional
        One of 'mcar', 'mar', 'mnar'. Only used if `info` is None.
    type : int, optional
        Mechanism variant ID. Only used if `info` is None.
    missing_rate : float, optional
        Overall missing rate. Only used if `info` is None.
    info : dict, optional
        Column-wise control dictionary. If provided, will override other mechanism settings.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    result : dict
        {
            "data": np.ndarray or pd.DataFrame with NaNs,
            "mask_int": same type as input (0 = missing, 1 = present),
            "mask_bool": same type as input (False = missing, True = present)
        }
    """

    is_input_df = isinstance(data, pd.DataFrame)

    if is_input_df:
        col_names = data.columns
        index = data.index
        data_np = data.to_numpy().astype(float)
    elif isinstance(data, np.ndarray):
        data_np = data.astype(float)
        col_names = [f"col{i}" for i in range(data.shape[1])]
        index = np.arange(data.shape[0])
    else:
        raise TypeError("Input must be a pandas DataFrame or a NumPy array.")

    # --- Unified mechanism mode ---
    if info is None:
        assert missing_type in ["mcar", "mar", "mnar"], "Invalid missing_type."
        if missing_type == "mcar":
            assert type in MCAR_TYPES, f"MCAR type {type} not registered."
            generator_fn = MCAR_TYPES[type]
        elif missing_type == "mar":
            assert type in MAR_TYPES, f"MAR type {type} not registered."
            generator_fn = MAR_TYPES[type]
        elif missing_type == "mnar":
            assert type in MNAR_TYPES, f"MNAR type {type} not registered."
            generator_fn = MNAR_TYPES[type]

        data_with_nan = generator_fn(
            data_np, missing_rate=missing_rate, seed=seed)

    else:
        raise NotImplementedError("Column-wise missing generation (info-based) not yet implemented.")

    mask_bool = ~np.isnan(data_with_nan)
    mask_int = mask_bool.astype(int)

    if is_input_df:
        data_result = pd.DataFrame(data_with_nan, columns=col_names, index=index)
        mask_int_result = pd.DataFrame(mask_int, columns=col_names, index=index)
        mask_bool_result = pd.DataFrame(mask_bool, columns=col_names, index=index)
    else:
        data_result = data_with_nan
        mask_int_result = mask_int
        mask_bool_result = mask_bool

    return {
        "data": data_result,
        "mask_int": mask_int_result,
        "mask_bool": mask_bool_result,
    }
