import numpy as np
import pandas as pd
from .mcar import MCAR_TYPES



# Simulate importing generate from missmecha.mechanism using previously defined code
def generate_missing(df, missing_type="mcar", info=None, seed=1):

    assert missing_type == "mcar", "Only 'mcar' is supported in this version."

    rate = info.get("rate", 0.1) if info else 0.1
    type_id = info.get("type", 1) if info else 1

    # Look up the generation function
    assert type_id in MCAR_TYPES, f"MCAR type {type_id} is not defined."
    generator_fn = MCAR_TYPES[type_id]

    data_np = df.to_numpy().astype(float)
    data_with_nan = generator_fn(data_np, missing_rate=rate, seed=seed)

    # Build masks
    mask_int = ~np.isnan(data_with_nan)
    mask_int = mask_int.astype(int)
    mask_bool = ~np.isnan(data_with_nan)

    # Back to DataFrame
    data_nan = pd.DataFrame(data_with_nan, index=df.index, columns=df.columns)
    mask_int_df = pd.DataFrame(mask_int, index=df.index, columns=df.columns)
    mask_bool_df = pd.DataFrame(mask_bool, index=df.index, columns=df.columns)

    return data_nan, mask_int_df, mask_bool_df



