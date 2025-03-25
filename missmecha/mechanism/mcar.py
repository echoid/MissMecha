
import numpy as np


def type_one(data, missing_rate=0.1, seed=1):
    data = data.astype(float)
    total_elements = data.size
    missing_elements = int(total_elements * missing_rate)

    data_with_missing = data.copy()
    np.random.seed(seed)
    mask_indices = np.random.choice(total_elements, missing_elements, replace=False)
    multi_indices = np.unravel_index(mask_indices, data.shape)
    data_with_missing[multi_indices] = np.nan

    return data_with_missing


def type_two(data, missing_rate=0.1, seed=1):
    data = data.astype(float)
    total_elements = data.size
    missing_elements = int(total_elements * missing_rate)

    data_with_missing = data.copy()
    np.random.seed(seed)
    mask_indices = np.random.choice(total_elements, missing_elements, replace=False)
    multi_indices = np.unravel_index(mask_indices, data.shape)
    data_with_missing[multi_indices] = np.nan

    return data_with_missing

# Map type numbers to functions
MCAR_TYPES = {
    1: type_one,
    2: type_two,
}