
import numpy as np
from ..util import verify_missing_rate


def verify_missing_rate(rate, var_name="missing_rate"):
    """
    Ensure the missing rate is a float between 0 and 1.
    """
    if not isinstance(rate, (float, int)):
        raise TypeError(f"{var_name} must be a float or int.")
    if not (0 <= rate <= 1):
        raise ValueError(f"{var_name} must be between 0 and 1 (got {rate}).")


def mcar_type1(data, missing_rate=0.1, seed=1):
    """
    MCAR Type 1 - Uniform missingness across entire dataset (entry-wise Bernoulli trial).

    Parameters
    ----------
    data : np.ndarray
        Input numeric array (n_samples x n_features).
    missing_rate : float
        Proportion of total values to be set as missing (between 0 and 1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    data_with_missing : np.ndarray
        Array with NaNs inserted uniformly at random.
    """
    verify_missing_rate(missing_rate)
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    rng = np.random.default_rng(seed)
    data = data.astype(float)
    mask = rng.uniform(0, 1, size=data.shape) < missing_rate
    data_with_missing = data.copy()
    data_with_missing[mask] = np.nan
    return data_with_missing


def mcar_type2(data, missing_rate=0.1, seed=1):
    """
    MCAR Type 2 - Classic fixed-size missing entries, sampled uniformly at random.

    Parameters
    ----------
    data : np.ndarray
        Input numeric array (n_samples x n_features).
    missing_rate : float
        Proportion of total values to be set as missing (between 0 and 1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    data_with_missing : np.ndarray
        Array with NaNs inserted uniformly at random.
    """
    verify_missing_rate(missing_rate)
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    rng = np.random.default_rng(seed)
    data = data.astype(float)
    total_elements = data.size
    missing_elements = int(round(total_elements * missing_rate))

    data_with_missing = data.copy()
    flat_indices = rng.choice(total_elements, size=missing_elements, replace=False)
    multi_indices = np.unravel_index(flat_indices, data.shape)
    data_with_missing[multi_indices] = np.nan
    return data_with_missing



def mcar_type3(data, missing_rate=0.1, seed=1):
    """
    MCAR Type 3- Evenly distributes missing values across all columns.

    Parameters
    ----------
    data : np.ndarray
        Input numeric array (n_samples x n_features).
    missing_rate : float
        Total proportion of missing values (between 0 and 1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    data_with_missing : np.ndarray
        Data with missing values uniformly distributed across columns.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if not (0 <= missing_rate <= 1):
        raise ValueError("missing_rate must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    data = data.astype(float)
    n, p = data.shape
    total_cells = n * p
    total_missing = int(round(total_cells * missing_rate))
    missing_per_col = total_missing // p

    data_with_missing = data.copy()
    for j in range(p):
        if missing_per_col > 0:
            rows = rng.choice(n, size=missing_per_col, replace=False)
            data_with_missing[rows, j] = np.nan

    return data_with_missing


# Map type numbers to functions
MCAR_TYPES = {
    1: mcar_type1,
    2: mcar_type2,
    3: mcar_type3,
}