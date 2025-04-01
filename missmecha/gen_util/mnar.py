import pandas as pd
import numpy as np
# Rewrite pick_coeffs and fit_intercepts in pure NumPy (no torch)
import numpy as np
from scipy.special import expit  # sigmoid
from scipy.optimize import bisect

def mnar_type1(data, missing_rate=0.1, seed=1, up_percentile=0.5, obs_percentile=0.5):
    """
    MNAR Type 1 — Diffuse MNAR masking based on value thresholds.
    Missingness is determined by the variable's own values and a subset of
    observed variables (approximation of self- and cross-influenced MNAR).

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Input numeric data.

    missing_rate : float, default=0.1
        Overall missing fraction (used to control proportion of masked columns).

    seed : int, default=1
        Random seed for reproducibility.

    up_percentile : float, default=0.5
        Upper quantile used for thresholding missing column.

    obs_percentile : float, default=0.5
        Threshold quantile for observed variable filtering.

    Returns
    -------
    data_with_missing : np.ndarray
        Copy of data with MNAR-injected missing values.
    """
    rng = np.random.default_rng(seed)

    def scale_data(x):
        min_vals = np.min(x, axis=0)
        max_vals = np.max(x, axis=0)
        scaled = (x - min_vals) / (max_vals - min_vals + 1e-8)
        return scaled

    data = scale_data(data)
    mask = np.ones(data.shape, dtype=bool)
    n_rows, n_cols = data.shape

    n_miss_cols = int(n_cols * missing_rate)
    miss_cols = rng.choice(n_cols, size=n_miss_cols, replace=False)
    obs_cols = [i for i in range(n_cols) if i not in miss_cols]

    for miss_col in miss_cols:
        threshold_miss = np.quantile(data[:, miss_col], up_percentile)
        mask_condition_1 = data[:, miss_col] > threshold_miss

        if len(obs_cols) > 0:
            obs_data = data[mask_condition_1][:, obs_cols]
            if obs_data.size > 0:
                threshold_obs = np.quantile(obs_data, obs_percentile)
                mask_condition_2 = data[:, miss_col] > threshold_obs
                merged_mask = np.logical_or(mask_condition_1, mask_condition_2)
            else:
                merged_mask = mask_condition_1
        else:
            merged_mask = mask_condition_1

        mask[:, miss_col] = ~merged_mask

    data_with_missing = data.copy()
    data_with_missing[~mask] = np.nan
    return data_with_missing



def pick_coeffs_numpy(X, idxs_obs=None, idxs_nas=None, self_mask=False):
    n, d = X.shape
    if self_mask:
        coeffs = np.random.randn(d)
        Wx = X * coeffs
        coeffs /= np.std(Wx, axis=0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = np.random.randn(d_obs, d_na)
        Wx = X[:, idxs_obs] @ coeffs
        coeffs /= np.std(Wx, axis=0, keepdims=True)
    return coeffs

def fit_intercepts_numpy(X, coeffs, p, self_mask=False):
    if self_mask:
        d = len(coeffs)
        intercepts = np.zeros(d)
        for j in range(d):
            f = lambda x: np.mean(expit(X * coeffs[j] + x)) - p
            intercepts[j] = bisect(f, -1000, 1000)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = np.zeros(d_na)
        for j in range(d_na):
            f = lambda x: np.mean(expit(X @ coeffs[:, j] + x)) - p
            intercepts[j] = bisect(f, -1000, 1000)
    return intercepts



# Rewriting MNAR type 2, 3, 4 in pure numpy using the converted utility functions
def mnar_type2(data, missing_rate=0.1, p_params=0.3, exclude_inputs=True, seed=1):
    np.random.seed(seed)
    X = data.copy()
    n, d = X.shape

    mask = np.zeros((n, d), dtype=bool)

    d_params = max(int(p_params * d), 1) if exclude_inputs else d
    d_na = d - d_params if exclude_inputs else d

    idxs_params = np.random.choice(d, d_params, replace=False) if exclude_inputs else np.arange(d)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_params]) if exclude_inputs else np.arange(d)

    coeffs = pick_coeffs_numpy(X, idxs_params, idxs_nas)
    intercepts = fit_intercepts_numpy(X[:, idxs_params], coeffs, missing_rate)

    ps = expit(X[:, idxs_params] @ coeffs + intercepts)
    mask[:, idxs_nas] = np.random.rand(n, d_na) < ps

    if exclude_inputs:
        mask[:, idxs_params] = np.random.rand(n, d_params) < missing_rate

    data_with_missing = X.copy()
    data_with_missing[mask] = np.nan
    return data_with_missing

def mnar_type3(data, missing_rate=0.1, seed=1):
    np.random.seed(seed)
    X = data.copy()
    n, d = X.shape

    coeffs = pick_coeffs_numpy(X, self_mask=True)
    intercepts = fit_intercepts_numpy(X, coeffs, missing_rate, self_mask=True)

    ps = expit(X * coeffs + intercepts)
    mask = np.random.rand(n, d) < ps

    data_with_missing = X.copy()
    data_with_missing[mask] = np.nan
    return data_with_missing

def mnar_type4(data, missing_rate=0.1, q=0.25, p_params=0.5, cut="both",seed=1):
    np.random.seed(seed)
    X = data.copy()
    n, d = X.shape

    mask = np.zeros((n, d), dtype=bool)
    d_na = max(int(p_params * d), 1)
    idxs_na = np.random.choice(d, d_na, replace=False)

    def compute_quantile(x, q_level):
        return np.quantile(x, q_level, axis=0)

    if cut == "upper":
        quants = compute_quantile(X[:, idxs_na], 1 - q)
        m = X[:, idxs_na] >= quants
    elif cut == "lower":
        quants = compute_quantile(X[:, idxs_na], q)
        m = X[:, idxs_na] <= quants
    elif cut == "both":
        u_quants = compute_quantile(X[:, idxs_na], 1 - q)
        l_quants = compute_quantile(X[:, idxs_na], q)
        m = (X[:, idxs_na] <= l_quants) | (X[:, idxs_na] >= u_quants)

    ber = np.random.rand(n, d_na)
    mask[:, idxs_na] = (ber < missing_rate) & m

    data_with_missing = X.copy()
    data_with_missing[mask] = np.nan
    return data_with_missing

import numpy as np
from scipy import optimize
from scipy.special import expit as sigmoid

def pick_coeffs_self_mask(X, seed=1):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    coeffs = rng.normal(size=d)
    Wx = X * coeffs
    stds = Wx.std(axis=0)
    stds[stds == 0] = 1  # Avoid divide-by-zero
    coeffs /= stds
    return coeffs

def fit_intercepts_self_mask(X, coeffs, p):
    d = X.shape[1]
    intercepts = np.zeros(d)

    for j in range(d):
        def f(x):
            return sigmoid(X[:, j] * coeffs[j] + x).mean() - p

        try:
            intercepts[j] = optimize.bisect(f, -50, 50)
        except ValueError:
            intercepts[j] = 0  # fallback if bisection fails

    return intercepts

def mnar_type5(data, missing_rate=0.1, seed=1):
    """
    MNAR Type 5 - Global self-masking based on logistic model (each variable controls its own missingness).

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data.
    missing_rate : float
        Target missing rate per variable (0 < rate < 1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    data_with_missing : np.ndarray
        Data with missing values (NaNs inserted).
    """
    if isinstance(data, np.ndarray):
        X = data.copy().astype(float)
    else:
        X = data.to_numpy().astype(float)

    n, d = X.shape
    rng = np.random.default_rng(seed)

    coeffs = pick_coeffs_self_mask(X, seed)
    intercepts = fit_intercepts_self_mask(X, coeffs, missing_rate)

    logits = X * coeffs + intercepts
    probs = sigmoid(logits)
    rand_mask = rng.random(size=(n, d)) < probs

    data_with_missing = X.copy()
    data_with_missing[rand_mask] = np.nan

    return data_with_missing
def mnar_type6(data, missing_rate=0.1,seed = 1):
    """
    MNAR Type 6 - Deterministic self-masking based on lower percentile threshold.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data.
    threshold : float, optional
        Quantile threshold (0 < threshold < 1). Values lower than this will be masked.
        For example, 0.25 = mask values below the 25th percentile.

    Returns
    -------
    data_with_missing : np.ndarray
        Data with NaNs inserted based on value thresholds.
    """
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    threshold = missing_rate
    data = data.copy()
    for col in range(data.shape[1]):
        cutoff = np.percentile(data[:, col], threshold * 100)
        data[:, col] = np.where(data[:, col] < cutoff, np.nan, data[:, col])

    return data
# def mnar_type5(data, missing_rate=0.1, label=None, seed=1):
#     """
#     MNAR Type 5 - Self-masking on most correlated feature with label (Twala09).
#     The lowest values of the most label-correlated feature are masked.

#     Parameters
#     ----------
#     data : np.ndarray or pd.DataFrame
#         Input data matrix.
#     missing_rate : float
#         Percentage (0–1) of missing values to insert in the selected column.
#     label : array-like, optional
#         Target variable used to determine the most correlated feature.
#         If None, the last column of data will be used as label.
#     seed : int
#         Random seed.

#     Returns
#     -------
#     data_with_missing : np.ndarray
#         Data with NaNs inserted.
#     """
#     rng = np.random.default_rng(seed)

#     if isinstance(data, pd.DataFrame):
#         data_np = data.to_numpy()
#     else:
#         data_np = data.copy()

#     n, p = data_np.shape
#     N = int(round(n * missing_rate))

#     if label is None:
#         if p < 2:
#             raise ValueError("Data must contain at least 2 columns to use the last column as label.")
#         label = data_np[:, -1]
#         data_np = data_np[:, :-1]  # exclude label from correlation

#     # Correlation with label
#     correlations = [
#         abs(np.corrcoef(data_np[:, i], label)[0, 1])
#         if not np.isnan(data_np[:, i]).all() else 0
#         for i in range(data_np.shape[1])
#     ]
#     idx_xs = int(np.argmax(correlations))

#     # Mask lowest N values
#     sorted_indices = np.argsort(data_np[:, idx_xs])
#     missing_indices = sorted_indices[:N]

#     data_with_missing = data_np.copy()
#     data_with_missing[missing_indices, idx_xs] = np.nan

#     return data_with_missing



# def mnar_type6(data, missing_rate=0.1, column=None, seed=1):
#     """
#     MNAR Type 6 - Mask highest values in a selected or random column (Xia17).

#     Parameters
#     ----------
#     data : np.ndarray or pd.DataFrame
#         Input data.
#     missing_rate : float
#         Missing rate as a float between 0 and 1.
#     column : int or None
#         If provided, mask values in this column; otherwise choose randomly.
#     seed : int
#         Random seed.

#     Returns
#     -------
#     data_with_missing : np.ndarray
#         Data with inserted NaNs.
#     """
#     rng = np.random.default_rng(seed)

#     if isinstance(data, pd.DataFrame):
#         data_np = data.to_numpy()
#     else:
#         data_np = data.copy()

#     n, p = data_np.shape
#     N = int(round(n * missing_rate))

#     idx_xs = column if column is not None else rng.integers(0, p)

#     # Highest N values → NaN
#     sorted_indices = np.argsort(data_np[:, idx_xs])
#     missing_indices = sorted_indices[-N:]

#     data_with_missing = data_np.copy()
#     data_with_missing[missing_indices, idx_xs] = np.nan

#     return data_with_missing


# Map type numbers to functions
MNAR_TYPES = {
    1: mnar_type1,
    2: mnar_type2,
    3: mnar_type3,
    4: mnar_type4,
    5: mnar_type5,
    6: mnar_type6,

}








def make_mnar_columnwise(data, col_info, q, random_seed=1):
    np.random.seed(random_seed)
    random.seed(random_seed)
    q = q * 100
    data_mnar = data.astype(float)

    missing_rates = {}

    for col, col_type in col_info.items():
        col_idx = int(col)  # Assuming the keys in `col_info` correspond to column indices
        num_to_remove = int(len(data_mnar) * q / 100)
        if "numerical" in col_type:
            # Calculate the percentile value for the numerical column
            threshold = np.percentile(data_mnar[:, col_idx], q)
            # Replace values less than the threshold with np.nan
            data_mnar[:, col_idx] = np.where(data_mnar[:, col_idx] < threshold, np.nan, data_mnar[:, col_idx])

            # Calculate the missing rate for this column
            missing_rate = np.mean(np.isnan(data_mnar[:, col_idx])) * 100
            missing_rates[col_idx] = missing_rate
            #print("numerical" ,missing_rate)

        elif "ordinal" in col_type:
            # Use the ordinal mapping from JSON to find the top two largest ordinal values
            ordinal_map = col_type['ordinal']
            max_value = max(ordinal_map.values())

            # Find the indices where the values in the column are greater than or equal to max_value - 1
            max_indices = np.where(data_mnar[:, col_idx] >= (max_value - 2))[0].tolist()

            # Find the rest of the indices (those not in max_indices)
            all_indices = set(range(data_mnar.shape[0]))
            other_indices = list(all_indices - set(max_indices))

            # Determine which indices to remove based on the number to remove
            if len(max_indices) >= num_to_remove:
                remove_indices = random.sample(max_indices, num_to_remove)
            else:
                # If there are not enough max_indices, take all max_indices and supplement with random others
                remove_indices = max_indices
                random_indices = random.sample(other_indices, num_to_remove - len(remove_indices))
                #remove_indices = remove_indices + random_indices

            data_mnar[remove_indices, col_idx] = np.nan

            # Calculate the missing rate for this column
            missing_rate = np.mean(np.isnan(data_mnar[:, col_idx])) * 100
            missing_rates[col_idx] = missing_rate
            #print("ordinal" ,missing_rate)

        elif "nominal" in col_type:
            # Nominal data: Randomly choose one category and make a portion of the data missing
            unique_vals = list(set(data_mnar[:, col_idx]))
            chosen_val = random.choice(unique_vals)

            # Get indices of the chosen category
            chosen_indices = np.where(data_mnar[:, col_idx] == chosen_val )[0].tolist()


            # Find the rest of the indices (those not in max_indices)
            all_indices = set(range(data_mnar.shape[0]))
            other_indices = list(all_indices - set(chosen_indices))

            # Determine which indices to remove based on the number to remove
            if len(chosen_indices) >= num_to_remove:
                remove_indices = random.sample(chosen_indices, num_to_remove)
            else:
                # If there are not enough max_indices, take all max_indices and supplement with random others
                remove_indices = chosen_indices
                random_indices = random.sample(other_indices, num_to_remove - len(remove_indices))
                remove_indices = remove_indices + random_indices


            data_mnar[remove_indices, col_idx] = np.nan

            # Calculate the missing rate for this column
            missing_rate = np.mean(np.isnan(data_mnar[:, col_idx])) * 100
            #print("nominal",missing_rate)
            missing_rates[col_idx] = missing_rate

    return data_mnar