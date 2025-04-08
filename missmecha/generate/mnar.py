import pandas as pd
import numpy as np
# Rewrite pick_coeffs and fit_intercepts in pure NumPy (no torch)
import numpy as np
from scipy.special import expit  # sigmoid
from scipy.optimize import bisect
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
            intercepts[j] = bisect(f, -50, 50)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = np.zeros(d_na)
        for j in range(d_na):
            f = lambda x: np.mean(expit(X @ coeffs[:, j] + x)) - p
            intercepts[j] = bisect(f, -50, 50)
    return intercepts

class MNARType1:
    def __init__(self, missing_rate=0.1, seed=1, up_percentile=0.5, obs_percentile=0.5, depend_on = None):
        self.missing_rate = missing_rate
        self.seed = seed
        self.up_percentile = up_percentile
        self.obs_percentile = obs_percentile
        self.fitted = False

        print("self.up_percentile",self.up_percentile)

    def fit(self, X, y=None):
        self.fitted = True

        rng = np.random.default_rng(self.seed)
        X = X.astype(float)

        def scale_data(x):
            min_vals = np.min(x, axis=0)
            max_vals = np.max(x, axis=0)
            return (x - min_vals) / (max_vals - min_vals + 1e-8)

        data = scale_data(X)
        n_rows, n_cols = data.shape
        #n_miss_cols = int(n_cols * self.missing_rate)
        self.miss_cols = rng.choice(n_cols, size=n_cols, replace=False)

        # Store thresholds for each missing column
        self.thresholds_miss = {}
        for col in self.miss_cols:
            self.thresholds_miss[col] = np.quantile(data[:, col], self.up_percentile)

        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Call .fit() before .transform().")

        rng = np.random.default_rng(self.seed)
        X = X.astype(float)

        def scale_data(x):
            min_vals = np.min(x, axis=0)
            max_vals = np.max(x, axis=0)
            return (x - min_vals) / (max_vals - min_vals + 1e-8)

        data = scale_data(X)
        mask = np.ones_like(data, dtype=bool)
        n_cols = data.shape[1]
        obs_cols = [i for i in range(n_cols) if i not in self.miss_cols]

        for miss_col in self.miss_cols:
            threshold_miss = self.thresholds_miss[miss_col]
            mask_condition_1 = data[:, miss_col] > threshold_miss

            if obs_cols:
                obs_data = data[mask_condition_1][:, obs_cols]
                if obs_data.size > 0:
                    threshold_obs = np.quantile(obs_data, self.obs_percentile)
                    mask_condition_2 = data[:, miss_col] > threshold_obs
                    merged_mask = np.logical_or(mask_condition_1, mask_condition_2)
                else:
                    merged_mask = mask_condition_1
            else:
                merged_mask = mask_condition_1

            mask[:, miss_col] = ~merged_mask

        data_with_missing = X.copy()
        data_with_missing[~mask] = np.nan
        return data_with_missing
import numpy as np
from scipy.special import expit
from scipy.optimize import bisect

class MNARType2:
    def __init__(self, missing_rate=0.1, para=0.3, exclude_inputs=True, seed=1, depend_on = None):
        self.missing_rate = missing_rate
        self.p_params = para
        self.exclude_inputs = exclude_inputs
        self.seed = seed
        self.fitted = False

    def fit(self, X, y=None):
        np.random.seed(self.seed)
        X = X.copy()
        n, d = X.shape
        self.d = d

        self.d_params = max(int(self.p_params * d), 1) if self.exclude_inputs else d
        self.d_na = d - self.d_params if self.exclude_inputs else d

        self.idxs_params = np.random.choice(d, self.d_params, replace=False) if self.exclude_inputs else np.arange(d)
        self.idxs_nas = np.array([i for i in range(d) if i not in self.idxs_params]) if self.exclude_inputs else np.arange(d)

        self.coeffs = pick_coeffs_numpy(X, self.idxs_params, self.idxs_nas)
        self.intercepts = fit_intercepts_numpy(X[:, self.idxs_params], self.coeffs, self.missing_rate)
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Call .fit() before .transform().")

        X = X.copy()
        n = X.shape[0]
        ps = expit(X[:, self.idxs_params] @ self.coeffs + self.intercepts)

        mask = np.zeros((n, self.d), dtype=bool)
        mask[:, self.idxs_nas] = np.random.rand(n, self.d_na) < ps

        if self.exclude_inputs:
            mask[:, self.idxs_params] = np.random.rand(n, self.d_params) < self.missing_rate

        X_missing = X.copy()
        X_missing[mask] = np.nan
        return X_missing
class MNARType3:
    def __init__(self, missing_rate=0.1, seed=1, depend_on = None):
        self.missing_rate = missing_rate
        self.seed = seed
        self.fitted = False

    def fit(self, X, y=None):
        self.coeffs = pick_coeffs_numpy(X, self_mask=True)
        self.intercepts = fit_intercepts_numpy(X, self.coeffs, self.missing_rate, self_mask=True)
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Call .fit() before .transform().")
        ps = expit(X * self.coeffs + self.intercepts)
        mask = np.random.rand(*X.shape) < ps
        X_missing = X.copy()
        X_missing[mask] = np.nan
        return X_missing
    
class MNARType4:
    def __init__(self, missing_rate=0.1, q=0.25, p=0.5, cut="both", seed=1, depend_on = None):
        self.missing_rate = missing_rate
        self.q = q
        self.p_params = p
        self.cut = cut
        self.seed = seed
        self.fitted = False

    def fit(self, X, y=None):
        np.random.seed(self.seed)
        n, d = X.shape
        #self.X_shape = (n, d)

        self.fitted = True

        idxs_na = np.random.choice(d, max(int(self.p_params * d), 1), replace=False)
        
        X = X.copy()
        if self.cut == "upper":
            self.quants = np.quantile(X[:, idxs_na], 1 - self.q, axis=0)
        elif self.cut == "lower":
            self.quants = np.quantile(X[:, idxs_na], self.q, axis=0)
        elif self.cut == "both":
            self.u_quants = np.quantile(X[:, idxs_na], 1 - self.q, axis=0)
            self.l_quants = np.quantile(X[:, idxs_na], self.q, axis=0)
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Call .fit() before .transform().")
        
        
        n, d = X.shape
        mask = np.zeros((n, d), dtype=bool)
        self.idxs_na = np.random.choice(d, max(int(self.p_params * d), 1), replace=False)
        X = X.copy()
        if self.cut == "upper":
            m = X[:, self.idxs_na] >= self.quants
        elif self.cut == "lower":
            quants = np.quantile(X[:, self.idxs_na], self.q, axis=0)
            m = X[:, self.idxs_na] <= self.quants
        elif self.cut == "both":
            self.u_quants = np.quantile(X[:, self.idxs_na], 1 - self.q, axis=0)
            self.l_quants = np.quantile(X[:, self.idxs_na], self.q, axis=0)
            m = (X[:, self.idxs_na] <= self.l_quants) | (X[:, self.idxs_na] >= self.u_quants)

        ber = np.random.rand(n, len(self.idxs_na))
        mask[:, self.idxs_na] = (ber < self.missing_rate) & m

        X_missing = X.copy()
        X_missing[mask] = np.nan
        return X_missing
import numpy as np
from scipy.special import expit as sigmoid
from scipy import optimize

class MNARType5:
    def __init__(self, missing_rate=0.1, seed=1, depend_on = None):
        self.missing_rate = missing_rate
        self.seed = seed
        self.fitted = False

    def _pick_coeffs(self, X):
        rng = np.random.default_rng(self.seed)
        d = X.shape[1]
        coeffs = rng.normal(size=d)
        Wx = X * coeffs
        stds = Wx.std(axis=0)
        stds[stds == 0] = 1  # Avoid divide-by-zero
        coeffs /= stds
        return coeffs

    def _fit_intercepts(self, X, coeffs):
        d = X.shape[1]
        intercepts = np.zeros(d)

        for j in range(d):
            def f(x):
                return sigmoid(X[:, j] * coeffs[j] + x).mean() - self.missing_rate

            try:
                intercepts[j] = optimize.bisect(f, -1000, 1000)
            except ValueError:
                intercepts[j] = 0  # fallback if bisection fails
        return intercepts

    def fit(self, X, y=None):
        X = X.astype(float)
        self.coeffs = self._pick_coeffs(X)
        self.intercepts = self._fit_intercepts(X, self.coeffs)
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Call .fit() before .transform().")

        X = X.astype(float)
        logits = X * self.coeffs + self.intercepts
        probs = sigmoid(logits)
        rng = np.random.default_rng(self.seed)
        mask = rng.random(size=X.shape) < probs

        X_missing = X.copy()
        X_missing[mask] = np.nan
        return X_missing


class MNARType6:
    def __init__(self, missing_rate=0.1, seed=1, depend_on = None):
        self.missing_rate = missing_rate
        self.seed = seed
        self.fitted = False

    def fit(self, X, y=None):
        """
        Store the threshold (percentile cutoff) for each column based on training data.
        """
        self.fitted = True
        rng = np.random.default_rng(self.seed)

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        self.n_features = X.shape[1]

        self.cutoffs = []
        for col in range(self.n_features):
            cutoff = np.percentile(X[:, col], self.missing_rate * 100)
            self.cutoffs.append(cutoff)

        return self

    def transform(self, X):
        """
        Apply the learned cutoffs to determine missing values per column.
        """
        if not self.fitted:
            raise RuntimeError("Call .fit() before .transform().")

        if isinstance(X, pd.DataFrame):
            return self._transform_df(X)
        else:
            return self._transform_array(X)

    def _transform_array(self, X):
        X = X.astype(float)
        X_missing = X.copy()
        for col in range(self.n_features):
            X_missing[:, col] = np.where(X[:, col] < self.cutoffs[col], np.nan, X[:, col])
        return X_missing

    def _transform_df(self, df):
        X_missing = df.copy().astype(float)
        for i, col_name in enumerate(df.columns):
            X_missing[col_name] = np.where(df[col_name] < self.cutoffs[i], np.nan, df[col_name])
        return X_missing


MNAR_TYPES = {
    1: MNARType1,
    2: MNARType2,
    3: MNARType3,
    4: MNARType4,
    5: MNARType5,
    6: MNARType6

}


# def make_mnar_columnwise(data, col_info, q, random_seed=1):
#     np.random.seed(random_seed)
#     random.seed(random_seed)
#     q = q * 100
#     data_mnar = data.astype(float)

#     missing_rates = {}

#     for col, col_type in col_info.items():
#         col_idx = int(col)  # Assuming the keys in `col_info` correspond to column indices
#         num_to_remove = int(len(data_mnar) * q / 100)
#         if "numerical" in col_type:
#             # Calculate the percentile value for the numerical column
#             threshold = np.percentile(data_mnar[:, col_idx], q)
#             # Replace values less than the threshold with np.nan
#             data_mnar[:, col_idx] = np.where(data_mnar[:, col_idx] < threshold, np.nan, data_mnar[:, col_idx])

#             # Calculate the missing rate for this column
#             missing_rate = np.mean(np.isnan(data_mnar[:, col_idx])) * 100
#             missing_rates[col_idx] = missing_rate
#             #print("numerical" ,missing_rate)

#         elif "ordinal" in col_type:
#             # Use the ordinal mapping from JSON to find the top two largest ordinal values
#             ordinal_map = col_type['ordinal']
#             max_value = max(ordinal_map.values())

#             # Find the indices where the values in the column are greater than or equal to max_value - 1
#             max_indices = np.where(data_mnar[:, col_idx] >= (max_value - 2))[0].tolist()

#             # Find the rest of the indices (those not in max_indices)
#             all_indices = set(range(data_mnar.shape[0]))
#             other_indices = list(all_indices - set(max_indices))

#             # Determine which indices to remove based on the number to remove
#             if len(max_indices) >= num_to_remove:
#                 remove_indices = random.sample(max_indices, num_to_remove)
#             else:
#                 # If there are not enough max_indices, take all max_indices and supplement with random others
#                 remove_indices = max_indices
#                 random_indices = random.sample(other_indices, num_to_remove - len(remove_indices))
#                 #remove_indices = remove_indices + random_indices

#             data_mnar[remove_indices, col_idx] = np.nan

#             # Calculate the missing rate for this column
#             missing_rate = np.mean(np.isnan(data_mnar[:, col_idx])) * 100
#             missing_rates[col_idx] = missing_rate
#             #print("ordinal" ,missing_rate)

#         elif "nominal" in col_type:
#             # Nominal data: Randomly choose one category and make a portion of the data missing
#             unique_vals = list(set(data_mnar[:, col_idx]))
#             chosen_val = random.choice(unique_vals)

#             # Get indices of the chosen category
#             chosen_indices = np.where(data_mnar[:, col_idx] == chosen_val )[0].tolist()


#             # Find the rest of the indices (those not in max_indices)
#             all_indices = set(range(data_mnar.shape[0]))
#             other_indices = list(all_indices - set(chosen_indices))

#             # Determine which indices to remove based on the number to remove
#             if len(chosen_indices) >= num_to_remove:
#                 remove_indices = random.sample(chosen_indices, num_to_remove)
#             else:
#                 # If there are not enough max_indices, take all max_indices and supplement with random others
#                 remove_indices = chosen_indices
#                 random_indices = random.sample(other_indices, num_to_remove - len(remove_indices))
#                 remove_indices = remove_indices + random_indices


#             data_mnar[remove_indices, col_idx] = np.nan

#             # Calculate the missing rate for this column
#             missing_rate = np.mean(np.isnan(data_mnar[:, col_idx])) * 100
#             #print("nominal",missing_rate)
#             missing_rates[col_idx] = missing_rate

#     return data_mnar


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