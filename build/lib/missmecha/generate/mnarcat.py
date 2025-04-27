import numpy as np
import random

class MNARCatType1:
    """
    MNAR Mechanism for Categorical and Ordinal Features (Column-wise Variant)

    Introduces missingness into categorical or ordinal columns based on feature-specific
    criteria:
    - For numerical columns, values below a quantile threshold are masked.
    - For ordinal columns, top-ranked values are targeted.
    - For nominal columns, randomly chosen categories are partially masked.

    Parameters
    ----------
    q : float, default=0.2
        Quantile or proportion threshold used for masking.
    seed : int, default=1
        Random seed for reproducibility.
    """

    def __init__(self, q=0.2, seed=1):
        self.q = q
        self.seed = seed
        self.fitted = False

    def fit(self, X, col_info):
        """
        Fit method (placeholder for compatibility).

        Parameters
        ----------
        X : np.ndarray
            Input data array.
        col_info : dict
            Dictionary mapping column indices to their type.

        Returns
        -------
        self : MNARCategorical
            Returns self.
        """
        self.col_info = col_info
        self.fitted = True
        return self

    def transform(self, X):
        """
        Apply MNAR masking to categorical/ordinal/numerical columns.

        Parameters
        ----------
        X : np.ndarray
            Input data to transform.

        Returns
        -------
        X_missing : np.ndarray
            Transformed array with missing values injected.
        """
        if not self.fitted:
            raise RuntimeError("Call .fit() before .transform().")

        np.random.seed(self.seed)
        random.seed(self.seed)
        X = X.astype(float)
        X_missing = X.copy()
        n = X.shape[0]
        q = self.q * 100

        for col, col_type in self.col_info.items():
            col_idx = int(col)
            num_to_remove = int(n * self.q)

            if "numerical" in col_type:
                threshold = np.percentile(X[:, col_idx], q)
                X_missing[:, col_idx] = np.where(X[:, col_idx] < threshold, np.nan, X[:, col_idx])

            elif "ordinal" in col_type:
                ordinal_map = col_type.get('ordinal', {})
                max_val = max(ordinal_map.values())
                max_indices = np.where(X[:, col_idx] >= (max_val - 2))[0].tolist()
                remove_indices = self._sample_indices(max_indices, n, num_to_remove)
                X_missing[remove_indices, col_idx] = np.nan

            elif "nominal" in col_type:
                unique_vals = list(set(X[:, col_idx]))
                chosen_val = random.choice(unique_vals)
                chosen_indices = np.where(X[:, col_idx] == chosen_val)[0].tolist()
                remove_indices = self._sample_indices(chosen_indices, n, num_to_remove)
                X_missing[remove_indices, col_idx] = np.nan

        return X_missing

    def _sample_indices(self, primary_indices, total, k):
        """
        Helper function to sample indices up to size k, combining primary and fallback.
        """
        all_indices = set(range(total))
        secondary = list(all_indices - set(primary_indices))
        if len(primary_indices) >= k:
            return random.sample(primary_indices, k)
        else:
            fill = random.sample(secondary, k - len(primary_indices))
            return primary_indices + fill


MNARCAT_TYPES = {
    1: MNARCatType1,
    # 2: MARType2,
    # 3: MARType3,
    # 4: MARType4,
    # 5: MARType5,
    # 6: MARType6,
    # 7: MARType7,
    # 8: MARType8
}