import numpy as np


class MCARType1:
    """
    MCAR Mechanism - Type 1 (Uniform Independent Masking)

    Randomly masks entries with a uniform probability across the entire dataset.
    This mechanism applies a global missing rate independently at each cell.

    Parameters
    ----------
    missing_rate : float, default=0.1
        The proportion of values to randomly set as missing (0 ≤ missing_rate ≤ 1).
    seed : int, default=1
        Random seed for reproducibility.
    """
    def __init__(self, missing_rate=0.1, seed=1):
        self.missing_rate = missing_rate
        self.seed = seed
        self.fitted = False  # 一般MCAR不需要 fit, 但我们保持统一接口

    def fit(self, X, y=None):
        """
        Placeholder fit method for interface compatibility.

        MCARType1 does not require fitting, but this method sets a flag for internal consistency.
        """
        # MCAR 不依赖 X 或 y，fit 只是设置标志位
        self.fitted = True
        return self

    def transform(self, X):
        """
        Apply MCARType1 transformation to introduce missingness.

        Each entry in the dataset has an independent probability of being set to NaN.

        Parameters
        ----------
        X : np.ndarray
            Input array to apply missingness (converted to float).

        Returns
        -------
        X_missing : np.ndarray
            The same array with missing values inserted.
        """
        if not self.fitted:
            raise RuntimeError("MCARType1 must be fit before calling transform.")

        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        mask = rng.uniform(0, 1, size=X.shape) < self.missing_rate
        X_missing = X.copy()
        X_missing[mask] = np.nan
        return X_missing

class MCARType2:
    """
    MCAR Mechanism - Type 2 (Random Cell Selection)

    Randomly selects a fixed number of entries based on the overall missing rate,
    and masks exactly that number of cells across the dataset.

    Parameters
    ----------
    missing_rate : float, default=0.1
        The proportion of values to randomly set as missing (0 ≤ missing_rate ≤ 1).
    seed : int, default=1
        Random seed for reproducibility.
    """
    def __init__(self, missing_rate=0.1, seed=1):
        self.missing_rate = missing_rate
        self.seed = seed
        self.fitted = False

    def fit(self, X, y=None):
        """
        Placeholder fit method for interface compatibility.

        MCARType2 does not require fitting, but this method sets a flag for internal consistency.
        """
        # MCAR 不依赖 X/y，fit 仅作为流程接口
        self.fitted = True
        return self

    def transform(self, X):
        """
        Apply MCARType2 transformation to introduce missingness.

        Randomly masks a fixed number of values across the entire array,
        based on the global missing rate.

        Parameters
        ----------
        X : np.ndarray
            Input array to apply missingness (converted to float).

        Returns
        -------
        X_missing : np.ndarray
            Array with missing entries randomly inserted.
        """
        if not self.fitted:
            raise RuntimeError("MCARType2 must be fit before calling transform.")
        if not isinstance(X, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        if not (0 <= self.missing_rate <= 1):
            raise ValueError("missing_rate must be between 0 and 1.")

        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        total_elements = X.size
        num_missing = int(round(total_elements * self.missing_rate))

        X_missing = X.copy()
        flat_indices = rng.choice(total_elements, size=num_missing, replace=False)
        multi_indices = np.unravel_index(flat_indices, X.shape)
        X_missing[multi_indices] = np.nan
        return X_missing


class MCARType3:
    """
    MCAR Mechanism - Type 3 (Column-wise Balanced Missingness)

    Applies missingness to each column independently, with approximately
    equal number of missing entries per column.

    Parameters
    ----------
    missing_rate : float, default=0.1
        The total proportion of missing values in the dataset.
    seed : int, default=1
        Random seed for reproducibility.
    """
    def __init__(self, missing_rate=0.1, seed=1):
        self.missing_rate = missing_rate
        self.seed = seed
        self.fitted = False

    def fit(self, X, y=None):
        """
        Placeholder fit method for interface compatibility.

        MCARType2 does not require fitting, but this method sets a flag for internal consistency.
        """
        self.fitted = True
        return self

    def transform(self, X):
        """
        Apply MCARType3 transformation to introduce missingness.

        Ensures that missing values are approximately evenly distributed across columns.

        Parameters
        ----------
        X : np.ndarray
            Input array to apply missingness (converted to float).

        Returns
        -------
        X_missing : np.ndarray
            Array with missing values inserted in a column-balanced way.
        """
        if not self.fitted:
            raise RuntimeError("MCARType3 must be fit before calling transform.")
        if not isinstance(X, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        if not (0 <= self.missing_rate <= 1):
            raise ValueError("missing_rate must be between 0 and 1.")

        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        n, p = X.shape
        total_cells = n * p
        total_missing = int(round(total_cells * self.missing_rate))
        missing_per_col = total_missing // p

        X_missing = X.copy()
        for j in range(p):
            if missing_per_col > 0:
                rows = rng.choice(n, size=missing_per_col, replace=False)
                X_missing[rows, j] = np.nan
        return X_missing



MCAR_TYPES = {
    1: MCARType1,
    2: MCARType2,
    3: MCARType3,
}
