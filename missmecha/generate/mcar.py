import numpy as np


class MCARType1:
    def __init__(self, missing_rate=0.1, seed=1):
        self.missing_rate = missing_rate
        self.seed = seed
        self.fitted = False  # 一般MCAR不需要 fit, 但我们保持统一接口

    def fit(self, X, y=None):
        # MCAR 不依赖 X 或 y，fit 只是设置标志位
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("MCARType1 must be fit before calling transform.")

        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        mask = rng.uniform(0, 1, size=X.shape) < self.missing_rate
        X_missing = X.copy()
        X_missing[mask] = np.nan
        return X_missing

class MCARType2:
    def __init__(self, missing_rate=0.1, seed=1):
        self.missing_rate = missing_rate
        self.seed = seed
        self.fitted = False

    def fit(self, X, y=None):
        # MCAR 不依赖 X/y，fit 仅作为流程接口
        self.fitted = True
        return self

    def transform(self, X):
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
    def __init__(self, missing_rate=0.1, seed=1):
        self.missing_rate = missing_rate
        self.seed = seed
        self.fitted = False

    def fit(self, X, y=None):
        self.fitted = True
        return self

    def transform(self, X):
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
