# Update mar_type1 to return data_with_missing instead of just the mask
import numpy as np
from scipy.special import expit
from scipy.optimize import bisect
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pointbiserialr

class MARType1:
    def __init__(self, missing_rate=0.1, seed=1, p_obs=0.3):
        self.missing_rate = missing_rate
        self.seed = seed
        self.p_obs = p_obs
        self.fitted = False

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.X_shape = (n, d)

        self.idxs_obs = rng.choice(d, max(int(self.p_obs * d), 1), replace=False)
        self.idxs_nas = np.array([i for i in range(d) if i not in self.idxs_obs])

        X_obs = X[:, self.idxs_obs].copy()
        X_obs_mean = np.nanmean(X_obs, axis=0)
        inds = np.where(np.isnan(X_obs))
        X_obs[inds] = np.take(X_obs_mean, inds[1])

        self.W = rng.standard_normal((len(self.idxs_obs), len(self.idxs_nas)))
        self.logits = X_obs @ self.W

        # Fit intercepts to achieve the desired missing rate
        self.intercepts = np.zeros(len(self.idxs_nas))
        for j in range(len(self.idxs_nas)):
            def f(x):
                return np.mean(expit(self.logits[:, j] + x)) - self.missing_rate
            self.intercepts[j] = bisect(f, -1000, 1000)

        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Call .fit() before .transform().")

        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        n, d = X.shape

        # Recompute logits using W
        X_obs = X[:, self.idxs_obs].copy()
        X_obs_mean = np.nanmean(X_obs, axis=0)
        inds = np.where(np.isnan(X_obs))
        X_obs[inds] = np.take(X_obs_mean, inds[1])

        logits = X_obs @ self.W
        ps = expit(logits + self.intercepts)

        mask = np.zeros((n, d), dtype=bool)
        mask[:, self.idxs_nas] = rng.random((n, len(self.idxs_nas))) < ps

        X_missing = X.copy()
        X_missing[mask] = np.nan
        return X_missing

import numpy as np
from sklearn.feature_selection import mutual_info_classif

class MARType2:
    def __init__(self, missing_rate=0.1, seed=1):
        self.missing_rate = missing_rate
        self.seed = seed
        self.fitted = False

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        n, p = X.shape

        # Create a pseudo-label based on random linear combination
        self.fake_label = (X @ rng.normal(size=(p,)) > 0).astype(int)

        # Compute mutual information
        self.mi = mutual_info_classif(X, self.fake_label, discrete_features='auto', random_state=self.seed)
        self.mi = np.clip(self.mi, a_min=1e-6, a_max=None)

        self.total_missing = int(round(n * p * self.missing_rate))
        self.probs = self.mi / self.mi.sum()
        self.missing_per_col = (self.probs * self.total_missing).astype(int)

        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Call .fit() before .transform().")

        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        n, p = X.shape

        X_missing = X.copy()
        for j in range(p):
            k = min(self.missing_per_col[j], n)
            rows = rng.choice(n, size=k, replace=False)
            X_missing[rows, j] = np.nan

        return X_missing


class MARType3:
    def __init__(self, missing_rate=0.1, seed=1):
        self.missing_rate = missing_rate
        self.seed = seed

    def fit(self, X, y=None):
        self.fitted = True
        return self

    def transform(self, X):
        rng = np.random.default_rng(self.seed)
        n, p = X.shape
        Y = (X @ rng.normal(size=(p,)) > 0).astype(int)

        corrs = []
        for j in range(p):
            try:
                r, _ = pointbiserialr(Y, X[:, j])
                corrs.append(abs(r))
            except Exception:
                corrs.append(0.0)
        corrs = np.array(corrs)
        corrs = np.clip(corrs, 1e-6, None)

        total_missing = int(round(n * p * self.missing_rate))
        probs = corrs / corrs.sum()
        missing_per_col = (probs * total_missing).astype(int)

        X_missing = X.copy().astype(float)
        for j in range(p):
            k = min(missing_per_col[j], n)
            rows = rng.choice(n, size=k, replace=False)
            X_missing[rows, j] = np.nan
        return X_missing
from typing import Optional
import numpy as np
from scipy.stats import pointbiserialr

class MARType4:
    def __init__(self, missing_rate=0.1, seed=1):
        self.missing_rate = missing_rate
        self.seed = seed
        self.fitted = False

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        n, p = X.shape

        # Generate pseudo-label
        Y = (X @ rng.normal(size=(p,)) > 0).astype(int)

        # Select xs features most correlated with Y
        corrs = []
        for j in range(p):
            try:
                r, _ = pointbiserialr(Y, X[:, j])
                corrs.append(abs(r))
            except Exception:
                corrs.append(0)
        self.xs_indices = np.argsort(corrs)
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Call .fit() before .transform().")
        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        n, p = X.shape
        data_with_missing = X.copy()
        total_missing = int(round(n * p * self.missing_rate))
        missing_each = max(total_missing // len(self.xs_indices), 1)

        for xs in self.xs_indices:
            corrs = []
            for j in range(p):
                if j == xs:
                    corrs.append(-np.inf)
                else:
                    try:
                        r, _ = pointbiserialr(X[:, xs], X[:, j])
                        corrs.append(abs(r))
                    except Exception:
                        corrs.append(0)
            xd = int(np.argmax(corrs))
            order = np.argsort(X[:, xd])
            selected_rows = order[:min(missing_each, n)]
            data_with_missing[selected_rows, xs] = np.nan

        return data_with_missing


class MARType5:
    def __init__(self, missing_rate=0.1, seed=1):
        self.missing_rate = missing_rate
        self.seed = seed
        self.fitted = False

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.seed)
        self.xd = rng.integers(0, X.shape[1])
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Call .fit() before .transform().")
        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        n, p = X.shape
        total_missing = int(round(n * p * self.missing_rate))
        xs_indices = [i for i in range(p) if i != self.xd]
        missing_per_col = max(total_missing // len(xs_indices), 1)

        xd_col = X[:, self.xd]
        order = np.argsort(xd_col)
        rank = np.empty_like(order)
        rank[order] = np.arange(1, n + 1)
        prob_vector = rank / rank.sum()

        data_with_missing = X.copy()
        for xs in xs_indices:
            selected_rows = rng.choice(n, size=min(missing_per_col, n), replace=False, p=prob_vector)
            data_with_missing[selected_rows, xs] = np.nan
        return data_with_missing


class MARType6:
    def __init__(self, missing_rate=0.1, seed=1):
        self.missing_rate = missing_rate
        self.seed = seed
        self.fitted = False

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.seed)
        self.xd = rng.integers(0, X.shape[1])
        
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Call .fit() before .transform().")
        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        n, p = X.shape
        xs_indices = [i for i in range(p) if i != self.xd]
        total_missing = int(round(n * p * self.missing_rate))
        missing_per_col = max(total_missing // len(xs_indices), 1)

        self.xd_col = X[:, self.xd]
        median_val = np.median(self.xd_col)
        group_high = self.xd_col >= median_val
        group_low = self.xd_col < median_val
        pb = np.zeros(n)
        pb[group_high] = 0.9 / group_high.sum()
        pb[group_low] = 0.1 / group_low.sum()

        data_with_missing = X.copy()
        for xs in xs_indices:
            selected_rows = rng.choice(n, size=min(missing_per_col, n), replace=False, p=pb)
            data_with_missing[selected_rows, xs] = np.nan
        return data_with_missing


class MARType7:
    def __init__(self, missing_rate=0.1, seed=1):
        self.missing_rate = missing_rate
        self.seed = seed
        self.fitted = False

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.seed)
        self.xd = rng.integers(0, X.shape[1])
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Call .fit() before .transform().")
        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        n, p = X.shape
        xs_indices = [i for i in range(p) if i != self.xd]
        total_missing = int(round(n * p * self.missing_rate))
        missing_per_col = max(total_missing // len(xs_indices), 1)

        xd_col = X[:, self.xd]
        top_indices = np.argsort(xd_col)[-missing_per_col:]

        data_with_missing = X.copy()
        for xs in xs_indices:
            data_with_missing[top_indices, xs] = np.nan
        return data_with_missing


class MARType8:
    def __init__(self, missing_rate=0.1, seed=1):
        self.missing_rate = missing_rate
        self.seed = seed
        self.fitted = False

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.seed)
        self.xd = rng.integers(0, X.shape[1])
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Call .fit() before .transform().")
        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        n, p = X.shape
        xs_indices = [i for i in range(p) if i != self.xd]
        total_missing = int(round(n * p * self.missing_rate))
        missing_per_col = max(total_missing // len(xs_indices), 1)

        xd_col = X[:, self.xd]
        sorted_indices = np.argsort(xd_col)
        if missing_per_col % 2 == 0:
            low_indices = sorted_indices[:missing_per_col // 2]
            high_indices = sorted_indices[-missing_per_col // 2:]
        else:
            low_indices = sorted_indices[:missing_per_col // 2 + 1]
            high_indices = sorted_indices[-missing_per_col // 2:]
        selected_indices = np.concatenate([low_indices, high_indices])

        data_with_missing = X.copy()
        for xs in xs_indices:
            data_with_missing[selected_indices, xs] = np.nan
        return data_with_missing

MAR_TYPES = {
    1: MARType1,
    2: MARType2,
    3: MARType3,
    4: MARType4,
    5: MARType5,
    6: MARType6,
    7: MARType7,
    8: MARType8

}