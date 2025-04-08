# Update mar_type1 to return data_with_missing instead of just the mask
import numpy as np
from scipy.special import expit
from scipy.optimize import bisect
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pointbiserialr



class MARType1:
    def __init__(self, missing_rate=0.1, seed=1, para=0.3, depend_on=None):
        self.missing_rate = missing_rate
        self.seed = seed
        self.p_obs = para  # ✅ 确保不是 None
        self.depend_on = depend_on
        self.fitted = False
        

    def fit(self, X, y=None, xs = None):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.X_shape = (n, d)
        self.xs = xs  # 当前要加缺失的列（可为 None）

        if self.depend_on is not None:
            self.idxs_obs = np.array([i for i in self.depend_on if i != xs])
        else:
            self.idxs_obs = rng.choice(d, max(int(self.p_obs * d), 1), replace=False)

        if xs is not None:
            self.idxs_nas = np.array([xs])
        else:
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



from sklearn.feature_selection import mutual_info_classif
import numpy as np

class MARType2:
    def __init__(self, missing_rate=0.1, seed=1, depend_on=None):
        self.missing_rate = missing_rate
        self.seed = seed
        self.depend_on = depend_on
        self.fitted = False

    def _verbose(self, msg):
        print(f"[{self.__class__.__name__}] {msg}")

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        n, p = X.shape

        if self.depend_on is not None:
            cols = self.depend_on
        else:
            cols = list(range(p))

        # Create fake label to estimate MI
        fake_label = (X @ rng.normal(size=(p,)) > 0).astype(int)

        self.mi = mutual_info_classif(X[:, cols], fake_label, discrete_features='auto', random_state=self.seed)
        self.mi = np.clip(self.mi, a_min=1e-6, a_max=None)

        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Call .fit() before .transform().")

        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        n, p = X.shape
        X_missing = X.copy()

        total_missing = int(round(n * p * self.missing_rate))
        missing_per_col = max(total_missing // p, 1)

        for j in range(p):
            k = min(missing_per_col, n)
            rows = rng.choice(n, size=k, replace=False)
            X_missing[rows, j] = np.nan

        return X_missing




import numpy as np
from scipy.stats import pointbiserialr


class MARType3:
    def __init__(self, missing_rate=0.1, seed=1, depend_on=None):
        self.missing_rate = missing_rate
        self.seed = seed
        self.depend_on = depend_on
        self.fitted = False

    def _verbose(self, msg):
        print(f"[{self.__class__.__name__}] {msg}")

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        n, p = X.shape

        if self.depend_on is not None:
            self.depend_cols = self.depend_on
        else:
            self.depend_cols = list(range(p))  # 默认全列

        if y is not None:
            Y = y
        else:
            self._verbose("No label provided. Using synthetic labels instead.")
            Y = (X @ rng.normal(size=(p,)) > 0).astype(int)

        corrs = []
        for j in self.depend_cols:
            try:
                r, _ = pointbiserialr(Y, X[:, j])
                corrs.append(abs(r))
            except Exception:
                corrs.append(0.0)

        self.corr_score = max(np.mean(corrs), 1e-6)
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Call .fit() before .transform().")

        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        n, p = X.shape
        X_missing = X.copy()

        total = int(round(n * p * self.missing_rate))
        idx = rng.choice(n * p, size=total, replace=False)
        rows, cols = np.unravel_index(idx, (n, p))
        X_missing[rows, cols] = np.nan
        return X_missing


import numpy as np
from scipy.stats import pointbiserialr

class MARType4:
    def __init__(self, missing_rate=0.1, seed=1, depend_on=None):
        self.missing_rate = missing_rate
        self.seed = seed
        self.depend_on = depend_on
        self.fitted = False
        print("My Depend on",self.depend_on)

    def _verbose(self, msg):
        print(f"[{self.__class__.__name__}] {msg}")

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        n, p = X.shape

        # 决定依赖列：用于生成伪标签（或真实标签）来计算 correlation
        if self.depend_on is not None:
            depend_cols = self.depend_on
        else:
            depend_cols = list(range(p))

        # 获取标签
        if y is not None:
            Y = y
        else:
            self._verbose("No label provided. Using synthetic labels instead.")
            Y = (X[:, depend_cols] @ rng.normal(size=(len(depend_cols),)) > 0).astype(int)

        # 用 Y 计算和每一列的相关性，排序出 xs（要加缺失的列）
        corrs = []
        for j in range(p):
            try:
                r, _ = pointbiserialr(Y, X[:, j])
                corrs.append(abs(r))
            except Exception:
                corrs.append(0)
        self.xs_indices = np.argsort(corrs)  # 从相关性小到大
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Call .fit() before .transform().")

        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        n, p = X.shape
        X_missing = X.copy()

        total_missing = int(round(n * p * self.missing_rate))
        missing_each = max(total_missing // len(self.xs_indices), 1)

        for xs in self.xs_indices:
            # 找出与当前列最相关的列 xd
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

            # 在 xd 上排序 → 取最小的值对应的行 → 对 xs 加缺失
            order = np.argsort(X[:, xd])
            selected_rows = order[:min(missing_each, n)]
            X_missing[selected_rows, xs] = np.nan

        return X_missing


import numpy as np

class MARType5:
    def __init__(self, missing_rate=0.1, seed=1, depend_on=None):
        self.missing_rate = missing_rate
        self.seed = seed
        self.depend_on = depend_on
        self.fitted = False

    def _verbose(self, msg):
        print(f"[{self.__class__.__name__}] {msg}")

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        n, p = X.shape

        # 如果用户有指定依赖列，则从中选择一个；否则从所有列中选
        if self.depend_on is not None:
            candidates = self.depend_on
        else:
            candidates = list(range(p))

        self.xd = rng.choice(candidates)
        self._verbose(f"Selected column {self.xd} as dependency (xd).")
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Call .fit() before .transform().")
        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        n, p = X.shape

        xs_indices = [i for i in range(p) if i != self.xd]
        total_missing = int(round(n * len(xs_indices) * self.missing_rate))
        missing_per_col = max(total_missing // len(xs_indices), 1)

        xd_col = X[:, self.xd]
        order = np.argsort(xd_col)
        rank = np.empty_like(order)
        rank[order] = np.arange(1, n + 1)
        prob_vector = rank / rank.sum()

        X_missing = X.copy()
        for xs in xs_indices:
            selected_rows = rng.choice(n, size=min(missing_per_col, n), replace=False, p=prob_vector)
            X_missing[selected_rows, xs] = np.nan

        return X_missing



import numpy as np

class MARType6:
    def __init__(self, missing_rate=0.1, seed=1, depend_on=None):
        self.missing_rate = missing_rate
        self.seed = seed
        self.depend_on = depend_on
        self.fitted = False

    def _verbose(self, msg):
        print(f"[{self.__class__.__name__}] {msg}")

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        n, p = X.shape

        # 依赖列选择逻辑
        if self.depend_on is not None:
            candidates = self.depend_on
        else:
            candidates = list(range(p))

        self.xd = rng.choice(candidates)
        self._verbose(f"Selected column {self.xd} as dependency (xd).")
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Call .fit() before .transform().")

        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        n, p = X.shape
        X_missing = X.copy()

        xs_indices = [i for i in range(p) if i != self.xd]
        total_missing = int(round(n * len(xs_indices) * self.missing_rate))
        missing_per_col = max(total_missing // len(xs_indices), 1)

        xd_col = X[:, self.xd]
        median_val = np.median(xd_col)
        group_high = xd_col >= median_val
        group_low = xd_col < median_val

        pb = np.zeros(n)
        if group_high.sum() > 0:
            pb[group_high] = 0.9 / group_high.sum()
        if group_low.sum() > 0:
            pb[group_low] = 0.1 / group_low.sum()

        for xs in xs_indices:
            selected_rows = rng.choice(n, size=min(missing_per_col, n), replace=False, p=pb)
            X_missing[selected_rows, xs] = np.nan

        return X_missing



import numpy as np

class MARType7:
    def __init__(self, missing_rate=0.1, seed=1, depend_on=None):
        self.missing_rate = missing_rate
        self.seed = seed
        self.depend_on = depend_on
        self.fitted = False

    def _verbose(self, msg):
        print(f"[{self.__class__.__name__}] {msg}")

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        n, p = X.shape

        if self.depend_on is not None:
            candidates = self.depend_on
        else:
            candidates = list(range(p))

        self.xd = rng.choice(candidates)
        self._verbose(f"Selected column {self.xd} as dependency (xd).")
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Call .fit() before .transform().")

        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        n, p = X.shape
        X_missing = X.copy()

        xs_indices = [i for i in range(p) if i != self.xd]
        total_missing = int(round(n * len(xs_indices) * self.missing_rate))
        missing_per_col = max(total_missing // len(xs_indices), 1)

        xd_col = X[:, self.xd]
        top_indices = np.argsort(xd_col)[-missing_per_col:]

        for xs in xs_indices:
            X_missing[top_indices, xs] = np.nan

        return X_missing



# class MARType8:
#     def __init__(self, missing_rate=0.1, seed=1):
#         self.missing_rate = missing_rate
#         self.seed = seed
#         self.fitted = False

#     def fit(self, X, y=None):
#         rng = np.random.default_rng(self.seed)
#         self.xd = rng.integers(0, X.shape[1])
#         self._verbose(f"Selected column {self.xd} as dependency (xd).")
#         self.fitted = True
#         return self

#     def transform(self, X):
#         if not self.fitted:
#             raise RuntimeError("Call .fit() before .transform().")
#         rng = np.random.default_rng(self.seed)
#         X = X.astype(float)
#         n, p = X.shape
#         xs_indices = [i for i in range(p) if i != self.xd]
#         total_missing = int(round(n * p * self.missing_rate))
#         missing_per_col = max(total_missing // len(xs_indices), 1)

#         xd_col = X[:, self.xd]
#         sorted_indices = np.argsort(xd_col)
#         if missing_per_col % 2 == 0:
#             low_indices = sorted_indices[:missing_per_col // 2]
#             high_indices = sorted_indices[-missing_per_col // 2:]
#         else:
#             low_indices = sorted_indices[:missing_per_col // 2 + 1]
#             high_indices = sorted_indices[-missing_per_col // 2:]
#         selected_indices = np.concatenate([low_indices, high_indices])

#         data_with_missing = X.copy()
#         for xs in xs_indices:
#             data_with_missing[selected_indices, xs] = np.nan
#         return data_with_missing
    

#     def _verbose(self, msg):
#         print(f"[{self.__class__.__name__}] {msg}")
import numpy as np

class MARType8:
    def __init__(self, missing_rate=0.1, seed=1, depend_on=None):
        self.missing_rate = missing_rate
        self.seed = seed
        self.depend_on = depend_on
        self.fitted = False

    def _verbose(self, msg):
        print(f"[{self.__class__.__name__}] {msg}")

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        n, p = X.shape

        if self.depend_on is not None:
            candidates = self.depend_on
        else:
            candidates = list(range(p))

        self.xd = rng.choice(candidates)
        self._verbose(f"Selected column {self.xd} as dependency (xd).")
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Call .fit() before .transform().")

        rng = np.random.default_rng(self.seed)
        X = X.astype(float)
        n, p = X.shape
        X_missing = X.copy()

        xs_indices = [i for i in range(p) if i != self.xd]
        total_missing = int(round(n * len(xs_indices) * self.missing_rate))
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

        for xs in xs_indices:
            X_missing[selected_indices, xs] = np.nan

        return X_missing

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