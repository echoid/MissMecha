# Update mar_type1 to return data_with_missing instead of just the mask
import numpy as np

from scipy.special import expit
from scipy.optimize import bisect
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pointbiserialr

class MARType1:    
    """
    MAR Mechanism - Type 1 (Logistic Missingness Based on Observed Features)

    Introduces missingness based on a logistic model, where the missingness probability
    depends on a subset of observed features.

    Parameters
    ----------
    missing_rate : float, default=0.1
        Target proportion of missing entries.
    seed : int, default=1
        Random seed for reproducibility.
    para : float, default=0.3
        Proportion of observed features to use when no `depend_on` is specified.
    depend_on : list[int] or None
        Indices of features to use as observed covariates. If None, sampled randomly.
    """
    def __init__(self, missing_rate=0.1, seed=1, para=0.3, depend_on=None):
        self.missing_rate = missing_rate
        self.seed = seed
        self.p_obs = para  # ✅ 确保不是 None
        self.depend_on = depend_on
        self.fitted = False
        

    def fit(self, X, y=None, xs = None):
        """
        Fit the logistic model to determine missingness probabilities.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix.
        y : Ignored
            Included for compatibility.
        xs : int or None
            Index of the feature to mask. If None, all non-observed features will be masked.

        Returns
        -------
        self : MARType1
            Fitted object with learned parameters.
        """
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
        """
        Apply the learned MARType1 mechanism to the input data.

        Parameters
        ----------
        X : np.ndarray
            Input data to apply missingness to.

        Returns
        -------
        X_missing : np.ndarray
            Array with NaN entries introduced based on the fitted logistic model.
        """
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

class MARType2:
    """
    MAR Mechanism - Type 2 (Mutual Information-Based Feature Ranking)

    Selects features with high mutual information scores relative to a synthetic label,
    and introduces missingness proportionally across features.

    Parameters
    ----------
    missing_rate : float, default=0.1
        Overall proportion of missing entries.
    seed : int, default=1
        Random seed for reproducibility.
    depend_on : list[int] or None
        List of features to compute mutual information against. If None, all features are used.
    """
    def __init__(self, missing_rate=0.1, seed=1, depend_on=None):
        self.missing_rate = missing_rate
        self.seed = seed
        self.depend_on = depend_on
        self.fitted = False

    def _verbose(self, msg):
        print(f"[{self.__class__.__name__}] {msg}")

    def fit(self, X, y=None):
        """
        Compute mutual information scores from observed features and fit internal parameters.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix.
        y : Ignored
            Included for compatibility.

        Returns
        -------
        self : MARType2
            Fitted object.
        """
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
        """
        Apply missingness proportionally across all features.

        Parameters
        ----------
        X : np.ndarray
            Input data to apply missingness to.

        Returns
        -------
        X_missing : np.ndarray
            Transformed array with missing entries.
        """
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


class MARType3:
    """
    MAR Mechanism - Type 3 (Point-Biserial Correlation with Observed or Synthetic Label)

    Estimates the importance of each feature by computing point-biserial correlation
    between each column and a binary target (real or synthetic). The overall correlation
    score determines the intensity of random missingness.

    Parameters
    ----------
    missing_rate : float, default=0.1
        Overall proportion of missing values to introduce.
    seed : int, default=1
        Random seed for reproducibility.
    depend_on : list[int] or None
        Columns used to construct synthetic labels if `y` is not provided.
    """
    def __init__(self, missing_rate=0.1, seed=1, depend_on=None):
        self.missing_rate = missing_rate
        self.seed = seed
        self.depend_on = depend_on
        self.fitted = False

    def _verbose(self, msg):
        print(f"[{self.__class__.__name__}] {msg}")

    def fit(self, X, y=None):
        """
        Compute feature correlations with a binary label and determine average correlation.

        If a label `y` is not provided, a synthetic label is generated by projecting the data
        onto a random direction. Point-biserial correlation is then calculated between each feature
        and the binary label to estimate dependency strength.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix (will be converted to float).
        y : np.ndarray or None
            Optional binary label. If not provided, a synthetic label will be generated.

        Returns
        -------
        self : MARType3
            Fitted object containing average correlation score.
        """
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
        """
        Apply uniform missingness with intensity guided by average point-biserial correlation.

        Missing entries are randomly introduced into the data matrix based on the
        fitted correlation score and the desired missing rate.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix to apply missingness.

        Returns
        -------
        X_missing : np.ndarray
            Transformed data with missing values inserted.
        """
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

class MARType4:
    """
    MAR Mechanism - Type 4 (Correlation-Driven Column Ranking with Pairwise Masking)

    Selects features with weakest correlation to a binary label (real or synthetic),
    then introduces missing values into those features based on their relationship
    with the most correlated partner column.

    Parameters
    ----------
    missing_rate : float, default=0.1
        Target proportion of missing entries.
    seed : int, default=1
        Random seed for reproducibility.
    depend_on : list[int] or None
        Columns to use when generating synthetic label. If None, all columns are used.
    """
    def __init__(self, missing_rate=0.1, seed=1, depend_on=None):
        self.missing_rate = missing_rate
        self.seed = seed
        self.depend_on = depend_on
        self.fitted = False
        print("My Depend on",self.depend_on)

    def _verbose(self, msg):
        print(f"[{self.__class__.__name__}] {msg}")

    def fit(self, X, y=None):
        """
        Compute feature correlations to a binary label and rank features by relevance.

        A synthetic label is generated from selected columns if `y` is not provided.
        Features with weakest correlation are selected as targets for masking. Their
        most correlated counterpart feature is later used to determine which rows to mask.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix (will be converted to float).
        y : np.ndarray or None
            Optional binary label. If not provided, a synthetic label is generated.

        Returns
        -------
        self : MARType4
            Fitted object storing ranked feature indices.
        """

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
        """
        Apply column-wise masking based on correlations with paired columns.

        For each target column, the most correlated other column is identified.
        Rows with the smallest values in the correlated column are masked in the target column.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix to apply missingness.

        Returns
        -------
        X_missing : np.ndarray
            Transformed data with missing values applied.
        """
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

class MARType5:
    """
    MAR Mechanism - Type 5 (Rank-Based Missingness from a Dependent Feature)

    Selects a single column as the dependency feature (xd), and generates missingness
    in all other columns based on ranks in xd. Rows with higher values in xd are more
    likely to be selected for missingness.

    Parameters
    ----------
    missing_rate : float, default=0.1
        Target proportion of missing entries.
    seed : int, default=1
        Random seed for reproducibility.
    depend_on : list[int] or None
        Candidate columns to select as the dependency column. If None, all columns are considered.
    """

    def __init__(self, missing_rate=0.1, seed=1, depend_on=None):
        self.missing_rate = missing_rate
        self.seed = seed
        self.depend_on = depend_on
        self.fitted = False

    def _verbose(self, msg):
        print(f"[{self.__class__.__name__}] {msg}")

    def fit(self, X, y=None):
        """
        Select a dependency feature to control missingness.

        A single column is randomly selected from the specified candidates (or all columns if `depend_on` is None)
        and stored as the controlling feature for masking.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix (converted to float).
        y : Ignored
            Included for compatibility.

        Returns
        -------
        self : MARType5
            Fitted object storing the selected dependency feature.
        """
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
        """
        Introduce missing values based on rank probabilities from the selected feature.

        The higher the rank (value) of a row in the selected dependency feature,
        the more likely it is to be chosen for masking across other columns.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix to apply missingness.

        Returns
        -------
        X_missing : np.ndarray
            Transformed data with NaNs introduced based on ranked dependency.
        """
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

class MARType6:
    """
    MAR Mechanism - Type 6 (Skewed Binary Grouping Based on Dependency Column)

    Partitions the dataset into two groups (high vs. low) based on the median of a selected
    dependency column. Then introduces missingness with skewed probabilities between the groups
    (e.g., 90% from the high group, 10% from the low group).

    Parameters
    ----------
    missing_rate : float, default=0.1
        Proportion of total values to mask.
    seed : int, default=1
        Random seed for reproducibility.
    depend_on : list[int] or None
        Candidate columns to select the controlling feature (xd). If None, all columns are considered.
    """

    def __init__(self, missing_rate=0.1, seed=1, depend_on=None):
        self.missing_rate = missing_rate
        self.seed = seed
        self.depend_on = depend_on
        self.fitted = False

    def _verbose(self, msg):
        print(f"[{self.__class__.__name__}] {msg}")

    def fit(self, X, y=None):
        """
        Select a dependency feature to define group-based masking.

        Randomly selects one feature (xd) from the candidate list or all columns.
        This feature is later used to partition rows into high/low groups.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix.
        y : Ignored
            Included for compatibility.

        Returns
        -------
        self : MARType6
            Fitted object storing the selected dependency column.
        """

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
        """
        Apply missingness by sampling more frequently from one group.

        The selected feature xd is used to split the rows into two groups
        based on median value. Rows from the higher-value group are sampled
        with greater probability to introduce missing values across other columns.

        Parameters
        ----------
        X : np.ndarray
            Input data to apply missingness.

        Returns
        -------
        X_missing : np.ndarray
            Transformed array with missing values introduced.
        """

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

class MARType7:
    """
    MAR Mechanism - Type 7 (Top Value Masking Based on Dependency Column)

    Selects a controlling feature (xd), ranks its values, and applies missingness
    to the top-ranked rows (those with the highest values) across the remaining columns.

    Parameters
    ----------
    missing_rate : float, default=0.1
        Target proportion of values to mask.
    seed : int, default=1
        Random seed to ensure reproducibility.
    depend_on : list[int] or None
        List of candidate features for controlling missingness. If None, selects from all columns.
    """

    def __init__(self, missing_rate=0.1, seed=1, depend_on=None):
        self.missing_rate = missing_rate
        self.seed = seed
        self.depend_on = depend_on
        self.fitted = False

    def _verbose(self, msg):
        print(f"[{self.__class__.__name__}] {msg}")

    def fit(self, X, y=None):
        """
        Randomly select a column to use for top-value-based masking.

        The selected feature (xd) will determine which rows receive missingness,
        by identifying the highest-valued entries.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix.
        y : Ignored
            Included for interface consistency.

        Returns
        -------
        self : MARType7
            Fitted object containing the controlling feature.
        """

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
        """
        Introduce missing values into the rows with the highest values in the selected feature.

        For each non-controlling column, missingness is applied to a fixed number of rows
        corresponding to the top-ranked values in the dependency column.

        Parameters
        ----------
        X : np.ndarray
            Input data to transform.

        Returns
        -------
        X_missing : np.ndarray
            Array with missing values inserted into top-ranked rows.
        """

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

class MARType8:
    """
    MAR Mechanism - Type 8 (Extreme Value Masking Based on Dependency Column)

    Applies missingness to rows with the most extreme values (both lowest and highest)
    in a selected controlling feature (xd), and masks the rest of the columns accordingly.

    Parameters
    ----------
    missing_rate : float, default=0.1
        Desired overall proportion of missing values.
    seed : int, default=1
        Random seed for reproducibility.
    depend_on : list[int] or None
        Columns to choose from as the dependency column. If None, selects from all features.
    """

    def __init__(self, missing_rate=0.1, seed=1, depend_on=None):
        self.missing_rate = missing_rate
        self.seed = seed
        self.depend_on = depend_on
        self.fitted = False

    def _verbose(self, msg):
        print(f"[{self.__class__.__name__}] {msg}")

    def fit(self, X, y=None):
        """
        Select a dependency feature and identify extreme-valued rows.

        The selected column (xd) is used to rank all rows.
        Both low and high extremes will be targeted for masking during transformation.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix (converted to float).
        y : Ignored
            Included for compatibility.

        Returns
        -------
        self : MARType8
            Fitted object storing the selected dependency column.
        """

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
        """
        Apply missingness to extreme-value rows in the selected column.

        Both the highest and lowest value rows in the dependency column are
        selected, and missing values are introduced into the remaining columns.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix to apply missingness.

        Returns
        -------
        X_missing : np.ndarray
            Transformed data with missing entries introduced in extreme rows.
        """

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