# Update mar_type1 to return data_with_missing instead of just the mask
import numpy as np
from scipy.special import expit
from scipy.optimize import bisect
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pointbiserialr
def mar_type1(data, missing_rate=0.1, seed=1, p_obs=0.3):
    """
    MAR Type 1 — Logistic model-based missing at random.
    Some variables are always observed; others are masked with probabilities
    derived from a logistic model conditioned on the observed variables.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Input dataset.

    missing_rate : float, default=0.1
        Proportion of missing values to introduce in maskable variables.

    seed : int, default=1
        Random seed for reproducibility (default is always 1).

    p_obs : float, default=0.3
        Proportion of variables that are always fully observed.

    Returns
    -------
    data_with_missing : np.ndarray
        A copy of the input data with missing values (np.nan) inserted.
    """
    rng = np.random.default_rng(seed)
    n, d = data.shape
    X = data.copy()

    d_obs = max(int(p_obs * d), 1)
    idxs_obs = rng.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    W = rng.standard_normal((d_obs, len(idxs_nas)))

    X_obs = X[:, idxs_obs]
    X_obs_mean = np.nanmean(X_obs, axis=0)
    inds = np.where(np.isnan(X_obs))
    X_obs[inds] = np.take(X_obs_mean, inds[1])

    logits = X_obs @ W


    intercepts = np.zeros(len(idxs_nas))
    for j in range(len(idxs_nas)):
        def f(x):
            return np.mean(expit(logits[:, j] + x)) - missing_rate
        intercepts[j] = bisect(f, -1000, 1000)

    ps = expit(logits + intercepts)
    mask = np.zeros((n, d), dtype=bool)
    mask[:, idxs_nas] = rng.random((n, len(idxs_nas))) < ps

    data_with_missing = data.copy().astype(float)
    data_with_missing[mask] = np.nan

    return data_with_missing

def mar_type2(data, missing_rate=0.1, seed=1):
    """
    MAR Type 2 (Global All Columns) — Inject missing values into all features,
    where each feature's missing probability is determined by its mutual information
    with a pseudo-label (most informative).

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Input dataset.

    missing_rate : float, default=0.1
        Total fraction of missing values in the dataset.

    seed : int, default=1
        Random seed for reproducibility.

    Returns
    -------
    data_with_missing : np.ndarray
        Dataset with NaNs inserted in every column based on relevance to pseudo-label.
    """
    rng = np.random.default_rng(seed)
    n, p = data.shape
    data = data.astype(float)

    fake_label = (data @ rng.normal(size=(p,)) > 0).astype(int)
    mi = mutual_info_classif(data, fake_label, discrete_features='auto', random_state=seed)
    mi = np.clip(mi, a_min=1e-6, a_max=None)  # avoid divide by zero

    total_missing = int(round(n * p * missing_rate))
    probs = mi / mi.sum()
    missing_per_col = (probs * total_missing).astype(int)

    data_with_missing = data.copy()
    for j in range(p):
        k = min(missing_per_col[j], n)
        rows = rng.choice(n, size=k, replace=False)
        data_with_missing[rows, j] = np.nan

    return data_with_missing




def mar_type3(data, missing_rate=0.1, seed=1):
    """
    MAR Type 3 — Label-driven MAR adapted from MCAR1univa.
    Missing values are introduced across all columns, with missing probability 
    proportional to their correlation with a pseudo-label.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Input dataset.

    missing_rate : float, default=0.1
        Total fraction of missing values across the dataset.

    seed : int, default=1
        Random seed for reproducibility.

    Returns
    -------
    data_with_missing : np.ndarray
        Dataset with missing values distributed proportionally to feature-label correlation.
    """
    rng = np.random.default_rng(seed)
    n, p = data.shape
    data = data.astype(float)

    # Simulated binary label
    Y = (data @ rng.normal(size=(p,)) > 0).astype(int)

    # Compute point-biserial correlation for each feature
    correlations = []
    for j in range(p):
        try:
            r, _ = pointbiserialr(Y, data[:, j])
            correlations.append(abs(r))
        except Exception:
            correlations.append(0.0)

    corrs = np.array(correlations)
    corrs = np.clip(corrs, 1e-6, None)  # avoid zero probabilities

    # Normalize to get missing distribution
    total_missing = int(round(n * p * missing_rate))
    probs = corrs / corrs.sum()
    missing_per_col = (probs * total_missing).astype(int)

    data_with_missing = data.copy()
    for j in range(p):
        k = min(missing_per_col[j], n)
        rows = rng.choice(n, size=k, replace=False)
        data_with_missing[rows, j] = np.nan

    return data_with_missing

import numpy as np
from scipy.stats import pointbiserialr

def mar_type4(data, missing_rate=0.1, seed=1):
    """
    MAR Type 4 — Global conditional MAR (from MAR1univa).
    Missing values are introduced in multiple columns (xs), where each is highly correlated with a pseudo-label.
    For each such column xs, the determining column xd is chosen as the most correlated column to xs.
    The lowest-ranked rows in xd are used to mask values in xs.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Input dataset.

    missing_rate : float, default=0.1
        Total proportion of missing values across the dataset.

    seed : int, default=1
        Random seed for reproducibility.

    Returns
    -------
    data_with_missing : np.ndarray
        Dataset with NaNs introduced by a conditional MAR process.
    """
    rng = np.random.default_rng(seed)
    n, p = data.shape
    data = data.astype(float)

    # Step 1: Simulate a pseudo binary label
    Y = (data @ rng.normal(size=(p,)) > 0).astype(int)

    # Step 2: Select k most label-correlated features (xs)
    correlations = []
    for j in range(p):
        try:
            r, _ = pointbiserialr(Y, data[:, j])
            correlations.append(abs(r))
        except Exception:
            correlations.append(0)
    correlations = np.array(correlations)
    xs_indices = np.argsort(correlations)  # top-k features (e.g., 3 max)

    # Step 3: For each xs, find most correlated xd
    data_with_missing = data.copy()
    total_missing = int(round(n * p * missing_rate))
    missing_each = max(total_missing // len(xs_indices), 1)

    for xs in xs_indices:
        corrs = []
        for j in range(p):
            if j == xs:
                corrs.append(-np.inf)
                continue
            try:
                r, _ = pointbiserialr(data[:, xs], data[:, j])
                corrs.append(abs(r))
            except Exception:
                corrs.append(0)
        xd = int(np.argmax(corrs))

        # Step 4: Select lowest-ranked values of xd
        order = np.argsort(data[:, xd])
        selected_rows = order[:min(missing_each, n)]

        # Step 5: Inject missing into xs
        data_with_missing[selected_rows, xs] = np.nan

    return data_with_missing


def mar_type5(data, missing_rate=0.1, seed=1):
    """
    MAR Type 5 (Global) — Rieger10-style MAR with ranking-based probability sampling.
    Injects missing values into all columns except xd, using xd's rank-derived probability.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Input dataset.

    missing_rate : float, default=0.1
        Total fraction of missing values across the dataset.

    seed : int, default=1
        Random seed for reproducibility.

    xd : int, optional
        Index of the determining column. If None, randomly selected.

    Returns
    -------
    data_with_missing : np.ndarray
        Dataset with missing values injected across all columns except xd.
    """
    rng = np.random.default_rng(seed)
    n, p = data.shape
    data = data.astype(float)

    xd = rng.integers(0, p)

    all_indices = list(range(p))
    all_indices.remove(xd)
    xs_indices = all_indices  # all columns except xd

    total_missing = int(round(n * p * missing_rate))
    missing_per_col = max(total_missing // len(xs_indices), 1)

    xd_col = data[:, xd]
    order = np.argsort(xd_col)
    rank = np.empty_like(order)
    rank[order] = np.arange(1, n + 1)
    prob_vector = rank / rank.sum()

    data_with_missing = data.copy()
    for xs in xs_indices:
        selected_rows = rng.choice(n, size=min(missing_per_col, n), replace=False, p=prob_vector)
        data_with_missing[selected_rows, xs] = np.nan

    return data_with_missing

def mar_type6(data, missing_rate=0.1, seed=1):
    """
    MAR Type 6 (Global) — Rieger10-style MAR using median-based binary probability split.
    Injects missing values into all columns except xd, where samples with higher xd values
    have a higher probability of being selected for missingness.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Input dataset.

    missing_rate : float, default=0.1
        Total fraction of missing values across the dataset.

    seed : int, default=1
        Random seed for reproducibility.

    xd : int, optional
        Index of the determining column. If None, randomly selected.

    Returns
    -------
    data_with_missing : np.ndarray
        Dataset with missing values injected based on median-probability sampling.
    """
    rng = np.random.default_rng(seed)
    n, p = data.shape
    data = data.astype(float)

    xd = rng.integers(0, p)

    all_indices = list(range(p))
    all_indices.remove(xd)
    xs_indices = all_indices  # all columns except xd

    total_missing = int(round(n * p * missing_rate))
    missing_per_col = max(total_missing // len(xs_indices), 1)

    xd_col = data[:, xd]
    median_val = np.median(xd_col)
    group_high = xd_col >= median_val
    group_low = xd_col < median_val

    pb = np.zeros(n)
    pb[group_high] = 0.9 / group_high.sum()
    pb[group_low] = 0.1 / group_low.sum()

    data_with_missing = data.copy()
    for xs in xs_indices:
        selected_rows = rng.choice(n, size=min(missing_per_col, n), replace=False, p=pb)
        data_with_missing[selected_rows, xs] = np.nan

    return data_with_missing

def mar_type7(data, missing_rate=0.1, seed=1):
    """
    MAR Type 7 (Global) — Value-threshold MAR.
    Injects missing values into all columns except xd by selecting rows
    with the highest values in xd (deterministic, top-N based).

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Input dataset.

    missing_rate : float, default=0.1
        Total fraction of missing values across the dataset.

    seed : int, default=1
        Random seed for reproducibility.

    xd : int, optional
        Index of the determining column. If None, randomly selected.

    Returns
    -------
    data_with_missing : np.ndarray
        Dataset with NaNs injected in top-ranked rows of xd.
    """
    rng = np.random.default_rng(seed)
    n, p = data.shape
    data = data.astype(float)

    xd = rng.integers(0, p)

    all_indices = list(range(p))
    all_indices.remove(xd)
    xs_indices = all_indices

    total_missing = int(round(n * p * missing_rate))
    missing_per_col = max(total_missing // len(xs_indices), 1)

    xd_col = data[:, xd]
    top_indices = np.argsort(xd_col)[-missing_per_col:]

    data_with_missing = data.copy()
    for xs in xs_indices:
        data_with_missing[top_indices, xs] = np.nan

    return data_with_missing
def mar_type8(data, missing_rate=0.1, seed=1):
    """
    MAR Type 8 (Global) — Extreme-value masking (two-sided).
    Injects missing values into all columns except xd by selecting rows with
    both highest and lowest values in xd.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Input dataset.

    missing_rate : float, default=0.1
        Total fraction of missing values across the dataset.

    seed : int, default=1
        Random seed for reproducibility.

    xd : int, optional
        Index of the determining column. If None, randomly selected.

    Returns
    -------
    data_with_missing : np.ndarray
        Dataset with missing values injected into rows corresponding to extreme values of xd.
    """
    rng = np.random.default_rng(seed)
    n, p = data.shape
    data = data.astype(float)

    xd = rng.integers(0, p)

    all_indices = list(range(p))
    all_indices.remove(xd)
    xs_indices = all_indices

    total_missing = int(round(n * p * missing_rate))
    missing_per_col = max(total_missing // len(xs_indices), 1)

    # Determine top and bottom half indices from xd
    xd_col = data[:, xd]
    sorted_indices = np.argsort(xd_col)

    if missing_per_col % 2 == 0:
        low_indices = sorted_indices[:missing_per_col // 2]
        high_indices = sorted_indices[-(missing_per_col // 2):]
    else:
        low_indices = sorted_indices[:missing_per_col // 2 + 1]
        high_indices = sorted_indices[-(missing_per_col // 2):]

    selected_indices = np.concatenate([low_indices, high_indices])

    data_with_missing = data.copy()
    for xs in xs_indices:
        data_with_missing[selected_indices, xs] = np.nan

    return data_with_missing

# Map type numbers to functions
MAR_TYPES = {
    1: mar_type1,
    2: mar_type2,
    3: mar_type3,
    4: mar_type4,
    5: mar_type5,
    6: mar_type6,
    7: mar_type7,
    8: mar_type8,
}