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
    MAR Type 4 — Label-driven MAR adapted from MCAR1univa.
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



# Map type numbers to functions
MAR_TYPES = {
    1: mar_type1,
    2: mar_type2,
    3: mar_type3,
}