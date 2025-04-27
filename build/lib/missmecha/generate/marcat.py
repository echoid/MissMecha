import numpy as np
import pandas as pd

class MARCatType1:
    """
    MAR Mechanism - Categorical Type 1 (Category-Conditioned Row-Wise Masking)

    Simulates Missing At Random (MAR) by introducing missingness **across rows**, conditioned
    on the value of a categorical feature. Each category is assigned a random masking
    probability such that the total missingness rate approximately matches `missing_rate`.

    This mechanism is particularly suitable for simulating structured row-wise missingness
    in tabular data with labeled groups or strata.

    Parameters
    ----------
    missing_rate : float, default=0.1
        Target total missing rate across the dataset.
    seed : int, default=1
        Random seed for reproducibility.
    cat_column : str or None
        Name of the categorical column used to drive missingness. If None, a column
        is randomly selected from the input DataFrame during `fit()`.
    """

    def __init__(self, missing_rate=0.1, seed=1, cat_column=None):
        self.missing_rate = missing_rate
        self.seed = seed
        self.cat_column = cat_column
        self.fitted = False

    def _verbose(self, msg):
        print(f"[{self.__class__.__name__}] {msg}")

    def fit(self, X, y=None):
        """
        Fit the masking distribution conditioned on a categorical column.

        Assigns each category a masking probability proportional to a random draw,
        normalized to ensure that the total missing rate matches `missing_rate`.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame containing the categorical column.
        y : Ignored
            Included for interface compatibility.

        Returns
        -------
        self : MARCatType1
            Fitted object with learned class-based probabilities.
        """

        rng = np.random.default_rng(self.seed)

        if not isinstance(X, pd.DataFrame):
            raise ValueError("MARCatType1 currently requires pandas DataFrame input.")

        # 如果没指定类别列，随机选一列（假设是 categorical）
        if self.cat_column is None:
            self.cat_column = rng.choice(X.columns)
            self._verbose(f"No categorical column specified. Randomly selected '{self.cat_column}' as cat_column.")

        # 将选定列转成 category 类型（确保是离散的）
        if not pd.api.types.is_categorical_dtype(X[self.cat_column]):
            X[self.cat_column] = X[self.cat_column].astype("category")

        self.classes = X[self.cat_column].cat.categories
        probs = rng.uniform(0, 1, len(self.classes))
        probs = probs / probs.sum() * self.missing_rate  # normalize total missing_rate

        self.class_probs = dict(zip(self.classes, probs))
        self.fitted = True
        return self

    def transform(self, X):
        """
        Apply row-wise missingness based on category-conditioned probabilities.

        For each category in the chosen column, a subset of rows is randomly selected
        and all columns in those rows are masked (i.e., set to NaN).

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        X_missing : pd.DataFrame
            DataFrame with row-level missing values introduced.
        """

        if not self.fitted:
            raise RuntimeError("Call .fit() before transform().")

        X_missing = X.copy()
        rng = np.random.default_rng(self.seed)

        for cat, prob in self.class_probs.items():
            rows = X[self.cat_column] == cat
            mask = rng.random(size=rows.sum()) < prob
            X_missing.loc[rows, :] = X_missing.loc[rows, :].mask(mask[:, None])

        return X_missing



MARCAT_TYPES = {
    1: MARCatType1,
    # 2: MARType2,
    # 3: MARType3,
    # 4: MARType4,
    # 5: MARType5,
    # 6: MARType6,
    # 7: MARType7,
    # 8: MARType8
}