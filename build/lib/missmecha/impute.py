import numpy as np
import pandas as pd
from collections import defaultdict

class SimpleSmartImputer:    
    """
    A simple imputer for both numerical and categorical variables.

    This class automatically detects or accepts user-specified column types and fills
    missing values using mean (for numerical) or mode (for categorical) strategies.

    The interface supports scikit-learn-style methods: `fit`, `transform`, and `fit_transform`.

    Parameters
    ----------
    cat_cols : list of str, optional
        A list of column names to be treated as categorical. If None, types are inferred automatically.
    verbose : bool, default=True
        Whether to print out the imputation summary during `fit`.

    Examples
    --------
    >>> from missmecha.impute import SimpleSmartImputer
    >>> df = pd.DataFrame({'age': [25, np.nan, 30], 'gender': ['M', 'F', np.nan]})
    >>> imputer = SimpleSmartImputer()
    >>> df_imputed = imputer.fit_transform(df)
    """
    
    def __init__(self, cat_cols=None, verbose=True):

        self.categorical_cols = cat_cols
        self.verbose = verbose
        self.fill_values = {}
        self.col_types = {}  # 'numerical' or 'categorical'

    def _infer_column_types(self, df):
        inferred = {}
        for col in df.columns:
            if self.categorical_cols is not None:
                inferred[col] = 'categorical' if col in self.categorical_cols else 'numerical'
            elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                inferred[col] = 'categorical'
            else:
                inferred[col] = 'numerical'
        return inferred

    def fit(self, df):
        """
        Fit the imputer on the provided DataFrame.

        This method determines the fill values for each column based on the strategy:
        - Numerical columns: mean
        - Categorical columns: mode

        Parameters
        ----------
        df : pandas.DataFrame
            Input data to analyze and compute fill values from.

        Returns
        -------
        self : SimpleSmartImputer
            The fitted instance with `fill_values` set.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        self.col_types = self._infer_column_types(df)

        for col, col_type in self.col_types.items():
            if col_type == 'numerical':
                self.fill_values[col] = df[col].mean()
            else:
                self.fill_values[col] = df[col].mode(dropna=True)[0]

            if self.verbose:
                print(f"[{self.__class__.__name__}] Column '{col}' treated as {col_type}. Fill value = {self.fill_values[col]}")

        return self

    def transform(self, df):
        """
        Apply the learned fill values to transform the dataset.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataset to be imputed using values from `fit()`.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with missing values filled.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        df_filled = df.copy()
        for col in df.columns:
            if col in self.fill_values:
                df_filled[col] = df[col].fillna(self.fill_values[col])
        return df_filled

    def fit_transform(self, df):
        """
        Fit the imputer and transform the dataset in one step.

        Equivalent to calling `fit()` followed by `transform()`.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataset to be imputed.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with missing values filled.
        """
        return self.fit(df).transform(df)
