import numpy as np
import pandas as pd
from .generate.mcar import MCAR_TYPES
from .generate.mar import MAR_TYPES
from .generate.mnar import MNAR_TYPES
from .generate.marcat import MARCAT_TYPES
from .generate.mnarcat import MNARCAT_TYPES
import numpy as np
import pandas as pd
from .util import safe_init
import numpy as np
import pandas as pd
MECHANISM_LOOKUP = {
    "mcar": MCAR_TYPES,
    "mar": MAR_TYPES,
    "mnar": MNAR_TYPES,
    "marcat": MARCAT_TYPES,
    "mnarcat":MNARCAT_TYPES
}
            
import warnings


class MissMechaGenerator:    
    """
    Flexible simulator for generating missing data under various mechanisms.

    This class serves as the central interface for simulating missing values using various predefined mechanisms.
    It supports both global and column-wise simulation, enabling fine-grained control over different missingness patterns, including MCAR, MAR, and MNAR.

    Parameters
    ----------
    mechanism : str, default="MCAR"
        The default missingness mechanism to use (if `info` is not specified).
    mechanism_type : int, default=1
        The subtype of the mechanism (e.g., MAR type 1, MNAR type 4).
    missing_rate : float, default=0.2
        Proportion of values to mask as missing (only if `info` is not provided).
    seed : int, default=1
        Random seed to ensure reproducibility.
    info : dict, optional
        Dictionary defining per-column missingness settings. Each key is a column
        or tuple of columns, and each value is a dict with fields like:
        - 'mechanism': str
        - 'type': int
        - 'rate': float
        - 'depend_on': list or str
        - 'para': dict of additional parameters
    cat_cols : list of str, optional
    List of columns treated as categorical variables. 
    Internally encoded into integers during missingness simulation.


    Examples
    --------
    >>> from missmecha.generator import MissMechaGenerator
    >>> import numpy as np
    >>> X = np.random.rand(100, 5)
    >>> generator = MissMechaGenerator(mechanism="mcar", mechanism_type=1, missing_rate=0.2)
    >>> X_missing = generator.fit_transform(X)

    """
    def __init__(self, mechanism="MCAR", mechanism_type=1, missing_rate=0.2, seed=1, info=None, cat_cols=None):
        """
        Multiple-mechanism generator. Uses 'info' dictionary for column-wise specification.

        Parameters
        ----------
        mechanism : str
            Default mechanism type (if info is not provided).
        mechanism_type : int
            Default mechanism subtype.
        missing_rate : float
            Default missing rate.
        seed : int
            Random seed.
        info : dict
            Column-specific missingness configuration.
        """
        self.mechanism = mechanism.lower()
        self.mechanism_type = int(mechanism_type)
        self.missing_rate = missing_rate
        self.seed = seed
        self.info = info
        #self.info = self._expand_info(info) if info is not None else None
        self._fitted = False
        self.label = None
        self.generator_map = {}
        self.cat_cols = cat_cols  
        self.cat_maps = {}  
        self.generator_class = None
        self.col_names = None
        self.is_df = None
        self.index = None
        self.mask = None          # Binary mask: 1 = observed, 0 = missing
        self.bool_mask = None     # Boolean mask: True = observed, False = missing

        if not info:
            # fallback to default generator for entire dataset
            self.generator_class = MECHANISM_LOOKUP[self.mechanism][self.mechanism_type]
            


        warnings.filterwarnings("ignore")


    def _resolve_columns(self, cols):
        """
        Resolve column names and indices based on input type.

        Parameters
        ----------
        cols : list, tuple, or range
            Column specification in either str or int format.

        Returns
        -------
        col_labels : list of str
            Column names.
        col_idxs : list of int
            Corresponding index positions.
        """
        if self.is_df:
            col_labels = list(cols)
            col_idxs = [self.col_names.index(c) for c in cols]
        else:
            if all(isinstance(c, int) for c in cols):
                col_labels = [f"col{c}" for c in cols]
                col_idxs = list(cols)
            elif all(isinstance(c, str) and c.startswith("col") for c in cols):
                col_idxs = [int(c.replace("col", "")) for c in cols]
                col_labels = list(cols)
            else:
                raise ValueError(f"Invalid column specification: {cols} for ndarray input")
        return col_labels, col_idxs

    def fit(self, X, y=None):
        """
        Fit the internal generators to the input dataset.

        This step prepares the missingness generators based on either global
        or column-specific configurations.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The complete input dataset.
        y : array-like, optional
            Label or target data (used for some MNAR or MAR configurations).

        Returns
        -------
        self : MissMechaGenerator
            Returns the fitted generator instance.
        """
        self.label = y
        self.is_df = isinstance(X, pd.DataFrame)
        self.col_names = X.columns.tolist() if self.is_df else [f"col{i}" for i in range(X.shape[1])]
        self.index = X.index if self.is_df else np.arange(X.shape[0])
        self.generator_map = {}
        
        # Handle categorical mapping
        if self.cat_cols:
            if not self.is_df:
                raise ValueError("Categorical handling requires DataFrame input.")

            self.cat_maps = {}  # {col_name: {int: str}}
            for col in self.cat_cols:
                unique_values = X[col].dropna().unique()
                value_to_int = {v: i for i, v in enumerate(sorted(unique_values))}
                int_to_value = {i: v for v, i in value_to_int.items()}
                self.cat_maps[col] = int_to_value  # Save inverse mapping

                # Replace original categorical values with numerical codes
                X[col] = X[col].map(value_to_int).astype(float)
        # Fallback: global generator
        if self.info is None:
            generator = self.generator_class(missing_rate=self.missing_rate, seed=self.seed)
            X_np = X.to_numpy() if self.is_df else X
            generator.fit(X_np, y=self.label)
            self.generator_map["global"] = generator

        # Column-wise generator using info
        else:
            for key, settings in self.info.items():
                cols = (key,) if isinstance(key, (str, int)) else key
                col_labels, col_idxs = self._resolve_columns(cols)

                mechanism = settings["mechanism"].lower()
                mech_type = settings["type"]
                rate = settings["rate"]
                depend_on = settings.get("depend_on", None)
                para = settings.get("para", {})
                if not isinstance(para, dict):
                    para = {"value": para}

                col_seed = self.seed + hash(str(key)) % 10000 if self.seed is not None else None

                init_kwargs = {
                    "missing_rate": rate,
                    "seed": col_seed,
                    "depend_on": depend_on,
                    **para
                }
                init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
                label = settings.get("label", y)

                generator_cls = MECHANISM_LOOKUP[mechanism][mech_type]
                generator = safe_init(generator_cls, init_kwargs)
                sub_X = X[list(col_labels)].to_numpy() if self.is_df else X[:, col_idxs]
                generator.fit(sub_X, y=label)
                self.generator_map[key] = generator

        self._fitted = True
        return self


    def transform(self, X):
        """
        Apply the fitted generators to introduce missing values.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The dataset to apply missingness to.

        Returns
        -------
        X_masked : same type as X
            Dataset with simulated missing values.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before transform().")

        data = X.copy()
        data_array = data.to_numpy().astype(float) if self.is_df else data.astype(float)

        if self.info is None:
            generator = self.generator_map["global"]
            masked = generator.transform(data_array)

            mask_array = ~np.isnan(masked)
            self.mask = mask_array.astype(int)
            self.bool_mask = mask_array
            
            if self.is_df:
                data = pd.DataFrame(masked, columns=self.col_names, index=self.index)
            else:
                return masked

        else:
            for key, generator in self.generator_map.items():
                cols = (key,) if isinstance(key, (str, int)) else key
                col_labels, col_idxs = self._resolve_columns(cols)
                sub_X = data[list(col_labels)].to_numpy() if self.is_df else data_array[:, col_idxs]
                masked = generator.transform(sub_X)

                if self.is_df:
                    for col in col_labels:
                        data[col] = masked[:, list(col_labels).index(col)].astype(float)
                else:
                    data_array[:, col_idxs] = masked


        # ✨ Convert categorical variables back to original string labels (if specified)
        if self.is_df and self.cat_cols:
            for col in self.cat_cols:
                if col in data.columns:
                    inverse_map = self.cat_maps.get(col)
                    if inverse_map:
                        data[col] = data[col].map(lambda x: inverse_map.get(int(x)) if pd.notna(x) else np.nan)

        mask_array = ~data.isna().to_numpy()  # ensure result is np.ndarray
        self.mask = mask_array.astype(int)
        self.bool_mask = mask_array

        return data if self.is_df else data_array

        
    def _expand_info(self, info):
        """
        Expand group-style `info` dict into one-entry-per-column format.

        Parameters
        ----------
        info : dict
            Original `info` mapping, possibly with multiple-column keys.

        Returns
        -------
        new_info : dict
            Expanded column-specific `info` dictionary.
        """
        new_info = {}
        for key, settings in info.items():
            if isinstance(key, (list, tuple, range)):
                for col in key:
                    new_info[col] = settings.copy()  # 每列一个 copy，避免共享引用
            else:
                new_info[key] = settings
        return new_info
        

    def get_mask(self):
        """
        Return the latest binary mask generated by `transform()`.

        Returns
        -------
        mask : np.ndarray
            Binary array where 1 = observed, 0 = missing.
        """
        if self.mask is None:
            raise RuntimeError("Mask not available. Please call `transform()` first.")
        return self.mask

    def get_bool_mask(self):
        """
        Return the latest boolean mask generated by `transform()`.

        Returns
        -------
        bool_mask : np.ndarray
            Boolean array where True = observed, False = missing.
        """
        if self.bool_mask is None:
            raise RuntimeError("Boolean mask not available. Please call `transform()` first.")
        return self.bool_mask
    def fit_transform(self, X, y=None):
        """
        Fit the generator and apply the transformation in a single step.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The complete input dataset.
        y : array-like, optional
            Label or target data (used for some MNAR or MAR configurations).

        Returns
        -------
        X_masked : same type as X
            Dataset with simulated missing values.
        """
        self.fit(X, y)
        return self.transform(X)


