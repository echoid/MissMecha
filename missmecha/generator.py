import numpy as np
import pandas as pd
from .generate.mcar import MCAR_TYPES
from .generate.mar import MAR_TYPES
from .generate.mnar import MNAR_TYPES
import numpy as np
import pandas as pd


def format_output(data_with_nan, original_data):
    """
    Format output to ensure mask and DataFrame structure are preserved.

    Parameters
    ----------
    data_with_nan : np.ndarray
        Data with NaNs inserted
    original_data : np.ndarray or pd.DataFrame
        Original input data before masking

    Returns
    -------
    output : dict
        {
            "data": np.ndarray or pd.DataFrame,
            "mask_bool": np.ndarray or pd.DataFrame,
            "mask_int": np.ndarray or pd.DataFrame
        }
    """

    is_input_df = isinstance(original_data, pd.DataFrame)

    if is_input_df:
        col_names = original_data.columns
        index = original_data.index
    else:
        col_names = [f"col{i}" for i in range(data_with_nan.shape[1])]
        index = np.arange(data_with_nan.shape[0])

    mask_bool = ~np.isnan(data_with_nan)
    mask_int = mask_bool.astype(int)

    if is_input_df:
        return {
            "data": pd.DataFrame(data_with_nan, columns=col_names, index=index),
            "mask_bool": pd.DataFrame(mask_bool, columns=col_names, index=index),
            "mask_int": pd.DataFrame(mask_int, columns=col_names, index=index)
        }
    else:
        return {
            "data": data_with_nan,
            "mask_bool": mask_bool,
            "mask_int": mask_int
        }

def generate_missing(data, missing_type="mcar", type=1, missing_rate=0.1, info=None, seed=1):
    """
    Generate missing values in a dataset using a specified missing mechanism.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Input dataset.
    missing_type : str, optional
        One of 'mcar', 'mar', 'mnar'. Only used if `info` is None.
    type : int, optional
        Mechanism variant ID. Only used if `info` is None.
    missing_rate : float, optional
        Overall missing rate. Only used if `info` is None.
    info : dict, optional
        Column-wise control dictionary. If provided, will override other mechanism settings.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    result : dict
        {
            "data": np.ndarray or pd.DataFrame with NaNs,
            "mask_int": same type as input (0 = missing, 1 = present),
            "mask_bool": same type as input (False = missing, True = present)
        }
    """

    is_input_df = isinstance(data, pd.DataFrame)

    if is_input_df:
        col_names = data.columns
        index = data.index
        data_np = data.to_numpy().astype(float)
    elif isinstance(data, np.ndarray):
        data_np = data.astype(float)
        col_names = [f"col{i}" for i in range(data.shape[1])]
        index = np.arange(data.shape[0])
    else:
        raise TypeError("Input must be a pandas DataFrame or a NumPy array.")

    # --- Unified mechanism mode ---
    if info is None:
        assert missing_type in ["mcar", "mar", "mnar"], "Invalid missing_type."
        if missing_type == "mcar":
            assert type in MCAR_TYPES, f"MCAR type {type} not registered."
            generator_fn = MCAR_TYPES[type]
        elif missing_type == "mar":
            assert type in MAR_TYPES, f"MAR type {type} not registered."
            generator_fn = MAR_TYPES[type]
        elif missing_type == "mnar":
            assert type in MNAR_TYPES, f"MNAR type {type} not registered."
            generator_fn = MNAR_TYPES[type]

        data_with_nan = generator_fn(
            data_np, missing_rate=missing_rate, seed=seed)

    else:
        raise NotImplementedError("Column-wise missing generation (info-based) not yet implemented.")

    mask_bool = ~np.isnan(data_with_nan)
    mask_int = mask_bool.astype(int)

    if is_input_df:
        data_result = pd.DataFrame(data_with_nan, columns=col_names, index=index)
        mask_int_result = pd.DataFrame(mask_int, columns=col_names, index=index)
        mask_bool_result = pd.DataFrame(mask_bool, columns=col_names, index=index)
    else:
        data_result = data_with_nan
        mask_int_result = mask_int
        mask_bool_result = mask_bool

    return {
        "data": data_result,
        "mask_int": mask_int_result,
        "mask_bool": mask_bool_result,
    }

import numpy as np
import pandas as pd
MECHANISM_LOOKUP = {
    "mcar": MCAR_TYPES,
    "mar": MAR_TYPES,
    "mnar": MNAR_TYPES,
}
class MissMechaGenerator:
    def __init__(self, mechanism="MCAR", mechanism_type=1, missing_rate=0.2, seed=1, info=None):
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
        self._fitted = False
        self.label = None
        self.generator_map = {}
        self.generator_class = None
        self.col_names = None
        self.is_df = None
        self.index = None
        self.mask = None          # Binary mask: 1 = observed, 0 = missing
        self.bool_mask = None     # Boolean mask: True = observed, False = missing

        if not info:
            # fallback to default generator for entire dataset
            self.generator_class = MECHANISM_LOOKUP[self.mechanism][self.mechanism_type]

    def _resolve_columns(self, cols):
        """Resolve column names or indices depending on input type."""
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
        self.label = y
        self.is_df = isinstance(X, pd.DataFrame)
        self.col_names = X.columns.tolist() if self.is_df else [f"col{i}" for i in range(X.shape[1])]
        self.index = X.index if self.is_df else np.arange(X.shape[0])
        self.generator_map = {}

        if self.info is None:
            generator = self.generator_class(missing_rate=self.missing_rate, seed=self.seed)
            X_np = X.to_numpy() if self.is_df else X
            generator.fit(X_np, y = self.label)
            self.generator_map["global"] = generator
        else:
            for key, settings in self.info.items():
                cols = (key,) if isinstance(key, (str, int)) else key
                col_labels, col_idxs = self._resolve_columns(cols)

                mechanism = settings["mechanism"].lower()
                mech_type = settings["type"]
                rate = settings["rate"]
                depend_on = settings.get("depend_on", None)
                para = settings.get("parameter", None)
                label = settings.get("label", y)


                init_kwargs = {k: v for k, v in {
                        "missing_rate": rate,
                        "seed": self.seed,
                        "para": para,
                        "depend_on": depend_on
                    }.items() if v is not None}


                generator_cls = MECHANISM_LOOKUP[mechanism][mech_type]
                #generator = generator_cls(missing_rate=rate, seed=self.seed, para = para, depend_on = depend_on)
                generator = generator_cls(**init_kwargs)
                sub_X = X[list(col_labels)].to_numpy() if self.is_df else X[:, col_idxs]
                generator.fit(sub_X, y=label)
                self.generator_map[key] = generator

        self._fitted = True
        return self

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("Call .fit() before transform().")

        data = X.copy()
        data_array = data.to_numpy().astype(float) if self.is_df else data.astype(float)

        if self.info is None:
            generator = self.generator_map["global"]
            masked = generator.transform(data_array)

            # 保存 mask（直接从 transformed 的数据中反推）
            mask_array = ~np.isnan(masked)
            self.mask = mask_array.astype(int)
            self.bool_mask = mask_array

            return pd.DataFrame(masked, columns=self.col_names, index=self.index) if self.is_df else masked
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

            mask_array = ~np.isnan(data_array)
            self.mask = mask_array.astype(int)
            self.bool_mask = mask_array


            return data if self.is_df else data_array
        

    def get_mask(self):
        if self.mask is None:
            raise RuntimeError("Mask not available. Please call `transform()` first.")
        return self.mask

    def get_bool_mask(self):
        if self.bool_mask is None:
            raise RuntimeError("Boolean mask not available. Please call `transform()` first.")
        return self.bool_mask



# class MissMechaGenerator:
#     def __init__(self, mechanism="MAR", mechanism_type=1, missing_rate=0.2, seed=1, additional_params=None):
#         self.mechanism = mechanism.lower()
#         self.mechanism_type = int(mechanism_type)
#         self.missing_rate = missing_rate
#         self.seed = seed
#         self.additional_params = additional_params or {}
#         self._fitted = False
#         self.label = None
#         self.generator_fn = None

#         # Assign mechanism class/function
#         if self.mechanism == "mcar":
#             self.generator_class = MCAR_TYPES[self.mechanism_type]
#         elif self.mechanism == "mar":
#             self.generator_class = MAR_TYPES[self.mechanism_type]
#         elif self.mechanism == "mnar":
#             self.generator_class = MNAR_TYPES[self.mechanism_type]
#         else:
#             raise ValueError("Invalid mechanism type.")
        

#     def fit(self, X, y=None):
#         self.label = y
#         self._fitted = True

#         if self.mechanism == "mcar":
#             cls = MCAR_TYPES[self.mechanism_type]
#             self.generator_model = cls(missing_rate=self.missing_rate, seed=self.seed, **self.additional_params)
#             self.generator_model.fit(X, y)
            
#         elif self.mechanism == "mar":
#             cls = MAR_TYPES[self.mechanism_type]
#             self.generator_model = cls(missing_rate=self.missing_rate, seed=self.seed, **self.additional_params)
#             self.generator_model.fit(X, y)

#         elif self.mechanism == "mnar":
#             cls = MNAR_TYPES[self.mechanism_type]
#             self.generator_model = cls(missing_rate=self.missing_rate, seed=self.seed, **self.additional_params)
#             self.generator_model.fit(X, y)

#         else:
#             raise ValueError("Unsupported mechanism type")
#         return self


#     def transform(self, X):
#         if not self._fitted or self.generator_model is None:
#             raise RuntimeError("You must call .fit(X, y) before .transform(X)")

#         # 调用机制模型生成缺失数据

#         data_with_nan = self.generator_model.transform(X)

#         # 统一格式化输出
#         output = format_output(data_with_nan, X)

#         # 存储用于 get_mask()
#         self.data_with_nan = output["data"]
#         self.mask_bool = output["mask_bool"]
#         self.mask_int = output["mask_int"]

#         return output["data"]

#     def get_mask(self, format="bool"):
#         if format == "bool":
#             return self.mask_bool
#         elif format == "int":
#             return self.mask_int
#         else:
#             raise ValueError("format must be 'bool' or 'int'")


#     def fit_transform(self, X, y=None):
#         self.fit(X, y)
#         return self.transform(X)