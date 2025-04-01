import pandas as pd
import numpy as np
from missmecha.generate import generate_missing
from missmecha.analysis import compute_missing_rate
df_cate = pd.DataFrame({
    "age": [25, 30, np.nan, 40],
    "income": [3000, np.nan, 2800, 5200],
    "gender": ["M", "F", np.nan, "F"]
})


data_num = np.random.default_rng(1).normal(loc=0.0, scale=1.0, size=(1000, 10))


# Define info dictionary for generate()
info = {
    "rate": 0.2,  # 20% missingness
    "type": 1     # Use type_two generator
}

type_list = [5,6]
missing_rate_list = [0,0.1,0.5,0.9]

for type_val in type_list:
    for rate in missing_rate_list:
        print("======================")
        print("Missing type", type_val,"Missing Rate", rate)
        result = generate_missing(data_num, missing_type="mnar", type=type_val, missing_rate=rate)
        compute_missing_rate(result["data"])
        print(result["mask_int"])
        print("======================")
