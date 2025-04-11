"""
A quick demo for MissMecha
===========================

This short example shows how to simulate a simple MAR pattern.
"""

# %%
# Your code block starts like this

import numpy as np
from missmecha.generator import MissMechaGenerator
...


#!/usr/bin/env python
# coding: utf-8

# ## Missingness Analysis & Imputation Evaluation Demo
# 
# This notebook demonstrates how to analyze missingness in a dataset and evaluate imputation quality using the `missmecha.analysis` modules.
# 
# We show:
# - Column-wise and overall missing rate analysis
# - Visual inspection of missing patterns
# - Evaluation of imputation quality using RMSE / Accuracy, depending on variable type
# 

# ### A Note on AvgERR
# 
# The idea behind `AvgERR` is to evaluate imputation performance based on variable types:
# 
# $
# \text{AvgErr}(v_j) =
# \begin{cases}
# \frac{1}{n} \sum\limits_{i=1}^{n} |X_{ij} - \hat{X}_{ij}|, & \text{if } v_j \text{ is continuous (MAE)} \\\\
# \sqrt{\frac{1}{n} \sum\limits_{i=1}^{n} (X_{ij} - \hat{X}_{ij})^2}, & \text{if } v_j \text{ is continuous (RMSE)} \\\\
# \frac{1}{n} \sum\limits_{i=1}^{n} (X_{ij} - \hat{X}_{ij})^2, & \text{if } v_j \text{ is continuous (MSE)} \\\\
# \frac{1}{n} \sum\limits_{i=1}^{n} \text{Acc}(X_{ij}, \hat{X}_{ij}), & \text{if } v_j \text{ is categorical}
# \end{cases}
# $
# 
# 
# In this implementation, if a `status` dictionary is provided, the function automatically applies the appropriate metric:
# - **Numerical columns** use the selected method (RMSE or MAE)
# - **Categorical/discrete columns** use classification accuracy

# ## Setup
# Import required packages and the evaluation function. We'll start by importing necessary packages and simulating a dataset with mixed-type variables and missing values.
# 

# In[1]:


import pandas as pd
import numpy as np
from missmecha.analysis import evaluate_imputation,compute_missing_rate


# ### Create fully observed mixed-type dataset
# 

# In[2]:


df_true = pd.DataFrame({
    "age": [25, 30, 22, 40, 35, 50],
    "income": [3000, 4500, 2800, 5200, 4100, 6000],
    "gender": ["M", "F", "M", "F", "F", "M"],
    "job_level": ["junior", "mid", "junior", "senior", "mid", "senior"]
})
df_true


# ### Inject missing values

# In[3]:


df_incomplete = df_true.copy()
df_incomplete.loc[1, "age"] = np.nan
df_incomplete.loc[2, "income"] = np.nan
df_incomplete.loc[3, "gender"] = np.nan
df_incomplete.loc[4, "job_level"] = np.nan
df_incomplete


# In[4]:


compute_missing_rate(df_incomplete)


# ### Impute missing values (integer mean for numeric, mode for categorical)

# In[6]:


df_filled = df_incomplete.copy()

for col in df_filled.columns:
    if df_filled[col].dtype.kind in "iufc":
        df_filled[col] = df_filled[col].fillna(round(df_filled[col].mean()))
    else:
        df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0])

df_filled


# ### Define variable types
# 

# In[7]:


status = {
    "age": "num",
    "income": "num",
    "gender": "cat",
    "job_level": "disc"
}


# ### Run `evaluate_imputation()` with AvgERR logi
# 
# 

# In[9]:


results = evaluate_imputation(
    ground_truth=df_true,
    filled_df=df_filled,
    incomplete_df=df_incomplete,
    method="mae",  # used for numerical columns
    status=status
)


# In[10]:


print("Column-wise scores:")
for k, v in results["column_scores"].items():
    print(f"  {k}: {v:.2f}")

print(f"\n Overall score: {results['overall_score']:.2f}")

