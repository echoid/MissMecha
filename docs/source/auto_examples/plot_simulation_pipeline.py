"""
=========================================================
Evaluating missing values with grid search and a pipeline
=========================================================

Generating missing values in a complete dataset (we call this `amputation`) seems like a bizarre thing to do. However, most people who work with all sorts of data will acknowledge that missing data is widespread and can be a severe issue for various types of analyses and models. In order to understand the effect of missing values and to know which missing data methods are appropriate in which situation, we perform simulation studies. And for that, we need amputation. 

With package ``pyampute``, we provide the multivariate amputation methodology proposed by `Schouten et al. (2018)`_. Because our :class:`~pyampute.ampute.MultivariateAmputation` class follows scikit-learn's ``fit`` and ``transform`` paradigm, it is straightforward to design a missing data experiment. 

Here, we demonstrate how that works.

.. _`Schouten et al. (2018)`: https://www.tandfonline.com/doi/full/10.1080/00949655.2018.1491577
"""

# Author: Rianne Schouten <https://rianneschouten.github.io/>

# %%
# General experimental setup
############################
#
# In general, evaluating the effect of missing values is done in four steps:
#
# 1. Generate or import a complete dataset
# 2. Ampute the dataset
# 3. Impute the dataset
# 4. Compare the performance of a model between the datasets in step 1, 2 and 3.
#
# It is often wise to first inspect the effect of amputation (by comparing the datasets in steps 1 and 2) before comparing with step 3. Let's get started.
#

# %%
# Complete dataset
##################
#
# A simulation starts with a complete dataset. Make sure that you use a dataset where variables are correlated with each other; otherwise it will not make sense to use a sophisticated amputation algorithm (see `Schouten et al. (2021)`_ for a discussion on this topic).
#
# .. _`Schouten et al. (2021)`: https://journals.sagepub.com/doi/full/10.1177/0049124118799376

import numpy as np

m = 5
n = 1000

mean = np.repeat(5, m)
cor = 0.5
cov = np.identity(m)
cov[cov == 0] = cor
rng = np.random.default_rng()
compl_dataset = rng.multivariate_normal(mean, cov, n)

# %%
# Multivariate amputation
#########################
#
# Vary the parameters of the amputation procedure. Read the `documentation`_ or `this blogpost`_ to understand how you can tune the parameters such that you create varying types of missingness.
#
# As an example, here, we generate `one` missing data pattern with missing values in the `first two variables`: ``"incomplete_vars":[0,1]``. We vary the proportion of incomplete rows between 0.1 and 0.9.
#
# We furthermore experiment with the three mechanisms: Missing Completely At Random (MCAR), Missing At Random (MAR) and Missing Not At Random (MNAR) (cf. `Rubin (1976)`_).
#
# .. _`documentation`: https://rianneschouten.github.io/pyampute/build/html/pyampute.ampute.html
# .. _`this blogpost`: https://rianneschouten.github.io/pyampute/build/html/mapping.html
# .. _`Rubin (1976)`: https://www.jstor.org/stable/2335739

parameters = {
    "amputation__prop": [0.1, 0.5, 0.9],
    "amputation__patterns": [
        [{"incomplete_vars": [0, 1], "mechanism": "MCAR"}],
        [{"incomplete_vars": [0, 1], "mechanism": "MAR"}],
        [{"incomplete_vars": [0, 1], "mechanism": "MNAR"}],
    ],
}

# %%
# Missing data methods
######################
#
# `SimpleImputer`_ is a univariate, single imputation method that is commonly used. However, in case of MCAR missingness, it distorts the relation with other variables, and in case of MAR and MNAR missingness it will not resolve issues with shifted variable distributions (see `Van Buuren (2018)`_). It may be better to use a method such as `IterativeImputer`_.
#
# Yet, to demonstrate the working of a simulation pipeline, we will work with SimpleImputer for now.
#
# .. _`SimpleImputer`: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
# .. _`Van Buuren (2018)`: https://stefvanbuuren.name/fimd/
# .. _`IterativeImputer`: https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html

parameters["imputation__strategy"] = ["mean"]

# %%
# Evaluation
############
#
# How you wish to evaluate the amputation and imputation greatly depends on the goal of your model. We will first show the experiment for a LinearRegression estimator, using predictors and an outcome feature.
#
# We recommend to read `A custom pipeline with more possibilities`_ to see how custom ``BaseEstimator``'s and ``TransformerMixin``'s can be used to gain a deeper understanding of the impact of missing values.
#
# .. _`A custom pipeline with more possibilities`: https://rianneschouten.github.io/pyampute/build/html/auto_examples/plot_custom_pipeline.html

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from pyampute.ampute import MultivariateAmputation

steps = [
    ("amputation", MultivariateAmputation()),
    ("imputation", SimpleImputer()),
    ("estimator", LinearRegression()),
]
pipe = Pipeline(steps)
grid = GridSearchCV(
    estimator=pipe, param_grid=parameters, scoring=make_scorer(mean_squared_error),
)

X, y = compl_dataset[:, :-1], compl_dataset[:, -1]
X_compl_train, X_compl_test, y_compl_train, y_compl_test = train_test_split(
    X, y, random_state=2022
)

grid.fit(X_compl_train, y_compl_train)
grid.score(X_compl_test, y_compl_test)
results = pd.DataFrame(grid.cv_results_)

res = results[
    [
        "param_amputation__patterns",
        "param_amputation__prop",
        "param_imputation__strategy",
        "mean_test_score",
    ]
]
res.columns = ["mechanism", "prop", "imputation", "score"]
res
