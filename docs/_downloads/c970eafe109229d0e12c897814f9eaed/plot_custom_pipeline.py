"""
=========================================
A custom pipeline with more possibilities
=========================================

Earlier, we demonstrated how :class:`~pyampute.ampute.MultivariateAmputation` can be integrated in a scikit-learn pipeline (see `A quick example`_ and `Evaluating missing values with grid search and a pipeline`_).

It may be valuable to understand the impact of missing values in more detail. Therefore, we demonstrate how a ``CustomTransformer`` and ``CustomEstimator`` can be used to do a more thorough analysis. Not only will such analysis gain insights in the statistical problems of missing data (and some imputation methods), but it will also help you to create real-world and realistic missingness scenarios.

Another example, of a more systematic approach, can be found in `Schouten and Vink (2021)`_.

.. _`A quick example`: https://rianneschouten.github.io/pyampute/build/html/auto_examples/plot_easy_example.html
.. _`Evaluating missing values with grid search and a pipeline`: https://rianneschouten.github.io/pyampute/build/html/auto_examples/plot_simulation_pipeline.html
.. _`Schouten and Vink (2021)`: https://journals.sagepub.com/doi/full/10.1177/0049124118799376

"""

# Author: Rianne Schouten <https://rianneschouten.github.io/>

# %%
# Recap
#######
#
# Given is the following setting (from `Evaluating missing values with grid search and a pipeline`_):
#
# .. _`Evaluating missing values with grid search and a pipeline`: https://rianneschouten.github.io/pyampute/build/html/auto_examples/plot_simulation_pipeline.html

import numpy as np

m = 5
n = 10000

mean = np.repeat(5, m)
cor = 0.5
cov = np.identity(m)
cov[cov == 0] = cor
rng = np.random.default_rng()
compl_dataset = rng.multivariate_normal(mean, cov, n)

# %%
# As amputation parameter settings, we will vary the proportion, the mechanism and the ``score_to_probability_func``. Since in  the latter have to be specified within the same dictionary, we define the parameters for the grid search as follows.
#

import itertools as it

mechs = ["MCAR", "MAR", "MNAR"]
funcs = ["sigmoid-right", "sigmoid-mid"]

parameters = {
    "amputation__prop": [0.1, 0.5, 0.9],
    "amputation__patterns": [
        [{"incomplete_vars": [0,1], "mechanism": mechanism, "score_to_probability_func": func}]
        for mechanism, func in list(it.product(mechs, funcs))]
}

# %%
# A transformer that drops incomplete rows
##########################################
#
# Previously, we evaluated the ``SimpleImputer`` class from scikit-learn. Another good way to evaluate the effect of missing values, is by analyzing the incomplete dataset directly. Since most prediction and analysis models do not accept missing values, we apply the `dropna` or `listwise deletion` or `complete case analysis` method (all names refer to the same strategy). To allow for integration in a pipeline, we set up a custom ``TransformerMixin``.
#

from sklearn.base import TransformerMixin

class DropTransformer(TransformerMixin):

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        self.X = X
        
        return self

    def transform(self, X, y=None):

        # drop incomplete rows
        Xp = pd.DataFrame(X)
        Xdrop = Xp.dropna().to_numpy()
		
        return Xdrop

# %%
# A custom estimator
####################
#
# Almost all, if not all, estimators and evaluation metrics in scikit-learn are aimed at prediction or classification. That is what most people want to do.
#
# However, for evaluating the effect of missing values on your model, it may be good to look further than just the prediction or classification accuracy. In this example, we will focus on the center of the distribution of one feature and evaluate the bias in that distribution.
#
# That could work as follows.
#

from sklearn.base import BaseEstimator 

class CustomEstimator(BaseEstimator):

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        self.X = X
        
        return self

    def predict(self, X):

        # return values of first feature
        values_used_for_score = X[:,0]
		
        return values_used_for_score

def my_evaluation_metric(y_true, y_pred):

    m1 = np.mean(y_true)
    m2 = np.mean(y_pred)

    bias = np.abs(m1 - m2)

    return bias

# %%
# An evaluation pipeline
########################
#
# As can be seen, the ``predict`` function returns the first feature of the transformed dataset. The evaluation metric then calculated the mean difference between that feature, and the truth.
#
# In our experiment, the complete dataset is the ground truth and we evaluate the impact of several missing data models (and imputation models) on that truth. 
#
# We then run the pipeline twice.
#

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from pyampute.ampute import MultivariateAmputation
from sklearn.metrics import make_scorer

# %%
# Once with the DropTransformer

steps = [('amputation', MultivariateAmputation()), ('imputation', DropTransformer()), ('estimator', CustomEstimator())]
pipe = Pipeline(steps)
grid = GridSearchCV(
    estimator=pipe,
    param_grid=parameters,
    scoring=make_scorer(my_evaluation_metric),
)

grid.fit(compl_dataset, np.zeros(len(compl_dataset)))
grid.score(compl_dataset, compl_dataset[:,0])
results_drop = pd.DataFrame(grid.cv_results_)

# %%
# Once with the SimpleImputer

steps = [('amputation', MultivariateAmputation()), ('imputation', SimpleImputer()), ('estimator', CustomEstimator())]
pipe = Pipeline(steps)
grid = GridSearchCV(
    estimator=pipe,
    param_grid=parameters,
    scoring=make_scorer(my_evaluation_metric),
)

grid.fit(compl_dataset, np.zeros(len(compl_dataset)))
grid.score(compl_dataset, compl_dataset[:,0])
results_mean = pd.DataFrame(grid.cv_results_)

# %%
# Comparison
############
#

res_drop = results_drop[['param_amputation__patterns', 'param_amputation__prop', 'mean_test_score']]
res_mean = results_mean[['param_amputation__patterns', 'param_amputation__prop', 'mean_test_score']]

res_drop.columns = ['mechanism, func', 'prop', 'score']
res_mean.columns = ['mechanism, func', 'prop', 'score']

res_drop

# %%

res_mean

# %%
#
# What you find here, is that a MCAR mechanism will not affect the center of the distribution of the first feature much, independent of the proportion of incomplete rows. 
# 
# A MAR mechanism with a sigmoid-right probability function will, on average, remove the right-hand side of the distribution (also, because there is a positive correlation between the observed data and the first feature). Therefore, the larger the proportion, the more bias. However, with a sigmoid-mid probability function, values in the center of the distribution of the first feature are removed, and there is therefore not much effect on the bias. 
#
# The same logic applies to MNAR missingness, but since MNAR missingness does not depend on the size of the correlation between observed data and incomplete data, the bias will be stronger.
#
# `Schouten and Vink (2021)`_ further discuss this topic and the effect of multiple imputation (which can be performed using scikit-learn's IterativeImputer).
#
# SimpleImputer will use the mean of the observed data in the first feature. Therefore, in case there is any bias, that bias will remain. In case there is no bias, mean imputation will distort the correlation structure with other features. But that is another story...
#
# .. _`Schouten and Vink (2021)`: https://journals.sagepub.com/doi/full/10.1177/0049124118799376

