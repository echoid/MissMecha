

Missingness Analysis & Imputation Evaluation Demo
-------------------------------------------------

This notebook demonstrates how to analyze missingness in a dataset and
evaluate imputation quality using the ``missmecha.analysis`` modules.

We show: - Column-wise and overall missing rate analysis - Visual
inspection of missing patterns - Evaluation of imputation quality using
RMSE / Accuracy, depending on variable type

A Note on AvgERR
~~~~~~~~~~~~~~~~

The idea behind ``AvgERR`` is to evaluate imputation performance based
on variable types:

$ :raw-latex:`\text{AvgErr}`(v_j) =

.. raw:: latex

   \begin{cases}
   \frac{1}{n} \sum\limits_{i=1}^{n} |X_{ij} - \hat{X}_{ij}|, & \text{if } v_j \text{ is continuous (MAE)} \\\\
   \sqrt{\frac{1}{n} \sum\limits_{i=1}^{n} (X_{ij} - \hat{X}_{ij})^2}, & \text{if } v_j \text{ is continuous (RMSE)} \\\\
   \frac{1}{n} \sum\limits_{i=1}^{n} (X_{ij} - \hat{X}_{ij})^2, & \text{if } v_j \text{ is continuous (MSE)} \\\\
   \frac{1}{n} \sum\limits_{i=1}^{n} \text{Acc}(X_{ij}, \hat{X}_{ij}), & \text{if } v_j \text{ is categorical}
   \end{cases}

$

In this implementation, if a ``status`` dictionary is provided, the
function automatically applies the appropriate metric: - **Numerical
columns** use the selected method (RMSE or MAE) - **Categorical/discrete
columns** use classification accuracy

Setup
~~~~~~~~~~~~~~~~

Import required packages and the evaluation function. We’ll start by
importing necessary packages and simulating a dataset with mixed-type
variables and missing values.

.. code:: ipython3

    import pandas as pd
    import numpy as np
    from missmecha.analysis import evaluate_imputation,compute_missing_rate

Create fully observed mixed-type dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    df_true = pd.DataFrame({
        "age": [25, 30, 22, 40, 35, 50],
        "income": [3000, 4500, 2800, 5200, 4100, 6000],
        "gender": ["M", "F", "M", "F", "F", "M"],
        "job_level": ["junior", "mid", "junior", "senior", "mid", "senior"]
    })
    df_true




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>age</th>
          <th>income</th>
          <th>gender</th>
          <th>job_level</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>25</td>
          <td>3000</td>
          <td>M</td>
          <td>junior</td>
        </tr>
        <tr>
          <th>1</th>
          <td>30</td>
          <td>4500</td>
          <td>F</td>
          <td>mid</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22</td>
          <td>2800</td>
          <td>M</td>
          <td>junior</td>
        </tr>
        <tr>
          <th>3</th>
          <td>40</td>
          <td>5200</td>
          <td>F</td>
          <td>senior</td>
        </tr>
        <tr>
          <th>4</th>
          <td>35</td>
          <td>4100</td>
          <td>F</td>
          <td>mid</td>
        </tr>
        <tr>
          <th>5</th>
          <td>50</td>
          <td>6000</td>
          <td>M</td>
          <td>senior</td>
        </tr>
      </tbody>
    </table>
    </div>



Inject missing values
~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    df_incomplete = df_true.copy()
    df_incomplete.loc[1, "age"] = np.nan
    df_incomplete.loc[2, "income"] = np.nan
    df_incomplete.loc[3, "gender"] = np.nan
    df_incomplete.loc[4, "job_level"] = np.nan
    df_incomplete




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>age</th>
          <th>income</th>
          <th>gender</th>
          <th>job_level</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>25.0</td>
          <td>3000.0</td>
          <td>M</td>
          <td>junior</td>
        </tr>
        <tr>
          <th>1</th>
          <td>NaN</td>
          <td>4500.0</td>
          <td>F</td>
          <td>mid</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.0</td>
          <td>NaN</td>
          <td>M</td>
          <td>junior</td>
        </tr>
        <tr>
          <th>3</th>
          <td>40.0</td>
          <td>5200.0</td>
          <td>NaN</td>
          <td>senior</td>
        </tr>
        <tr>
          <th>4</th>
          <td>35.0</td>
          <td>4100.0</td>
          <td>F</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>5</th>
          <td>50.0</td>
          <td>6000.0</td>
          <td>M</td>
          <td>senior</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    compute_missing_rate(df_incomplete)


.. parsed-literal::

    Overall missing rate: 16.67%
    4 / 24 total values are missing.
    
    Top variables by missing rate:



.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>n_missing</th>
          <th>missing_rate (%)</th>
          <th>n_unique</th>
          <th>dtype</th>
          <th>n_total</th>
        </tr>
        <tr>
          <th>column</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>age</th>
          <td>1</td>
          <td>16.67</td>
          <td>5</td>
          <td>float64</td>
          <td>6</td>
        </tr>
        <tr>
          <th>income</th>
          <td>1</td>
          <td>16.67</td>
          <td>5</td>
          <td>float64</td>
          <td>6</td>
        </tr>
        <tr>
          <th>gender</th>
          <td>1</td>
          <td>16.67</td>
          <td>2</td>
          <td>object</td>
          <td>6</td>
        </tr>
        <tr>
          <th>job_level</th>
          <td>1</td>
          <td>16.67</td>
          <td>3</td>
          <td>object</td>
          <td>6</td>
        </tr>
      </tbody>
    </table>
    </div>




.. parsed-literal::

    {'report':            n_missing  missing_rate (%)  n_unique    dtype  n_total
     column                                                            
     age                1             16.67         5  float64        6
     income             1             16.67         5  float64        6
     gender             1             16.67         2   object        6
     job_level          1             16.67         3   object        6,
     'overall_missing_rate': 16.67}



Impute missing values (integer mean for numeric, mode for categorical)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    df_filled = df_incomplete.copy()
    
    for col in df_filled.columns:
        if df_filled[col].dtype.kind in "iufc":
            df_filled[col] = df_filled[col].fillna(round(df_filled[col].mean()))
        else:
            df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0])
    
    df_filled




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>age</th>
          <th>income</th>
          <th>gender</th>
          <th>job_level</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>25.0</td>
          <td>3000.0</td>
          <td>M</td>
          <td>junior</td>
        </tr>
        <tr>
          <th>1</th>
          <td>34.0</td>
          <td>4500.0</td>
          <td>F</td>
          <td>mid</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.0</td>
          <td>4560.0</td>
          <td>M</td>
          <td>junior</td>
        </tr>
        <tr>
          <th>3</th>
          <td>40.0</td>
          <td>5200.0</td>
          <td>M</td>
          <td>senior</td>
        </tr>
        <tr>
          <th>4</th>
          <td>35.0</td>
          <td>4100.0</td>
          <td>F</td>
          <td>junior</td>
        </tr>
        <tr>
          <th>5</th>
          <td>50.0</td>
          <td>6000.0</td>
          <td>M</td>
          <td>senior</td>
        </tr>
      </tbody>
    </table>
    </div>



Define variable types
~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    status = {
        "age": "num",
        "income": "num",
        "gender": "cat",
        "job_level": "disc"
    }


Run ``evaluate_imputation()`` with AvgERR logi
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    results = evaluate_imputation(
        ground_truth=df_true,
        filled_df=df_filled,
        incomplete_df=df_incomplete,
        method="mae",  # used for numerical columns
        status=status
    )

.. code:: ipython3

    print("Column-wise scores:")
    for k, v in results["column_scores"].items():
        print(f"  {k}: {v:.2f}")
    
    print(f"\n Overall score: {results['overall_score']:.2f}")


.. parsed-literal::

    Column-wise scores:
      age: 4.00
      income: 1760.00
      gender: 0.00
      job_level: 0.00
    
     Overall score: 441.00



.. raw:: html

    <div class="sd-text-center" style="margin-top: 2em; display: flex; justify-content: center; gap: 1.5em; flex-wrap: wrap;">
        <a href="_static/demo2.py" download
           style="background: linear-gradient(to bottom, #fdf6c5, #f7e98d); padding: 1em 1.5em; border-radius: 10px;
                  font-family: monospace; font-weight: bold; color: #000; text-decoration: none;
                  box-shadow: 0 3px 6px rgba(0,0,0,0.1); display: inline-block;">
            Download Python source code: <code>demo2.py</code>
        </a>
        <a href="_static/demo2.ipynb" download
           style="background: linear-gradient(to bottom, #fdf6c5, #f7e98d); padding: 1em 1.5em; border-radius: 10px;
                  font-family: monospace; font-weight: bold; color: #000; text-decoration: none;
                  box-shadow: 0 3px 6px rgba(0,0,0,0.1); display: inline-block;">
            Download Jupyter notebook: <code>demo2.ipynb</code>
        </a>
    </div>

