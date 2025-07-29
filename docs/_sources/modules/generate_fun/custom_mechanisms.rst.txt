Custom Missing Mechanisms
==========================

MissMecha supports not only built-in mechanisms like MCAR, MAR, and MNAR, 
but also user-defined missingness generators for advanced use cases. 
This tutorial shows you how to define and apply your own mechanism class.

Why Custom?
-----------

Sometimes you want to simulate structured missingness patterns 
(e.g., temporal patterns, block-wise missingness, deep learning–based masking) 
that go beyond standard MCAR/MAR/MNAR models.  
With `MissMechaGenerator`, you can inject your own mechanism as long as it implements a compatible interface.

Basic Interface
---------------

A valid custom mechanism class must implement:

.. code-block:: python

   class MyMasker:
       def fit(self, X, y=None):
           # Prepare thresholds, mask logic, etc.
           return self

       def transform(self, X):
           # Apply masking and return masked X
           return X_masked

You can use either **global simulation** or **per-column configuration**.

Global Example
--------------

.. code-block:: python

   from missmecha.generator import MissMechaGenerator

   class MyMasker:
       def fit(self, X, y=None): ...
       def transform(self, X): ...

   gen = MissMechaGenerator(
       mechanism="custom",
       custom_class=MyMasker,
       missing_rate=0.3,
       seed=42
   )
   X_missing = gen.fit_transform(X)

Column-wise Example
-------------------

.. code-block:: python

   info = {
       ("col1", "col2"): {
           "mechanism": "custom",
           "custom_class": MyMasker,
           "rate": 0.4
       }
   }

   gen = MissMechaGenerator(info=info)
   X_missing = gen.fit_transform(X)

Debugging Tips
--------------

- If your custom class raises an error during `fit()` or `transform()`, MissMecha will raise it directly.
- Use `compute_missing_rate()` to check whether your mask performs as expected.
- You can use random seeds inside your custom class to ensure reproducibility.

See Also
--------

- :doc:`/modules/generate`
