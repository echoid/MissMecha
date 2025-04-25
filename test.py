import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from missmecha.analysis import compute_missing_rate, MCARTest
from missmecha.generator import MissMechaGenerator

# Generate synthetic complete data
data_num = np.random.default_rng(1).normal(loc=0.0, scale=1.0, size=(1000, 10))
X_train, X_test = train_test_split(data_num, test_size=0.3, random_state=42)

# Define test cases across mechanisms and types
test_cases = [
    # {"name": "MCAR Type 1", "info": {range(10): {"mechanism": "mcar", "type": 1, "rate": 0.2}}},
    # {"name": "MCAR Type 2", "info": {range(10): {"mechanism": "mcar", "type": 2, "rate": 0.2}}},
    # {"name": "MCAR Type 3", "info": {range(10): {"mechanism": "mcar", "type": 3, "rate": 0.2}}},

    #{"name": "MAR Type 1", "info": {range(10): {"mechanism": "mar", "type": 1, "rate": 0.2, "para": 0.3, "depend_on":(1,2,3)}}},
    {"name": "MAR Type 2", "info": {range(10): {"mechanism": "mar", "type": 2, "rate": 0.2, "para": 0.3, "depend_on":(1,2,3)}}},
    {"name": "MAR Type 3", "info": {range(10): {"mechanism": "mar", "type": 3, "rate": 0.2, "para": 0.3, "depend_on":(1,2,3)}}},
    # {"name": "MAR Type 3", "info": {range(10): {"mechanism": "mar", "type": 3, "rate": 0.2}}},
    # {"name": "MAR Type 4", "info": {range(10): {"mechanism": "mar", "type": 4, "rate": 0.2}}},
    # {"name": "MAR Type 5", "info": {range(10): {"mechanism": "mar", "type": 5, "rate": 0.2}}},
    # {"name": "MAR Type 6", "info": {range(10): {"mechanism": "mar", "type": 6, "rate": 0.2}}},
    # {"name": "MAR Type 7", "info": {range(10): {"mechanism": "mar", "type": 7, "rate": 0.2}}},
    # {"name": "MAR Type 8", "info": {range(10): {"mechanism": "mar", "type": 8, "rate": 0.2}}},

    # {"name": "MNAR Type 1", "info": {range(10): {"mechanism": "mnar", "type": 1, "rate": 0.2}}},
    # {"name": "MNAR Type 2", "info": {range(10): {"mechanism": "mnar", "type": 2, "rate": 0.2}}},
    # {"name": "MNAR Type 3", "info": {range(10): {"mechanism": "mnar", "type": 3, "rate": 0.2}}},
    # {"name": "MNAR Type 4", "info": {range(10): {"mechanism": "mnar", "type": 4, "rate": 0.2}}},
    # {"name": "MNAR Type 5", "info": {range(10): {"mechanism": "mnar", "type": 5, "rate": 0.2}}},
    # {"name": "MNAR Type 6", "info": {range(10): {"mechanism": "mnar", "type": 6, "rate": 0.2}}},

]

# Step 1: Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
X = X.astype(float)
# X_train = pd.DataFrame(X, columns=[f"col{i}" for i in range(X.shape[1])])
# y_train = pd.Series(y, name="label")

# Step 2: Define test cases
test_cases = [
    {
        "name": "MAR",
        "info": {
            range(10): {
                "mechanism": "MNAR",
                "type": 3,
                "rate": 0.3,
                "depend_on": [2],
                "para":{"para":0.9}
            }
        }
    }
]

# Step 3: Run tests
results = []

for case in test_cases:
    print(f"\n=== Testing {case['name']} ===")

    generator = MissMechaGenerator(info=case["info"], seed=42)
    #generator = MissMechaGenerator(mechanism = "mar", mechanism_type=2, missing_rate=0.2, seed=42)
    generator.fit(X, y=y)  # always pass y, fallback handled inside
    #generator.fit(X)  # always pass y, fallback handled inside
    X_train_missing = generator.transform(X_train)

    print(generator.get_mask())

    print("\n[Missing Rate Summary]")
    print(compute_missing_rate(X_train_missing))

    p_value = MCARTest(method="little")(pd.DataFrame(X_train_missing))
    print(f"[Little's MCAR Test] p-value: {p_value:.6f}")

    results.append((case["name"], p_value))

# Step 4: Summary table
print("\n=== Summary of Little's MCAR Test Results ===")
summary_df = pd.DataFrame(results, columns=["Mechanism", "MCAR Test p-value"])
summary_df
