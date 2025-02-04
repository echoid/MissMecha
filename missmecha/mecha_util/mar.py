



import torch
import numpy as np
from scipy import optimize

def fit_intercepts(X, coeffs, p, self_mask=False):
    if self_mask:
        d = len(coeffs)
        intercepts = torch.zeros(d)
        for j in range(d):
            def f(x):
                return torch.sigmoid(X * coeffs[j] + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = torch.zeros(d_na)
        
        # Ensure X and coeffs are in floating-point format
        X = X.float()
        coeffs = coeffs.float()

        for j in range(d_na):
            def f(x):
                return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p
            
            intercepts[j] = optimize.bisect(f, -50, 50)
    
    return intercepts

def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False):
    n, d = X.shape
    if self_mask:
        coeffs = torch.randn(d).float()  # Ensure coeffs are float
        Wx = X * coeffs
        coeffs /= torch.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = torch.randn(d_obs, d_na).float()  # Ensure coeffs are float

        # Convert indices to LongTensor for PyTorch operations
        idxs_obs = torch.tensor(idxs_obs, dtype=torch.long)
        idxs_nas = torch.tensor(idxs_nas, dtype=torch.long)

        # Ensure the data is in floating-point format
        X = X.float()  # Ensure X is a floating-point tensor

        # Perform operations
        Wx = X[:, idxs_obs].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)
    return coeffs

def make_mar(X, p, p_obs):
    n, d = X.shape

    to_torch = torch.is_tensor(X)  # Determine if X is a PyTorch tensor or a NumPy array
    if not to_torch:
        X = torch.from_numpy(X)

    # Initialize a boolean mask
    mask = torch.zeros(n, d).bool()

    # Calculate the number of observed variables and the number of potentially missing variables
    d_obs = max(int(p_obs * d), 1)
    d_na = d - d_obs

    # Select indices for observed and potentially missing variables
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    # Generate coefficients and intercepts for the logistic model
    coeffs = pick_coeffs(X, idxs_obs, idxs_nas)
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)

    # Ensure X, coeffs, and intercepts are floating-point tensors
    X = X.float()
    coeffs = coeffs.float()
    intercepts = intercepts.float()

    # Calculate the probabilities using the logistic model
    ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)

    # Generate random values and apply the mask
    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    # Apply the mask to X, setting the masked elements to NaN
    X[mask] = float('nan')

    # Convert back to numpy array if the input was a numpy array
    if not to_torch:
        X = X.numpy()

    return X



