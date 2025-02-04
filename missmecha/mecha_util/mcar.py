





def make_mcar(data, missing_rate=0.1, seed=1):
    # Convert the data to float if it's not already, so it can hold NaN values
    data = data.astype(float)
    
    total_elements = data.size
    missing_elements = int(total_elements * missing_rate)

    # Create a copy of the data to avoid modifying the original array
    data_with_missing = data.copy()

    np.random.seed(seed)
    mask_indices = np.random.choice(total_elements, missing_elements, replace=False)

    # Convert flat indices to multi-dimensional indices
    multi_indices = np.unravel_index(mask_indices, data.shape)

    # Set selected elements to NaN
    data_with_missing[multi_indices] = np.nan

    return data_with_missing
