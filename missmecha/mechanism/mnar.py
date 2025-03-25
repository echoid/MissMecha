import pandas as pd
import numpy as np

class MNARGenerator:
    """基于阈值生成非随机缺失"""
    
    def __init__(self, target_col: str, threshold: float, direction: str = "above"):
        self.target_col = target_col
        self.threshold = threshold
        self.direction = direction
        
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        data_missing = data.copy()
        mask = data[self.target_col] > self.threshold if self.direction == "above" \
               else data[self.target_col] < self.threshold
        data_missing.loc[mask, self.target_col] = np.nan
        return data_missing
    






def make_mnar(X, percentile):
    # Copy the array to avoid altering the original one
    X_mnar = X.copy()
    percentile = percentile * 100
    # Iterate over each column in the array
    for col in range(X_mnar.shape[1]):
        # Calculate the percentile value for the current column
        threshold = np.percentile(X_mnar[:, col], percentile)

        # Replace values less than the threshold with np.nan
        X_mnar[:, col] = np.where(X_mnar[:, col] < threshold, np.nan, X_mnar[:, col])

    return X_mnar





def make_mnar_columnwise(data, col_info, q, random_seed=1):
    np.random.seed(random_seed)
    random.seed(random_seed)
    q = q * 100
    data_mnar = data.astype(float)

    missing_rates = {}

    for col, col_type in col_info.items():
        col_idx = int(col)  # Assuming the keys in `col_info` correspond to column indices
        num_to_remove = int(len(data_mnar) * q / 100)
        if "numerical" in col_type:
            # Calculate the percentile value for the numerical column
            threshold = np.percentile(data_mnar[:, col_idx], q)
            # Replace values less than the threshold with np.nan
            data_mnar[:, col_idx] = np.where(data_mnar[:, col_idx] < threshold, np.nan, data_mnar[:, col_idx])

            # Calculate the missing rate for this column
            missing_rate = np.mean(np.isnan(data_mnar[:, col_idx])) * 100
            missing_rates[col_idx] = missing_rate
            #print("numerical" ,missing_rate)

        elif "ordinal" in col_type:
            # Use the ordinal mapping from JSON to find the top two largest ordinal values
            ordinal_map = col_type['ordinal']
            max_value = max(ordinal_map.values())

            # Find the indices where the values in the column are greater than or equal to max_value - 1
            max_indices = np.where(data_mnar[:, col_idx] >= (max_value - 2))[0].tolist()

            # Find the rest of the indices (those not in max_indices)
            all_indices = set(range(data_mnar.shape[0]))
            other_indices = list(all_indices - set(max_indices))

            # Determine which indices to remove based on the number to remove
            if len(max_indices) >= num_to_remove:
                remove_indices = random.sample(max_indices, num_to_remove)
            else:
                # If there are not enough max_indices, take all max_indices and supplement with random others
                remove_indices = max_indices
                random_indices = random.sample(other_indices, num_to_remove - len(remove_indices))
                #remove_indices = remove_indices + random_indices

            data_mnar[remove_indices, col_idx] = np.nan

            # Calculate the missing rate for this column
            missing_rate = np.mean(np.isnan(data_mnar[:, col_idx])) * 100
            missing_rates[col_idx] = missing_rate
            #print("ordinal" ,missing_rate)

        elif "nominal" in col_type:
            # Nominal data: Randomly choose one category and make a portion of the data missing
            unique_vals = list(set(data_mnar[:, col_idx]))
            chosen_val = random.choice(unique_vals)

            # Get indices of the chosen category
            chosen_indices = np.where(data_mnar[:, col_idx] == chosen_val )[0].tolist()


            # Find the rest of the indices (those not in max_indices)
            all_indices = set(range(data_mnar.shape[0]))
            other_indices = list(all_indices - set(chosen_indices))

            # Determine which indices to remove based on the number to remove
            if len(chosen_indices) >= num_to_remove:
                remove_indices = random.sample(chosen_indices, num_to_remove)
            else:
                # If there are not enough max_indices, take all max_indices and supplement with random others
                remove_indices = chosen_indices
                random_indices = random.sample(other_indices, num_to_remove - len(remove_indices))
                remove_indices = remove_indices + random_indices


            data_mnar[remove_indices, col_idx] = np.nan

            # Calculate the missing rate for this column
            missing_rate = np.mean(np.isnan(data_mnar[:, col_idx])) * 100
            #print("nominal",missing_rate)
            missing_rates[col_idx] = missing_rate

    return data_mnar