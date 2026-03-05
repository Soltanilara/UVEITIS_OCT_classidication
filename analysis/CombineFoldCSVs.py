import os
import pandas as pd

# Set the parent directory containing fold_0 to fold_9
parent_dir = "CrossValidationOutputs"  # change this if needed

# Initialize list to hold DataFrames
dfs = []

# Process folds in order
for i in range(10):
    fold_name = f"fold{i}"
    fold_path = os.path.join(parent_dir, fold_name)
    csv_path = os.path.join(fold_path, "test_probabilities.csv")

    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        dfs.append(df)
    else:
        print(f"Warning: {csv_path} does not exist and will be skipped.")

# Concatenate all dataframes
if dfs:
    aggregated_df = pd.concat(dfs, ignore_index=True)
    output_path = os.path.join(parent_dir, "test_probabilities_aggregated_folds.csv")
    aggregated_df.to_csv(output_path, index=False)
    print(f"Aggregated CSV saved to: {output_path}")
else:
    print("No CSV files found to aggregate.")
