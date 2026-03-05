import pandas as pd
import numpy as np
import random
import os

# --- Parameters ---
n_folds = 10  # Number of cross-validation folds
n_val = 10  # Number of patients in the validation set for each fold
seedval = 42  # Random seed for reproducibility

# --- Fix all random seeds ---
np.random.seed(seedval)
random.seed(seedval)
os.environ['PYTHONHASHSEED'] = str(seedval)

# --- Load data ---
df = pd.read_csv('Annotation 01032025.csv')
# Create a column with the patient prefix (e.g., patient ID)
df['Prefix'] = df['Image File'].apply(lambda x: x.split('/')[0])

# Get unique patients (prefixes) and sort them
unique_prefixes = np.sort(df['Prefix'].unique())
num_patients = len(unique_prefixes)

if num_patients < n_val + n_folds:
    raise ValueError("Not enough patients for the specified number of folds and validation size.")

# --- Shuffle the prefixes once (with the chosen seed) and then split into folds ---
shuffled_prefixes = np.random.permutation(unique_prefixes)
# Use np.array_split to partition into 10 parts as evenly as possible
folds = np.array_split(shuffled_prefixes, n_folds)


# A helper function to compute and save the classWeights.npy (for negative vs. not negative)
def save_class_weights(prefix_path, df_subset):
    """
    Saves class weights for binary classification of 'negative' vs 'non-negative'
    to classWeights.npy in the given prefix_path folder.
    """
    label_counts = df_subset['Label'].value_counts()
    total = label_counts.sum()

    # For safety, handle case if 'negative' is not in label_counts
    neg_count = label_counts.get('negative', 0)
    weight_negative = 2 * neg_count / total
    weight_positive = 2 - weight_negative  # for non-negative

    class_weights = np.array([weight_positive, weight_negative])
    np.save(os.path.join(prefix_path, 'classWeights.npy'), class_weights)


# A helper function to compute and save the gradedWeights.npy (for negative, mild, moderate, severe)
def save_graded_weights(prefix_path, df_subset):
    """
    Saves graded weights for labels = ['negative', 'mild', 'moderate', 'severe']
    to gradedWeights.npy in the given prefix_path folder.
    """
    label_categories = ["negative", "mild", "moderate", "severe"]
    label_counts = df_subset['Label'].value_counts()
    total_samples = len(df_subset)

    weights = np.zeros(len(label_categories), dtype=np.float32)
    for i, label in enumerate(label_categories):
        # If a label does not appear, set that weight to 0
        if label in label_counts:
            weights[i] = total_samples / label_counts[label]
        else:
            weights[i] = 0.0

    # Normalize the weights so they sum to 1
    if weights.sum() > 0:
        weights = weights / weights.sum()

    np.save(os.path.join(prefix_path, 'gradedWeights.npy'), weights)


# --- Create the 10 folds ---
for i in range(n_folds):
    # Identify the test set for this fold
    test_prefixes = folds[i]  # patients in the i-th fold
    # Remaining patients
    remaining_prefixes = np.concatenate([folds[j] for j in range(n_folds) if j != i])

    # Randomly select n_val patients from the remaining for validation
    val_prefixes = np.random.choice(remaining_prefixes, size=n_val, replace=False)

    # The rest are training patients
    train_prefixes = list(set(remaining_prefixes) - set(val_prefixes))

    # Sort them so that CSV entries are in a deterministic order
    test_prefixes = np.sort(test_prefixes)
    val_prefixes = np.sort(val_prefixes)
    train_prefixes = np.sort(train_prefixes)

    # Split the dataframe
    test_df = df[df['Prefix'].isin(test_prefixes)].copy()
    val_df = df[df['Prefix'].isin(val_prefixes)].copy()
    train_df = df[df['Prefix'].isin(train_prefixes)].copy()

    # Create train_final as train + val
    train_final_df = pd.concat([train_df, val_df], ignore_index=True)

    # Sort within each split by 'Image File' to keep them consistent
    test_df.sort_values(by='Image File', inplace=True)
    val_df.sort_values(by='Image File', inplace=True)
    train_df.sort_values(by='Image File', inplace=True)
    train_final_df.sort_values(by='Image File', inplace=True)

    # Drop the 'Prefix' column
    test_df.drop(columns=['Prefix'], inplace=True)
    val_df.drop(columns=['Prefix'], inplace=True)
    train_df.drop(columns=['Prefix'], inplace=True)
    train_final_df.drop(columns=['Prefix'], inplace=True)

    # Make a directory for this fold
    fold_dir = f'fold_{i}'
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    # Save CSV files
    test_df.to_csv(os.path.join(fold_dir, 'test.csv'), index=False)
    val_df.to_csv(os.path.join(fold_dir, 'val.csv'), index=False)
    train_df.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
    train_final_df.to_csv(os.path.join(fold_dir, 'train_final.csv'), index=False)

    # --- Save weights for train.csv ---
    save_class_weights(fold_dir, train_df)
    save_graded_weights(fold_dir, train_df)
    # Rename them so they don't get overwritten by the final set
    os.rename(os.path.join(fold_dir, 'classWeights.npy'),
              os.path.join(fold_dir, 'classWeights_train.npy'))
    os.rename(os.path.join(fold_dir, 'gradedWeights.npy'),
              os.path.join(fold_dir, 'gradedWeights_train.npy'))

    # --- Save weights for train_final.csv ---
    save_class_weights(fold_dir, train_final_df)
    save_graded_weights(fold_dir, train_final_df)
    # Rename them to distinguish them
    os.rename(os.path.join(fold_dir, 'classWeights.npy'),
              os.path.join(fold_dir, 'classWeights_final.npy'))
    os.rename(os.path.join(fold_dir, 'gradedWeights.npy'),
              os.path.join(fold_dir, 'gradedWeights_final.npy'))

print("10-fold cross-validation splits have been created successfully!")
