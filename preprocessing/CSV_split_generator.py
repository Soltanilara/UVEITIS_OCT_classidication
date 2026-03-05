import pandas as pd
import numpy as np
import random
import os

# Parameters for the number of samples in each split
n_val = 10  # number of validation samples
n_test = 10  # number of test samples

# Set the random seed for reproducibility
seedval = 20
np.random.seed(seedval)
random.seed(seedval)
os.environ['PYTHONHASHSEED'] = str(seedval)

# Load the CSV file
df = pd.read_csv('Annotation 01032025.csv')

# Extract the unique first parts of the paths
df['Prefix'] = df['Image File'].apply(lambda x: x.split('/')[0])

# Find unique prefixes
unique_prefixes = df['Prefix'].unique()

# Sort the unique prefixes to ensure consistent order before sampling
unique_prefixes = np.sort(unique_prefixes)

# Check if there are enough prefixes to sample for validation and test
if len(unique_prefixes) < (n_val + n_test):
    raise ValueError("Not enough unique prefixes to sample for validation and test splits. Reduce n_val or n_test.")

# Sample for validation and test using deterministic random choice
val_prefixes = np.random.choice(unique_prefixes, size=n_val, replace=False)
test_prefixes = np.random.choice(list(set(unique_prefixes) - set(val_prefixes)), size=n_test, replace=False)

# Remaining for train
train_prefixes = list(set(unique_prefixes) - set(val_prefixes) - set(test_prefixes))

# Sort the prefixes to ensure order stability for reproducibility
train_prefixes.sort()
val_prefixes.sort()
test_prefixes.sort()

# Split the dataframe based on the sampled prefixes
train_df = df[df['Prefix'].isin(train_prefixes)].copy()
val_df = df[df['Prefix'].isin(val_prefixes)].copy()
test_df = df[df['Prefix'].isin(test_prefixes)].copy()

# Now safely remove the Prefix column
train_df.drop('Prefix', axis=1, inplace=True)
val_df.drop('Prefix', axis=1, inplace=True)
test_df.drop('Prefix', axis=1, inplace=True)

if not os.path.exists('split_'+str(seedval)):
    os.makedirs('split_'+str(seedval))

# Save the dataframes to CSV files
train_df.to_csv('split_'+str(seedval)+'/train.csv', index=False)
val_df.to_csv('split_'+str(seedval)+'/val.csv', index=False)
test_df.to_csv('split_'+str(seedval)+'/test.csv', index=False)

# Count the number of each unique entry in the second column 'Label' of train_df

label_counts = test_df['Label'].value_counts()
print("Label counts in test_df:")
print(label_counts)

label_counts = df['Label'].value_counts()
print("Label counts in filtered_data:")
print(label_counts)

label_counts = train_df['Label'].value_counts()
print("Label counts in train_df:")
print(label_counts)
print('seed: ', seedval, 2*label_counts['negative']/sum(label_counts), 2-2*label_counts['negative']/sum(label_counts))
np.save('split_'+str(seedval)+'/classWeights',np.array([2-2*label_counts['negative']/sum(label_counts),2*label_counts['negative']/sum(label_counts)]))
print(val_prefixes,test_prefixes)

# Define the label categories
label_categories = ["negative", "mild", "moderate", "severe"]

# Initialize the weights array
weights = np.zeros(4)

# Calculate weights inversely proportional to the label frequency
total_samples = len(train_df)
for i, label in enumerate(label_categories):
    if label in label_counts:
        weights[i] = total_samples / label_counts[label]
    else:
        weights[i] = 0

# Normalize the weights
weights = weights / weights.sum()

# Save the weights array as gradedWeights.npy
np.save('split_'+str(seedval)+'/gradedWeights', weights)