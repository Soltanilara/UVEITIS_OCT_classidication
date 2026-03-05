import pandas as pd

# Load the test and test_probabilities CSV files
folder_path = 'output_split_9/finetune_resnet50_pretraining_swav_batch_64_lr_0.001_epochs_50_seed_0/checkpoint/'
test_file_path = 'split_9/test.csv'  # Replace with the path to your test CSV file
test_probabilities_path = folder_path+'test_probabilities.csv'  # Replace with the path to your test_probabilities CSV file

test_data = pd.read_csv(test_file_path)
test_probabilities = pd.read_csv(test_probabilities_path)

# Add the 'Match' column with 'X' or ' ' based on matching logic
test_probabilities['Incorrects'] = test_probabilities.apply(
    lambda row: ' ' if ((row['Prob_Negative'] > row['Prob_Positive'] and row['True_Label'] == 0) or
                        (row['Prob_Positive'] > row['Prob_Negative'] and row['True_Label'] == 1)) else 'X',
    axis=1
)

# Rename columns for better readability
test_probabilities.rename(columns={
    'Prob_Negative': 'Negative Class Probability',
    'Prob_Positive': 'Positive Class Probability',
    'True_Label': 'True Label'
}, inplace=True)

# Insert the new columns into the test data after the 'Label' column
merged_data = pd.concat([
    test_data.iloc[:, :2],  # Up to and including the 'Label' column
    test_probabilities[['Incorrects', 'Negative Class Probability', 'Positive Class Probability', 'True Label']],
    test_data.iloc[:, 2:]  # The rest of the original columns
], axis=1)

# Save the new CSV file
output_path = folder_path+'test_with_incorrects_and_probabilities.csv'  # Replace with your desired output path
merged_data.to_csv(output_path, index=False)

print(f"Updated CSV saved to {output_path}")
