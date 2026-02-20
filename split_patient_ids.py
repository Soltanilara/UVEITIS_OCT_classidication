import pandas as pd

# Load the CSV file
split = 10
val_or_test = 'train'
# val_or_test = 'val'
# val_or_test = 'test'
file_path = f'split_{str(split)}/{val_or_test}.csv'  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

# Extract unique patient IDs
data['Patient ID'] = data['Image File'].apply(lambda x: x.split('/')[0])
unique_patient_ids = data['Patient ID'].unique()

# Save unique patient IDs to a new CSV file
output_path = f'split_{str(split)}/{val_or_test}_patient_ids.csv'  # Replace with your desired output path
unique_patient_df = pd.DataFrame(unique_patient_ids, columns=['Patient ID'])
unique_patient_df.to_csv(output_path, index=False)

print(f"Unique patient IDs saved to {output_path}")
