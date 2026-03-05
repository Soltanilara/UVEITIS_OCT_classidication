import numpy as np
import pandas as pd

for seed in range(1,13):

    # Load the CSV file
    df = pd.read_csv('split_'+str(seed)+'/train.csv')

    # Define the label categories
    label_categories = ["negative", "mild", "moderate", "severe"]

    # Count occurrences of each label
    label_counts = df['Label'].value_counts()

    # Initialize the weights array
    weights = np.zeros(4)

    # Calculate weights inversely proportional to the label frequency
    total_samples = len(df)
    for i, label in enumerate(label_categories):
        if label in label_counts:
            weights[i] = total_samples / label_counts[label]
        else:
            weights[i] = 0

    # Normalize the weights
    weights = weights / weights.sum()

    # Save the weights array as gradedWeights.npy
    np.save('split_'+str(seed)+'/gradedWeights', weights)

    print("Weights array saved as 'gradedWeights.npy'")