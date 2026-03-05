import pandas as pd
import numpy as np
from PIL import Image
import os

def save_images_from_csv(csv_file, directory):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Extract the Intensity Values, Image File path, and Label
        intensity_values = np.array(eval(row['Intensity Values']), dtype=np.uint8)
        image_file = row['Image File']
        label = row['Label']

        # Ensure the intensity values are reshaped into the correct image width (512) and height (1)
        intensity_values = intensity_values.reshape((1, 512))

        # Extend the height from 1 pixel to 50 pixels by repeating the rows
        intensity_values = np.repeat(intensity_values, 50, axis=0)

        # Create an image from the intensity values
        img = Image.fromarray(intensity_values)

        # Generate the filename
        filename = f"{label}_{image_file.replace(',', '').replace('/', '_')}_FA.png"

        # Save the image
        img.save(directory + filename)
        # print(f"Saved: {filename}")

# Example usage
directory = 'output (split_2)/finetune_resnet50_pretraining_swav_batch_32_lr_0.001_epochs_100_seed_0/checkpoint/FA_GradCAM_Match/FA/'
csv_file = directory + 'trainUv.csv'  # Replace with your actual CSV file path
save_images_from_csv(csv_file, directory)