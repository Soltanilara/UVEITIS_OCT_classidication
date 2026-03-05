import os
import shutil
import pandas as pd

# Load the datasets
test_df = pd.read_csv('/home/amin/PycharmProjects/Uveitis/split_2/test.csv')
probabilities_df = pd.read_csv('test_probabilities.csv')

# Directory where images are stored
image_folder = 'GradCAM'

# Define new folder paths
folders = {
    'True Positive': os.path.join(image_folder, 'True Positive'),
    'False Positive': os.path.join(image_folder, 'False Positive'),
    'False Negative': os.path.join(image_folder, 'False Negative'),
    'True Negative': os.path.join(image_folder, 'True Negative')
}

# Create folders if they don't exist
for folder in folders.values():
    os.makedirs(folder, exist_ok=True)

# Process each image based on the provided conditions
for index, (img_info, prob) in enumerate(zip(test_df.iterrows(), probabilities_df.iterrows())):
    image_name = f"{index}.png"
    original_path = img_info[1]['Image File']
    label = img_info[1]['Label']
    true_label = prob[1]['True_Label']
    prob_positive = prob[1]['Prob_Positive']

    # Determine the classification
    if prob_positive < 0.5 and true_label == 0:
        classification = 'True Negative'
    elif prob_positive < 0.5 and true_label == 1:
        classification = 'False Negative'
    elif prob_positive >= 0.5 and true_label == 0:
        classification = 'False Positive'
    else:  # prob_positive >= 0.5 and true_label == 1
        classification = 'True Positive'

    # New image name and path
    new_image_name = f"{index}_{prob_positive}_{label}_{original_path.replace(',', '').replace('/', '_')}.png"
    new_image_path = os.path.join(folders[classification], new_image_name)

    # Full path of the current image
    current_image_path = os.path.join(image_folder, image_name)

    # Rename and move the image
    shutil.move(current_image_path, new_image_path)

print("Images have been processed and moved accordingly.")
