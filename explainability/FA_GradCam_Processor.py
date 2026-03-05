import os
from PIL import Image

# Define the paths to the folders
directory = 'output (split_3)/finetune_resnet50_pretraining_swav_batch_32_lr_0.001_epochs_100_seed_0/checkpoint/FA_GradCAM_Match/'
true_positive_folder = directory+'True Positive'
fa_folder = directory+'FA'
combined_folder = directory+'Combined_FA_GradCAM'

# Create the combined folder if it doesn't exist
os.makedirs(combined_folder, exist_ok=True)

# Iterate over the files in the True Positive folder
for true_positive_filename in os.listdir(true_positive_folder):
    if true_positive_filename.endswith('.png'):
        # Extract the blablabla part from the true positive filename
        parts = true_positive_filename.split('_')
        blablabla_part = '_'.join(parts[2:])[:-4]
        # print(blablabla_part)

        # Construct the corresponding FA filename
        fa_filename = f"{blablabla_part}_FA.png"

        # Load the images
        true_positive_image_path = os.path.join(true_positive_folder, true_positive_filename)
        fa_image_path = os.path.join(fa_folder, fa_filename)

        if os.path.exists(fa_image_path):
            true_positive_image = Image.open(true_positive_image_path)
            fa_image = Image.open(fa_image_path)

            # Ensure the FA image is (50, 512)
            fa_image = fa_image.resize((512, 50))

            # Ensure the True Positive image is (496, 1024)
            true_positive_image = true_positive_image.resize((1024, 496))

            # Create a new image with the combined height and the same width
            combined_image = Image.new('RGB', (1024, 546), (0, 0, 0))  # Black background

            # Paste the FA image on the top right corner
            combined_image.paste(fa_image, (512, 0))

            # Paste the True Positive image below the FA image
            combined_image.paste(true_positive_image, (0, 50))

            # Save the combined image in the new folder
            combined_image.save(os.path.join(combined_folder, f"Combined_{blablabla_part}.png"))

            # print(f"Saved: Combined_{blablabla_part}.png")
        else:
            print(f"FA image not found for: {true_positive_filename}")