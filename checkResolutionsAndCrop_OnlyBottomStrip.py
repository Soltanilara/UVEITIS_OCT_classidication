import pandas as pd
from PIL import Image


def load_images_and_process_resolutions(csv_file, prefix):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Initialize the list for storing image resolutions
    # resolutions = []
    unique_resolutions = set()
    # patients_w_hiRes = set()
    # iloc1520=set()

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        # Form the full path to the image
        image_path = prefix + row.iloc[0]

        # Load the image
        try:
            with Image.open(image_path) as img:
                original_size = img.size

                img = img.crop((0, 0, original_size[0], 496))

                # Overwrite the original image
                img.save(image_path)

                # Record the new resolution
                unique_resolutions.add(original_size)
                # resolutions.append(img.size)
        except IOError:
            print(f"Error opening image at {image_path}")

    # # Get unique resolutions
    # unique_resolutions = set(resolutions)

    # Print unique resolutions
    # print("Unique Resolutions:", unique_resolutions)
    #
    # # Optionally check the length of resolutions list matches the number of rows
    # assert len(resolutions) == len(df), "Number of image resolutions does not match number of rows."
    # print(patients_w_hiRes)
    return unique_resolutions


# Usage
csv_file = 'Test02082025_annotations.csv'
prefix = 'Test02082025 (anonymized with en face)/'
# prefix = 'Dataset 06042024/'
unique_resolutions = load_images_and_process_resolutions(csv_file, prefix)
