import pandas as pd
from PIL import Image

def load_images_and_zero_rectangle(csv_file, prefix):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Track unique resolutions if needed
    unique_resolutions = set()

    for index, row in df.iterrows():
        # Form the full path to the image
        image_path = prefix + row.iloc[0]

        try:
            with Image.open(image_path) as img:
                width, height = img.size

                # Get pixel access
                pixels = img.load()

                # Define the 60x50 rectangle at the bottom-left:
                # Left: 0, Right: 60, Top: height - 50, Bottom: height
                rect_left = 0
                rect_right = min(50, width)       # In case width < 60
                rect_top = max(height - 70, 0)    # In case height < 50
                rect_bottom = height

                # Zero out pixels within this rectangle
                # (Assuming RGB; if grayscale, use (0,) instead of (0,0,0))
                for x in range(rect_left, rect_right):
                    for y in range(rect_top, rect_bottom):
                        pixels[x, y] = (0, 0, 0)

                # Overwrite the original image
                img.save(image_path)

                # Record the resolution
                unique_resolutions.add((width, height))

        except IOError:
            print(f"Error opening image at {image_path}")

    return unique_resolutions


# Usage
csv_file = 'Annotation 01032025.csv'
prefix = 'DatasetNoScale03052025/'
unique_resolutions = load_images_and_zero_rectangle(csv_file, prefix)
