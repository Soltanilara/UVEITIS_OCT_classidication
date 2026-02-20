from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw
import pandas as pd
import os
import numpy as np
import pickle

def add_black_pixels_bottom(image_path, new_height, new_width):
    # Load the original image
    original_image = Image.open(image_path)

    # Get the original image size
    original_width, original_height = original_image.size
    if original_width != int(new_width):
        ratio = new_width/original_width
        original_image = original_image.resize((int(new_width), int(ratio*original_height)))
    new_image = Image.new('RGB', (new_width, new_height), "black")

    # Paste the original image onto the new image (at the top)
    new_image.paste(original_image, (0, 0))
    return new_image

def blacken_top_bottom_left(image, h=40, w=60):
    image_np = np.array(image)
    image_np[:h,:w]=(0,0,0)
    image_np[496-h:496,:w]=(0,0,0)
    return Image.fromarray(image_np)
def blacken_bottom(image):
    image_np = np.array(image)
    image_np[496:,:]=(0,0,0)
    return Image.fromarray(image_np)
def process_row(row, save=False):

    foreground_path = 'From New Uveitis Code/Test02082025 (Before Crop and Resize)/'+row['Image File']
    background_path = 'From New Uveitis Code/Test02082025 (Before Crop and Resize)/'+row['Background File']
    # foreground = add_black_pixels_bottom(foreground_path, row['Current Height'], row['Current Width'])
    foreground = Image.open(foreground_path)
    foreground = foreground.crop((0,0,496,foreground.height))
    foreground = blacken_top_bottom_left(foreground)
    if foreground.height>496:
        foreground = blacken_bottom(foreground)
    background = Image.open(background_path)

    x_pos = int((row["Current X"] + row["Background Width"] / 2 - row["Background X"] - row["Current Width"] / 2))
    y_pos = int((row["Current Y"] + row["Background Height"] / 2 - row["Background Y"] - row["Current Height"] / 2))

    background = background.resize((int(row["Background Width"]), int(row["Background Height"])))
    background = background.rotate(-row["Current Rotation"], center=(x_pos + int(row["Current Width"] / 2), y_pos + int(row["Current Height"] / 2)), expand=False)
    foreground = foreground.resize((int(row["Current Width"]), int(row["Current Height"])))

    composite = Image.new('RGBA', background.size)
    composite_bg = Image.new('RGBA', background.size)
    composite_bg.paste(background, (0, 0))
    composite_bg = composite_bg.convert('RGB')

    composite.paste(foreground, (x_pos, y_pos))
    composite = composite.convert("RGB")

    enhancer = ImageEnhance.Color(composite)
    foreground_enhanced = enhancer.enhance(2.0)
    enhancer = ImageEnhance.Contrast(foreground_enhanced)
    foreground_enhanced = enhancer.enhance(2.0)
    # enhancer = ImageEnhance.Sharpness(foreground_enhanced)
    # foreground_enhanced = enhancer.enhance(2.0)
    # foreground_enhanced = composite
    foreground_np = np.array(foreground_enhanced)
    # foreground_enhanced.show()

    green_mask = (foreground_np[:, :, 1] > 250) & (foreground_np[:, :, 0] < 150) & (foreground_np[:, :, 2] < 150)
    # green_mask = (abs(foreground_np[:, :, 1] - foreground_np[:, :, 0])>240) & (abs(foreground_np[:, :, 1] - foreground_np[:, :, 2])>240) & (foreground_np[:, :, 1] > 254)
    green_mask_image = Image.fromarray(np.uint8(green_mask * 255), 'L')
    green_mask_thin = green_mask_image
    green_mask_thin_np = np.array(green_mask_thin)
    # green_mask_thin.save('GREEN_MASK_THIN.png')

    # enhancer = ImageEnhance.Color(composite)
    # foreground_enhanced_ar = enhancer.enhance(2.0)
    # foreground_np_ar = np.array(foreground_enhanced_ar)
    #
    # lower_green = np.array([0, 100, 0])
    # upper_green = np.array([10, 255, 10])
    #
    # green_mask_ar = ((foreground_np_ar >= lower_green) & (foreground_np_ar <= upper_green)).all(axis=2)
    # green_mask_image_ar = Image.fromarray(np.uint8(green_mask_ar * 255), 'L')
    # green_mask_thin_ar = green_mask_image_ar.filter(ImageFilter.MinFilter(5))  # This is a simple way to perform thinning
    # # green_mask_thin_ar.show()
    # green_mask_thin_ar_np = np.array(green_mask_thin_ar)

    vertical_lines_indices = np.where(np.any(green_mask_thin_np, axis=0))[0]
    horizontal_lines_indices = np.where(np.any(green_mask_thin_np, axis=1))[0]
    # print(vertical_lines_indices)
    top_line = horizontal_lines_indices[0]
    bottom_line = horizontal_lines_indices[-1]
    left_line = vertical_lines_indices[0]
    right_line = vertical_lines_indices[-1]

    shift_20 = np.roll(green_mask_thin_np, 20, axis=1)
    # shift_200 = np.roll(green_mask_thin_np, 200, axis=1)
    shift_neg_1 = np.roll(green_mask_thin_np, -1, axis=1)
    shift_3_7_b = np.roll(green_mask_thin_np, (3, 7), axis=(0, 1))
    shift_3_7_t = np.roll(green_mask_thin_np, (-3, 7), axis=(0, 1))

    # Combine the conditions to identify the arrowhead
    arrowhead_mask = green_mask & shift_20 & shift_3_7_b & shift_3_7_t & ~shift_neg_1

    # Find the y-coordinates of the arrowhead
    arrowhead_indices = np.where(np.any(arrowhead_mask, axis=1))[0]
    arrowhead_height = int(arrowhead_indices.mean())

    # arrowhead_indices = np.where(np.any(green_mask_thin_ar_np, axis=1))[0]
    # arrowhead_height = int((arrowhead_indices[0] + arrowhead_indices[-1]) / 2)

    left = left_line
    upper = arrowhead_height
    right = right_line + 1
    lower = arrowhead_height + 1

    if save:
        # Calculate crop box based on green rectangle bounding box
        rect_left = left_line
        rect_top = top_line
        rect_right = right_line + 1  # +1 because slicing is exclusive
        rect_bottom = bottom_line + 1

        # Crop the corresponding region from the background
        cropped_background = composite_bg.crop((rect_left, rect_top, rect_right, rect_bottom))

        return cropped_background
    else:
        # gray_values = np.array(composite_bg.convert('L'))[upper, left:right]
        return None#, None

def main():
    csv_path = 'Exp_90_92/test_90_92.csv'
    df = pd.read_csv(csv_path)

    # save_interval = 45
    output_dir = 'Exp_90_92/Region Crops'
    os.makedirs(output_dir, exist_ok=True)

    # gray_values_list = []
    visit = set()
    for index, row in df.iterrows():
        if '/'.join(row['Image File'].split('/')[:-1]) not in visit and row['Image File'][-8:-4] not in {'0000','0048', '0096'}:
            # if index<18281:
            # if row['Image File'].split('/')[-1][:11] not in {'ZahriyaA003', 'ZahriyaA032', 'ZahriyaA040', 'ZahriyaA048'}:
            # # # if index!=613:
            #     continue
            # print(index,row['Image File'])
            save = True#(index + 1) % save_interval == 0
            # gray_values, crop_composite, green_mask, ar_mask = process_row(row, save=save)
            crop_composite = process_row(row, save=save)
            # gray_values_list.append(gray_values)
            # with open('gray_values.pkl','ab') as f:
            #     pickle.dump(gray_values,f)
            # print(visit)
            if save:# and crop_composite is not None:
                # print(index,row['Image File'])
                foreground_path = row['Image File']
                background_path = row['Background File']
                output_path = os.path.join(output_dir, f"{index}_{os.path.splitext(os.path.basename(foreground_path))[0]}___{os.path.splitext(os.path.basename(background_path))[0]}.png")
                crop_composite.save(output_path)
                # output_path_green_mask = os.path.join(output_dir, f"green_mask_{index}_{os.path.splitext(os.path.basename(foreground_path))[0]}___{os.path.splitext(os.path.basename(background_path))[0]}.png")
                # green_mask.save(output_path_green_mask)
                # output_path_ar_mask = os.path.join(output_dir, f"arrow_mask_{os.path.splitext(os.path.basename(foreground_path))[0]}___{os.path.splitext(os.path.basename(background_path))[0]}.png")
                # ar_mask.save(output_path_ar_mask)
            visit.add('/'.join(row['Image File'].split('/')[:-1]))

if __name__ == "__main__":
    main()
