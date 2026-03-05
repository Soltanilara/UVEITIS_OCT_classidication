import pandas as pd
import cv2
import os
import numpy as np
from skimage.filters import sobel
from skimage.measure import shannon_entropy
from imquality import brisque
from brisque import BRISQUE
from pypiqe import piqe
import torch
# from piq import niqe
from tqdm import tqdm

csv_list = [
'split_2/test.csv',
'split_3/test.csv',
'split_9/test.csv',
'split_10/test.csv'
]
for csv_file_path in csv_list:
    # Load the CSV file
    file_dir, file_name = os.path.split(csv_file_path)
    file_base, file_ext = os.path.splitext(file_name)
    output_csv_path = os.path.join(file_dir, f"{file_base}_w_ImageQuality{file_ext}")

    data = pd.read_csv(csv_file_path)

    # Constants for theoretical maximums
    IMAGE_HEIGHT, IMAGE_WIDTH = 496, 512
    MAX_ENTROPY = 8  # Theoretical max for 8-bit images
    MAX_EDGE_SHARPNESS = (np.sqrt(2) * 4 * 255) / (IMAGE_HEIGHT * IMAGE_WIDTH)

    # Normalization helper function
    def normalize(value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)

    # Flip metric: for metrics where higher is better (e.g., entropy, sharpness)
    def flip_metric(value):
        return 1 - value

    # # Function to compute NIQE using piq
    # def calculate_niqe(img):
    #     img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    #     return niqe(img_tensor).item()

    # Function to compute edge sharpness
    def calculate_edge_sharpness(img):
        edges = sobel(img)
        return edges.mean()

    # Function to compute image quality scores
    def calculate_quality_scores(image_path):
        img = cv2.imread('Dataset 09192024/'+image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None, None, None

        # Compute BRISQUE, NIQE, and PIQE
        if len(img.shape) == 2:  # Grayscale image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img
        brisque_model = BRISQUE()
        brisque_score = brisque_model.score(img_rgb)
        # niqe_score = niqe.score(img)
        piqe_score,_,_,_ = piqe(img)
        #
        # Compute entropy
        entropy_score = shannon_entropy(img)

        # Compute edge sharpness
        edge_sharpness_score = calculate_edge_sharpness(img)

        return brisque_score, piqe_score, entropy_score, edge_sharpness_score

    # Add new columns to the DataFrame
    brisque_scores, piqe_scores = [], []
    entropy_scores, edge_sharpness_scores = [], []
    combined_scores = []

    print("Calculating quality scores...")
    for image_file in tqdm(data['Image File']):
        scores = calculate_quality_scores(image_file)
        if scores[0] is None:
            # Default values for missing images
            brisque_scores.append(None)
            # niqe_scores.append(None)
            piqe_scores.append(None)
            entropy_scores.append(None)
            edge_sharpness_scores.append(None)
            combined_scores.append(None)
            continue

        brisque_score, piqe_score, entropy_score, edge_sharpness_score = scores

        # Append raw scores
        brisque_scores.append(brisque_score)
        # niqe_scores.append(niqe_score)
        piqe_scores.append(piqe_score)
        entropy_scores.append(entropy_score)
        edge_sharpness_scores.append(edge_sharpness_score)

        # Normalize and flip scores where needed
        brisque_norm = normalize(brisque_score, 0, 100)  # Lower is better
        # niqe_norm = normalize(niqe_score, 0, 10)        # Lower is better
        piqe_norm = normalize(piqe_score, 0, 100)        # Lower is better

        # # Entropy normalization (flip for lower = better)
        # entropy_norm = flip_metric(normalize(entropy_score, 0, MAX_ENTROPY))

        # Edge sharpness normalization (flip for lower = better)
        edge_sharpness_norm = flip_metric(normalize(edge_sharpness_score, 0, MAX_EDGE_SHARPNESS))

        # Combined score using normalized metrics
        combined_score = 1-((
            brisque_norm +
            piqe_norm
        ) / 2)  # Average of all normalized scores

        combined_scores.append(combined_score)

    # Append scores to the DataFrame
    data['BRISQUE'] = brisque_scores
    # data['NIQE'] = niqe_scores
    data['PIQE'] = piqe_scores
    data['Entropy'] = entropy_scores
    data['Edge Sharpness'] = edge_sharpness_scores
    data['Combined Quality Score'] = combined_scores

    # Save the updated DataFrame to a new CSV
    data.to_csv(output_csv_path, index=False)
    print(f"Updated CSV with quality scores saved to {output_csv_path}")
