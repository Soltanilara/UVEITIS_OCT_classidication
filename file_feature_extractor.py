import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Paths to your CSV files
# csv_files = ["split_3/test.csv", "split_10/test.csv", "split_2/test.csv", "split_9/test.csv"]
csv_files = ["split_3/test.csv", "split_10/test.csv", "split_3/val.csv", "split_10/val.csv",]
output_path = 'Pretrained Features'
# Preprocess images for ResNet-50
transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom dataset to load images and labels
class ImageDataset(Dataset):
    def __init__(self, csv_files, transform):
        self.data = []
        for idx, file_path in enumerate(csv_files):
            df = pd.read_csv(file_path)
            for img_path in df['Image File']:
                self.data.append(('Dataset 09192024/'+img_path, idx))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = torch.zeros(3, 496, 512)  # Use a blank image if loading fails
        return img, label

# Load the dataset and create a DataLoader for batching
batch_size = 32  # Adjust the batch size based on your system's memory capacity
dataset = ImageDataset(csv_files, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load ResNet-50 and modify it to return features
model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
model = nn.Sequential(*list(model.children())[:-1])  # Remove the final classification layer
model.eval()
model = model.cuda() if torch.cuda.is_available() else model

# Initialize lists to store features and labels
all_features = []
all_labels = []

# Process images in batches
for images, labels in tqdm(dataloader, desc="Extracting features in batches"):
    images = images.cuda() if torch.cuda.is_available() else images
    with torch.no_grad():
        batch_features = model(images).squeeze(-1).squeeze(-1)  # Shape (batch_size, 2048)
    all_features.append(batch_features.cpu().numpy())
    all_labels.extend(labels.numpy())

# Convert lists to numpy arrays and save
features = np.concatenate(all_features, axis=0)
labels = np.array(all_labels)

if not os.path.exists(output_path):
    os.makedirs(output_path)

# np.save(f"{output_path}/test_features_3_10_2_9.npy", features)
# np.save(f"{output_path}/labels_3_10_2_9.npy", labels)
np.save(f"{output_path}/test_val_features_3_10.npy", features)
np.save(f"{output_path}/test_val_labels_3_10.npy", labels)

print("Feature extraction and saving completed.")
