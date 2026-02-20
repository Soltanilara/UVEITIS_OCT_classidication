import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import random
import os
import copy
import csv
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import f1_score, confusion_matrix, classification_report, roc_curve, auc

# Create the parser and add arguments
parser = argparse.ArgumentParser(description="Train or evaluate a classifier model.")
parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train', help='Mode to run the script in: "train" or "eval"')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pt', help='Path to the model checkpoint for evaluation')
parser.add_argument('--ckpt_interval', type=int, default=10, help='Interval of epochs to save checkpoints')
# Add all other arguments as in your original script here

args = parser.parse_args()

# Seed and environment setup as in your original script
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)

# Directories and file paths setup as in your original script
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

# Data loading, transformation, and model setup code as in your original script
# Ensure all original functionalities are retained

# The modified train_model function with checkpointing
def train_model(model, criterion, optimizer, num_epochs, ckpt_interval):
    for epoch in range(num_epochs):
        # Training loop as in your original script
        # At the end of each epoch:
        if (epoch + 1) % ckpt_interval == 0:
            checkpoint_path = os.path.join(args.output_path, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)

# The modified evaluate_model function with additional metrics
def evaluate_model(model, test_loader):
    model.eval()
    running_corrects = 0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = nn.functional.softmax(outputs, dim=1)
            running_corrects += torch.sum(preds == labels.data)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    acc = running_corrects.double() / len(test_loader.dataset)
    f1 = f1_score(all_labels, preds.cpu().numpy(), average='binary')
    confusion = confusion_matrix(all_labels, preds.cpu().numpy())
    report = classification_report(all_labels, preds.cpu().numpy())
    fpr, tpr, _ = roc_curve(all_labels, [p[1] for p in all_probs])
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    # Save probabilities to a CSV file
    df_probs = pd.DataFrame(all_probs, columns=['Prob_Negative', 'Prob_Positive'])
    df_probs.to_csv(os.path.join(args.output_path, 'test_probabilities.csv'), index=False)

    # Rest of your evaluation output code

# Main execution logic
if args.mode == 'train':
    model, _ = train_model(model, criterion, optimizer, args.num_epochs, args.ckpt_interval)
elif args.mode == 'eval':
    model.load_state_dict(torch.load(args.checkpoint))
    evaluate_model(model, test_loader)
