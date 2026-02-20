import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import random
import os
import copy  # To save the best model
import csv
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import (f1_score, confusion_matrix, classification_report, roc_curve, auc,
                             roc_auc_score,precision_score, recall_score, cohen_kappa_score)
from pytorch_pretrained_vit import ViT

# Create the parser
parser = argparse.ArgumentParser()

# Define arguments
parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train', help='Mode to run the script in: "train" or "eval"')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pt', help='Path to the model checkpoint for evaluation')
parser.add_argument('--pickby', type=str, default='f1', choices=['qwk','f1','auc'], help='Criterion for picking the best model')
parser.add_argument('--mimo', action='store_true', help='Combine mild/moderate')
parser.add_argument('--mose', action='store_true', help='Combine moderate/severe')
parser.add_argument('--ckpt_interval', type=int, default=50, help='Interval of epochs to save checkpoints')
parser.add_argument('--hflip', action='store_true', help='Apply horizontal flip')
parser.add_argument('--elastic', action='store_true', help='Apply elastic transformation')
parser.add_argument('--brightness', action='store_true', help='Apply brightness adjustment')
parser.add_argument('--contrast', action='store_true', help='Apply contrast adjustment')
parser.add_argument('--gnoise', action='store_true', help='Apply Gaussian noise')
parser.add_argument('--batch_size', type=int, default=64, help='Input batch size for training (default: 128)')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate (default: 0.01)')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warm-up epochs (default: 10)')
parser.add_argument('--T_0', type=int, default=10, help='Number of epochs to first warm restart (default: 10)')
parser.add_argument('--T_multi', type=int, default=1, help='Factor for increasing T_i (default: 1)')
parser.add_argument('--eta_min', type=float, default=1e-5, help='Minimum learning rate (default: 1e-6)')
parser.add_argument('--earlystop', action='store_true', help='Enable early stopping')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping (default: 10)')
parser.add_argument('--loss', type=str, default='CE', choices=['CE','focal','wasserstein'], help='Loss function (default: wasserstein)')
parser.add_argument('--unweighted', action='store_true', help='Enable for unweighted loss function')
parser.add_argument('--gamma', type=float, default=2.0, help='Focus parameter for focal loss (default: 2.0)')
parser.add_argument('--p', type=float, default=1.0, help='Wasserstein p parameter(default: 1.0)')
parser.add_argument('--protocol', type=str, default='finetune', choices=['finetune','lin_eval','scratch','complement'], help='Training protocol (default: finetune)')
parser.add_argument('--model', type=str, default='resnet50', help='backbone model')
parser.add_argument('--pretraining', type=str, default='swav', choices=['swav','barlowtwins','supervised','vit','dino'], help='Training protocol (default: swav)')
parser.add_argument('--beta', type=float, default=0.0, help='Orthogonalization intensity for complement tuning (default: 0)')
parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
parser.add_argument('--dataset_path', type=str, default='/home/amin/Datasets/OCT2017', help='Path to images')
parser.add_argument('--output_path', type=str, default='pretraining_OCT2017_output', help='Path to save the output')
parser.add_argument('--gpu', type=int, default=0, help='GPU device index to use (default: 0)')

# args, unknown = parser.parse_known_args()

# Parse arguments
args = parser.parse_args()

# Python's built-in random module
random.seed(args.seed)

# NumPy
np.random.seed(args.seed)

# PyTorch
torch.manual_seed(args.seed)

# If you are using CUDA (PyTorch with GPU)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)  # for multi-GPU setups

# Additional configurations for further ensuring reproducibility:
# This makes sure that the same algorithm is used each time by PyTorch
# Note: This can have a performance impact and may result in different behavior
# when switching between CPU and GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set a fixed value for the hash seed (for hash-based operations in Python)
os.environ['PYTHONHASHSEED'] = str(args.seed)

folder_name = (f"{args.protocol}_{args.model}_{f'pretraining_{args.pretraining}_' if args.protocol!='scratch' else ''}"
               f"{args.loss}_loss_{str(args.gamma)+'_' if args.loss=='focal' else str(args.p)+'_' if args.loss=='wasserstein' else ''}"
               f"{'MiMo_' if args.mimo else 'MoSe_' if args.mose else ''}"
               f"{f'unweighted_' if args.unweighted else ''}pickby_{args.pickby}"
               f"batch_{args.batch_size}_lr_{args.lr}_{args.warmup_epochs}_{args.T_0}_{args.T_multi}_{args.eta_min}"
               f"_epochs_{args.num_epochs}{f'_hflip' if args.hflip else ''}{f'_elastic' if args.elastic else ''}"
               f"{f'_brightness' if args.brightness else ''}{f'_contrast' if args.contrast else ''}"
               f"{f'_gnoise' if args.gnoise else ''}{f'_beta_{args.beta}' if args.protocol=='complement' else ''}"
               f"_seed_{args.seed}")
full_output_path = os.path.join(args.output_path, folder_name)

# Check if the full_output_path exists, if not, create it
if not os.path.exists(full_output_path):
    os.makedirs(full_output_path)

# Class-to-target mapping
label_mapping = {'NORMAL': 0, 'DME': 1, 'CNV': 2, 'DRUSEN': 3}

# Function to load data from folder structure
def load_data_from_folders(root_folder):
    paths = []
    labels = []

    # Traverse the class folders inside the root folder
    for class_name, label in label_mapping.items():
        class_folder = os.path.join(root_folder, class_name)

        # List all image files in the current class folder
        for file_name in os.listdir(class_folder):
            file_path = os.path.join(class_folder, file_name)
            if os.path.isfile(file_path):  # Ensure it's a file
                paths.append(file_path)
                labels.append(label)

    return paths, torch.tensor(labels, dtype=torch.long)

# Custom Dataset class
class CustomImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.paths, self.labels = load_data_from_folders(folder_path)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


if args.model in {'B_16_imagenet1k', 'L_16_imagenet1k'} or args.pretraining == 'vit':
    val_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5]),
    ])
elif args.model in {'vitb14_dino'}:
    val_transform = transforms.Compose([
        transforms.Resize((518,518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5]),
    ])
elif args.protocol=='scratch':
    val_transform = transforms.Compose([
        transforms.Resize((496,512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5]),
        # transforms.Normalize(mean=train_mean, std=train_std),
    ])
else:
    val_transform = transforms.Compose([
        transforms.Resize((496,512)),
        # transforms.Resize((224, 224)),  # ResNet-50 requires 224x224 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Start building the transform pipeline
transform_list = []
if args.model in {'B_16_imagenet1k', 'L_16_imagenet1k'} or args.pretraining == 'vit':
    transform_list.append(transforms.Resize((384, 384)))
elif args.model in {'vitb14_dino'}:
    transform_list.append(transforms.Resize((518,518)))
else:
    transform_list.append(transforms.Resize((496,512)))
transform_list.append(transforms.ToTensor())

# Conditionally apply augmentations based on command-line flags
if args.hflip:
    transform_list.append(transforms.RandomHorizontalFlip())

# if args.rotate > 0.0:
#     transform_list.append(transforms.RandomRotation(degrees=args.rotate))

if args.elastic:
    # Placeholder for elastic transform: Requires a more advanced library like Albumentations or a custom implementation
    transform_list.append(transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.8,1.2), shear=10))  # Simulating elastic-like distortions with affine transformations

# if args.random_crop:
#     transform_list.append(transforms.RandomResizedCrop(size=(496, 512), scale=(0.9, 1.0)))

if args.brightness or args.contrast:
    brightness = 0.2 if args.brightness else 0.0
    contrast = 0.2 if args.contrast else 0.0
    transform_list.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))

if args.gnoise:
    # Define a custom transform to add Gaussian noise
    transform_list.append(transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x[0, :, :].unsqueeze(0))))

if args.model in {'B_16_imagenet1k', 'L_16_imagenet1k','vitb14_dino'} or args.pretraining in {'vit','dino'} or args.protocol=='scratch':
    transform_list.append(transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5]))
else:
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

# Create the final transform pipeline
transform = transforms.Compose(transform_list)

# The transform pipeline will now apply the augmentations based on the flags provided

# Update dataset paths
train_dataset_path = os.path.join(args.dataset_path, 'train')
# val_dataset_path = os.path.join(args.dataset_path, 'val')
test_dataset_path = os.path.join(args.dataset_path, 'test')

# Create datasets and dataloaders
train_dataset = CustomImageDataset(train_dataset_path, transform=transform)
train_loader = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True)

val_dataset = CustomImageDataset(test_dataset_path, transform=val_transform)
val_loader = DataLoader(val_dataset,
                        batch_size=args.batch_size,
                        shuffle=False)

test_dataset = CustomImageDataset(test_dataset_path, transform=val_transform)
test_loader = DataLoader(test_dataset,
                         batch_size=args.batch_size,
                         shuffle=False)


if args.mimo or args.mose:
    out_class = 3
else:
    out_class = 4

if args.protocol == 'complement':
    class R50_Complement(nn.Module):
        def __init__(self):
            super(R50_Complement, self).__init__()

            # Load a pretrained ResNet-50 model for frozen50
            if args.pretraining == 'swav':
                self.frozen50 = torch.hub.load('facebookresearch/swav:main', 'resnet50')
                # saved_weights = torch.nn.Parameter(self.frozen50.conv1.weight[:, :2, :, :])
                # self.frozen50.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                # self.frozen50.conv1.weight = saved_weights
            elif args.pretraining == 'barlowtwins':
                self.frozen50 = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
                # saved_weights = torch.nn.Parameter(self.frozen50.conv1.weight[:, :2, :, :])
                # self.frozen50.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                # self.frozen50.conv1.weight = saved_weights
            self.frozen50.fc = nn.Identity()  # Remove the final layer
            # Freeze all layers in frozen50
            for param in self.frozen50.parameters():
                param.requires_grad = False
            # Initialize a new ResNet-50 model for encoder
            self.encoder = models.resnet50(pretrained=False)
            # self.encoder.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.encoder.fc = nn.Identity()  # Remove the final layer
            # Define the classifier
            self.classifier = nn.Linear(4096, out_class)  # 4096 inputs, 10 outputs
        def forward(self, x):
            # Pass the input through frozen50 and encoder
            out1 = self.frozen50(x)
            out2 = self.encoder(x)
            # Concatenate the outputs from frozen50 and encoder
            combined = torch.cat((out1, out2), dim=1)
            # Pass the concatenated outputs through the classifier
            out = self.classifier(combined)
            return out, out1, out2
    # Create an instance of the custom model
    model = R50_Complement()
elif args.protocol == 'scratch':
    model = models.resnet50(pretrained=False)
    # model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, out_class)  # Adjusting for CIFAR-10 classes
else:
    if args.pretraining == 'swav':
        if args.model=='resnet50':
            model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        elif args.model=='resnet50w5':
            model = nn.Sequential(
                torch.hub.load('facebookresearch/swav:main', 'resnet50w5'),
                nn.Linear(10240,out_class)
            )
    elif args.pretraining == 'barlowtwins':
        model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
        # saved_weights = torch.nn.Parameter(model.conv1.weight[:, :2, :, :])
        # model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # model.conv1.weight = saved_weights
    elif args.pretraining == 'dino':
        if args.model == 'vitb14_dino':
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
    elif args.pretraining == 'supervised':
        if args.model=='resnet18':
            model = models.resnet18(pretrained=True)
            # print(model)
            # saved_weights = torch.nn.Parameter(model.conv1.weight[:, :2, :, :])
            # model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # model.conv1.weight = saved_weights
        elif args.model=='resnet50':
            model = models.resnet50(pretrained=True)
            # print(model)
            # saved_weights = torch.nn.Parameter(model.conv1.weight[:, :2, :, :])
            # model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # model.conv1.weight = saved_weights
    # print(model)
    # print(model(torch.rand((2,3,224,224)).to(f"cuda:0")).size())
    elif args.pretraining == 'vit':
        model = ViT(args.model, pretrained=True)
        # num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs, out_class)
        # print(model)

    if args.protocol == 'lin_eval':
        for param in model.parameters():
            param.requires_grad = False
    if args.model not in {'resnet50w5','vitb14_dino'}:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, out_class)
    elif args.model == 'vitb14_dino':
        num_ftrs = model.linear_head.in_features
        model.linear_head = nn.Linear(num_ftrs, out_class)

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# Step 3: Define Loss Function and Optimizer

class FocalLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2.0, reduction='mean'):
        """
        :param class_weights: Tensor of shape [num_classes] containing the weight for each class.
        :param gamma: Focusing parameter, typically set to 2.0.
        :param reduction: Specifies the reduction to apply to the output. Options are 'none', 'mean', and 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Convert inputs to log probabilities
        log_probs = F.log_softmax(inputs, dim=-1)

        # Gather the log probabilities for the target classes
        targets = targets.view(-1, 1)  # Reshape targets to match log_probs shape
        log_probs = log_probs.gather(1, targets)
        log_probs = log_probs.view(-1)

        # Get the probabilities of the correct class (p_t)
        probs = log_probs.exp()  # Get p_t from log_probs

        # Calculate the focal loss term (1 - p_t)^gamma
        focal_weight = (1 - probs) ** self.gamma

        # Apply the class weights (if provided)
        if self.class_weights is not None:
            # Gather the class weights for the target labels
            class_weights = self.class_weights.gather(0, targets.view(-1))
            focal_weight = focal_weight * class_weights

        # Compute the final loss
        loss = -focal_weight * log_probs

        # Apply reduction (mean or sum)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Define the custom Wasserstein loss that applies class weights after the distance calculation
class WassersteinLoss(nn.Module):
    def __init__(self, class_weights=None,p=1):
        super(WassersteinLoss, self).__init__()
        # Store class weights if provided, otherwise set to None
        self.class_weights = class_weights
        self.p = p

    def forward(self, logits, targets):
        # Apply softmax to the logits to get probabilities
        prob_dist = torch.softmax(logits, dim=1)

        # Convert targets to one-hot encoding for calculating CDF
        num_classes = logits.size(1)
        one_hot_targets = F.one_hot(targets, num_classes=num_classes).float()

        # Compute CDFs for both the predicted and true distributions
        cdf_pred = torch.cumsum(prob_dist, dim=1)
        cdf_true = torch.cumsum(one_hot_targets, dim=1)

        # Calculate the Wasserstein distance (without class weights)
        wasserstein_distance = torch.sum(torch.abs(cdf_pred - cdf_true) ** self.p, dim=1) ** (1 / self.p)

        # Apply class weights based on the true class of each sample
        if self.class_weights is not None:
            sample_weights = self.class_weights[targets]  # Get the weight for each sample's true class
            wasserstein_distance = wasserstein_distance * sample_weights

        # Return the mean weighted Wasserstein distance over the batch
        return torch.mean(wasserstein_distance)

classWeights = [0.14123477, 0.32751083, 0.09989499, 0.43135944]
if args.loss == 'CE':
    if args.unweighted:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(classWeights).to(device))
elif args.loss == 'focal':
    if args.unweighted:
        criterion = FocalLoss(gamma=args.gamma)
    else:
        criterion = FocalLoss(class_weights=torch.tensor(classWeights).to(device), gamma=args.gamma)
elif args.loss == 'wasserstein':
    if args.unweighted:
        criterion = WassersteinLoss()
    else:
        criterion = WassersteinLoss(class_weights=torch.tensor(classWeights).to(device))
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.7225,1.2775]).to(device))       # split_1
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.4821,1.5179]).to(device))       # split_2
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.4948,1.5052]).to(device))       # split_3
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.4936,1.5064]).to(device))       # split_5
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.4602,1.5398]).to(device))       # split_9
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5425,4.4575]).to(device))     # (H6P36,H12P19P43)
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.6683,4.3317]).to(device))     # (H6P36,H12P19P43) randomized trainval
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.55,4.45]).to(device))
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.434,9.567]).to(device))
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)

# Define a function for linear warm-up
def warmup_lr_lambda(current_step, warmup_steps=args.warmup_epochs):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0

# Warm-up scheduler
warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: warmup_lr_lambda(step, warmup_steps=args.warmup_epochs))

# Replace with CosineAnnealingWarmRestarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=args.T_0,    # Initial restart period, same as your warm-up duration if needed
    T_mult=args.T_multi,  # Multiplication factor for increasing the restart period (can be adjusted as needed)
    eta_min=args.eta_min  # Minimum learning rate
)

# Step 4: Training Loop
model = model.to(device)

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=args.patience, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_auc_max = float('-inf')
        self.delta = delta

    def __call__(self, val_auc, model):
        score = val_auc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
            self.counter = 0

    def save_checkpoint(self, val_auc, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation performance increased ({self.val_auc_max:.6f} --> {val_auc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), full_output_path+f'/checkpoint.pt')
        self.val_auc_max = val_auc

# Update the train_model function to include validation loss computation
def train_model(model, criterion, optimizer, num_epochs=args.num_epochs, ckpt_interval = args.ckpt_interval):
    early_stopping = EarlyStopping(verbose=True)
    stats = {'train_loss':[], 'train_acc':[], 'train_f1':[], 'val_loss':[], 'val_acc':[], 'val_f1':[], 'val_auc':[]}
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        train_preds, train_labels = [], []

        train_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs - 1}', leave=False)
        for inputs, labels in train_progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            if args.protocol == 'complement':
                outputs, out1, out2 = model(inputs)
                out1 = out1 - out1.mean(dim=0)
                out1 = out1/torch.norm(out1,dim=0)
                out2 = out2 - out2.mean(dim=0)
                out2 = out2/torch.norm(out2,dim=0)
                # print(criterion(outputs, labels), torch.sum(torch.pow(torch.matmul(out1.T, out2),2)))
                loss = criterion(outputs, labels) + args.beta * torch.sum(torch.pow(torch.matmul(out1.T, out2),2))
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            train_preds.extend(preds.view(-1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            # Update progress bar
            train_progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        stats['train_loss'].append(epoch_loss)
        stats['train_acc'].append(epoch_acc.item())
        stats['train_f1'].append(train_f1)

        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, '
              f'F1: {train_f1:.4f}')

        # Update the learning rate: Warm-up or Cosine Annealing based on the current epoch
        if epoch < args.warmup_epochs:  # First 10 epochs for warm-up
            warmup_scheduler.step()
        else:  # Cosine annealing for the remaining epochs
            scheduler.step()

        # Calculate the validation loss after each epoch
        val_loss = 0.0
        val_corrects = 0
        val_preds, val_labels = [], []
        val_probs = []
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                if args.protocol == 'complement':
                    outputs,_,_ = model(inputs)
                else:
                    outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                probs = 1-nn.functional.softmax(outputs, dim=1)[:,0]
                val_corrects += torch.sum(preds == labels.data)
                val_preds.extend(preds.view(-1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_auc = roc_auc_score([int(lab>0.1) for lab in val_labels], val_probs)  # Calculate AUC based on labels and probabilities

        # Store the results
        stats['val_loss'].append(val_loss)
        stats['val_acc'].append(val_acc.item())
        stats['val_f1'].append(val_f1)
        stats['val_auc'].append(val_auc)

        # Print the results
        print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc: .4f}, '
              f'Validation F1: {val_f1:.4f}, Validation AUC: {val_auc:.4f}')

        # Call early stopping
        if args.pickby=='auc':
            early_stopping(val_auc, model)
        if args.pickby=='f1':
            early_stopping(val_f1, model)

        if args.earlystop and early_stopping.early_stop:
            print("Early stopping")
            break

        if (epoch + 1) % ckpt_interval == 0:
            checkpoint_path = os.path.join(full_output_path, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
    # Load the last checkpoint with the best model
    model.load_state_dict(torch.load(full_output_path+f'/checkpoint.pt'))

    return model,stats

def evaluate_model(model, test_loader):
    model.eval()
    running_corrects = 0
    all_preds = []
    all_labels = []
    all_probs = []

    eval_progress_bar = tqdm(test_loader, desc='Evaluating', leave=False)
    with torch.no_grad():
        for inputs, labels in eval_progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            if args.protocol == 'complement':
                outputs, _, _ = model(inputs)
            else:
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = nn.functional.softmax(outputs, dim=1)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = running_corrects.double() / len(test_loader.dataset)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    confusion = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    print(f'Test Accuracy: {acc:.4f}')
    print(f'Test Precision: {f1:.4f}')
    print(f'Test Recall: {f1:.4f}')
    print(f'Test F1 Score: {f1:.4f}')
    print('Confusion Matrix:\n', confusion)
    print('Classification Report:\n', report)

    if not os.path.exists(os.path.join(full_output_path,args.checkpoint[:-3])):
        os.makedirs(os.path.join(full_output_path,args.checkpoint[:-3]))
    eval_output_path = os.path.join(full_output_path,args.checkpoint[:-3])
    filename = eval_output_path + "/test_metrics.csv"

    # Write metrics to CSV
    with open(filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([acc, precision, recall, f1])
    # Optionally, save the classification report to a file
    with open(eval_output_path + "/confusion_matrix.txt", 'w') as confusion_file:
        confusion_file.write(np.array2string(confusion))
    with open(eval_output_path + "/classification_report.txt", 'w') as report_file:
        report_file.write(report)
    fpr, tpr, _ = roc_curve([int(lab>0.1) for lab in all_labels], [1-p[0] for p in all_probs])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(eval_output_path, 'ROC.png'))
    # plt.show()


if args.mode == 'train':
    # Train and Evaluate
    model,history = train_model(model, criterion, optimizer)

    # Number of epochs
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot for accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_f1'], label='Training f1 Score')
    plt.plot(epochs, history[f'val_{args.pickby}'], label=f'Validation {args.pickby} Score')
    plt.title('Training and Validation Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Performance')
    plt.legend()

    # Plot for loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(full_output_path+'/history.png')

    evaluate_model(model, test_loader)
elif args.mode == 'eval':
    model.load_state_dict(torch.load(full_output_path+'/'+args.checkpoint))
    evaluate_model(model, test_loader)