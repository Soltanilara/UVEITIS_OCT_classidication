import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from timm.models.vgg import vgg19_bn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import random
import os
import copy  # To save the best model
import csv
import cv2
import json
from PIL import Image
import pandas as pd
import matplotlib
matplotlib.rc('font', family='serif', size=14)   # or 12-14 range
matplotlib.rc('axes', titlesize=14)
matplotlib.rc('axes', labelsize=14)
matplotlib.rc('legend', fontsize=12)
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import (f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score, precision_recall_curve)
from pytorch_pretrained_vit import ViT
from captum.attr import IntegratedGradients
import shap


# Create the parser
parser = argparse.ArgumentParser()

# Define arguments
parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train', help='Mode to run the script in: "train" or "eval"')
parser.add_argument('--final', action='store_true', help='Final training on train+val')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pt', help='Path to the model checkpoint for evaluation')
parser.add_argument('--ckpt_interval', type=int, default=100, help='Interval of epochs to save checkpoints')
parser.add_argument('--mixup', action='store_true', help='Enable input mixup')
parser.add_argument('--mixup_alpha', type=float, default=1.0, help='Mixup alpha')
parser.add_argument('--mixup_beta', type=float, default=1.0, help='Mixup beta')
parser.add_argument('--weightedSampling', action='store_true', help='Enable Weighted Random Sampling')
parser.add_argument('--hflip', action='store_true', help='Apply horizontal flip')
parser.add_argument('--elastic', action='store_true', help='Apply elastic transformation')
parser.add_argument('--brightness', action='store_true', help='Apply brightness adjustment')
parser.add_argument('--contrast', action='store_true', help='Apply contrast adjustment')
parser.add_argument('--gnoise', action='store_true', help='Apply Gaussian noise')
parser.add_argument('--batch_size', type=int, default=64, help='Input batch size for training (default: 128)')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate (default: 0.001)')
parser.add_argument('--decay', type=float, default=0.01, help='Weight decay (default: 0.01)')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warm-up epochs (default: 10)')
parser.add_argument('--T_0', type=int, default=10, help='Number of epochs to first warm restart (default: 10)')
parser.add_argument('--T_multi', type=int, default=1, help='Factor for increasing T_i (default: 1)')
parser.add_argument('--eta_min', type=float, default=1e-5, help='Minimum learning rate (default: 1e-6)')
parser.add_argument('--earlystop', action='store_true', help='Enable early stopping')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping (default: 10)')
parser.add_argument('--loss', type=str, default='CE', choices=['CE','focal'], help='Loss function (default: CE)')
parser.add_argument('--unweighted', action='store_true', help='Enable for unweighted loss function')
parser.add_argument('--gamma', type=float, default=2.0, help='Focus parameter for focal loss (default: 2.0)')
parser.add_argument('--protocol', type=str, default='finetune', choices=['finetune','lin_eval','scratch','complement'], help='Training protocol (default: finetune)')
parser.add_argument('--model', type=str, default='resnet50', help='backbone model')
parser.add_argument('--pretraining', type=str, default='swav', choices=['swav','barlowtwins','supervised','vit','dino','oct'], help='Training protocol (default: swav)')
parser.add_argument('--beta', type=float, default=0.0, help='Orthogonalization intensity for complement tuning (default: 0)')
parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
parser.add_argument('--csvpath', type=str, default='fold_0', help='Path to input csv files')
parser.add_argument('--metadata_path', type=str, default='metadata_0', help='Path to final training metadata')
parser.add_argument('--dataset_path', type=str, default='Dataset 01032025', help='Path to images')
parser.add_argument('--output_path', type=str, default='output_fold_0', help='Path to save the output')
parser.add_argument('--gradcam', action='store_true', help='Apply GradCAM')
parser.add_argument('--IG', action='store_true', help='Apply Integrated Gradients')
parser.add_argument('--shap', action='store_true', help='Apply SHAP (GradientExplainer)')
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

folder_name = (f"{f'final_' if args.final else ''}"
               f"{args.protocol}_{args.model}_{f'pretraining_{args.pretraining}_' if args.protocol!='scratch' else ''}"
               f"{args.loss}_loss_{str(args.gamma)+'_' if args.loss=='focal' else ''}"
               f"{f'unweighted_' if args.unweighted else ''}"
               f"{f'weightedSampling_' if args.weightedSampling else ''}"
               f"{f'mixup_{args.mixup_alpha}_{args.mixup_beta}_' if args.mixup else ''}"
               f"batch_{args.batch_size}_lr_{args.lr}_decay_{args.decay}_{args.warmup_epochs}_{args.T_0}_{args.T_multi}_{args.eta_min}"
               f"_epochs_{args.num_epochs}{f'_hflip' if args.hflip else ''}{f'_elastic' if args.elastic else ''}"
               f"{f'_brightness' if args.brightness else ''}{f'_contrast' if args.contrast else ''}"
               f"{f'_gnoise' if args.gnoise else ''}{f'_beta_{args.beta}' if args.protocol=='complement' else ''}"
               f"_seed_{args.seed}")
full_output_path = os.path.join(args.output_path, folder_name)
if args.final:
    full_metadata_path = os.path.join(args.output_path, args.metadata_path)
    print(f"Metadata loaded from {full_metadata_path}.")


# Check if the full_output_path exists, if not, create it
if not os.path.exists(full_output_path):
    os.makedirs(full_output_path)


def load_data(csv_file, csvpath, folder):
    df = pd.read_csv(os.path.join(csvpath,csv_file))
    df['Label'] = df['Label'].map(lambda x: 0 if x == 'negative' else 1)
    paths = df['Image File'].apply(lambda x: os.path.join(folder, x))
    labels = df['Label'].values
    return paths, torch.tensor(labels, dtype=torch.long)

# train_mean = [0.5,0.5,0.5]
# train_std = [0.5,0.5,0.5]

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, csvpath, folder, transform=None):
        self.paths, self.labels = load_data(csv_file, csvpath, folder)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

if args.model in {'B_16_imagenet1k', 'L_16_imagenet1k'}:# or args.pretraining == 'vit':
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
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5]),
        # transforms.Normalize(mean=train_mean, std=train_std),
    ])
else:
    val_transform = transforms.Compose([
        # transforms.Resize((224, 224)),  # ResNet-50 requires 224x224 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Start building the transform pipeline
transform_list = []
if args.model in {'B_16_imagenet1k', 'L_16_imagenet1k'}:# or args.pretraining == 'vit':
    transform_list.append(transforms.Resize((384, 384)))
elif args.model in {'vitb14_dino'}:
    transform_list.append(transforms.Resize((518,518)))
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

if args.model in {'B_16_imagenet1k', 'L_16_imagenet1k','vitb14_dino'} or args.protocol=='scratch':# or args.pretraining in {'vit','dino'}:
    transform_list.append(transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5]))
else:
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

# Create the final transform pipeline
transform = transforms.Compose(transform_list)

if args.final:
    classWeights = list(np.load(args.csvpath + '/classWeights_final.npy').astype(np.float32))
else:
    classWeights = list(np.load(args.csvpath +'/classWeights_train.npy').astype(np.float32))

if args.final:
    train_dataset = CustomImageDataset('train_final.csv', args.csvpath, args.dataset_path, transform=transform)
else:
    train_dataset = CustomImageDataset('train.csv', args.csvpath, args.dataset_path, transform=transform)
# Use WeightedRandomSampler if --weightedSampling is True:
if args.weightedSampling:
    sample_weights = [classWeights[label.item()] for label in train_dataset.labels]

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler
    )
else:
    # Default: normal DataLoader with shuffling
    train_loader = DataLoader(train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

# Test data
test_dataset = CustomImageDataset('test.csv', args.csvpath, args.dataset_path, transform=val_transform)
test_loader = DataLoader(test_dataset,
    batch_size=args.batch_size,
    shuffle=False # don't necessarily have to shuffle the testing data
)

if not args.final:
    # Validation data
    val_dataset = CustomImageDataset('val.csv', args.csvpath, args.dataset_path, transform=val_transform)
    val_loader = DataLoader(val_dataset,
        batch_size=args.batch_size,
        shuffle=False # don't necessarily have to shuffle the testing data
    )

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
            self.classifier = nn.Linear(4096, 2)  # 4096 inputs, 10 outputs
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
    if args.model == 'resnet50':
        model = models.resnet50(pretrained=False)
    elif args.model == 'resnet18':
        model = models.resnet18(pretrained=False)
    elif args.model == 'densenet121':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False)
    elif args.model == 'densenet201':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=False)
    if args.model in {'densenet121','densenet201'}:
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)  # Adjusting for CIFAR-10 classes
    else:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # Adjusting for CIFAR-10 classes
else:
    if args.pretraining == 'swav':
        if args.model=='resnet50':
            model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        elif args.model=='resnet50w5':
            model = nn.Sequential(
                torch.hub.load('facebookresearch/swav:main', 'resnet50w5'),
                nn.Linear(10240,2)
            )
    elif args.pretraining == 'oct':
        if args.model=='resnet50':
            model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 4)
            oct_model_path = ('pretraining_OCT2017_output/finetune_resnet50_pretraining_swav_CE_loss_unweighted_pickby'
                              '_f1batch_64_lr_0.001_10_10_1_1e-05_epochs_100_hflip_seed_0/checkpoint.pt')
            model.load_state_dict(torch.load(oct_model_path))
        elif args.model=='L_16_imagenet1k':
            model = ViT(args.model, pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 4)
            oct_model_path = ('pretraining_OCT2017_output/finetune_L_16_imagenet1k_pretraining_vit_CE_loss_unweighted'
                              '_pickby_f1batch_24_lr_1e-05_10_10_1_1e-07_epochs_100_hflip_seed_0/checkpoint.pt')
            model.load_state_dict(torch.load(oct_model_path))
        elif args.model=='vitb14_dino':
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
            num_ftrs = model.linear_head.in_features
            model.linear_head = nn.Linear(num_ftrs, 4)
            oct_model_path = ('pretraining_OCT2017_output/finetune_vitb14_dino_pretraining_dino_CE_loss_unweighted'
                              '_pickby_f1batch_16_lr_1e-05_10_10_1_1e-07_epochs_100_hflip_seed_0/checkpoint.pt')
            model.load_state_dict(torch.load(oct_model_path))
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
        elif args.model=='vgg19_bn':
            model = models.vgg19_bn(pretrained=True)
            model.classifier[6]=nn.Linear(4096,2)
        elif args.model == 'densenet121':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        elif args.model == 'densenet201':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
    # print(model)
    # print(model(torch.rand((2,3,224,224)).to(f"cuda:0")).size())
    elif args.pretraining == 'vit':
        model = ViT(args.model, pretrained=True)
        # num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs, 2)
        # print(model)

    if args.protocol == 'lin_eval':
        for param in model.parameters():
            param.requires_grad = False
    if args.model not in {'resnet50w5','vitb14_dino','vgg19_bn','densenet121','densenet201'}:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif args.model == 'vitb14_dino':
        num_ftrs = model.linear_head.in_features
        model.linear_head = nn.Linear(num_ftrs, 2)
    elif args.model in {'densenet121', 'densenet201'}:
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)  # Adjusting for CIFAR-10 classes

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
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)

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
    def __init__(self, patience=args.patience, verbose=False, delta=0.0005):
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
        self.val_auc_max = 0
        self.delta = delta
        self.best_epoch = 0

    def __call__(self, val_auc, model, epoch):
        score = val_auc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_auc, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_auc, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_auc, model, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation AUC increased ({self.val_auc_max:.6f} --> {val_auc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), full_output_path+f'/checkpoint.pt')
        self.val_auc_max = val_auc
        self.best_epoch = epoch



# 3. Define Mixup utilities
def mixup_data(x, y, mixup_alpha=1.0, mixup_beta=1.0):
    lam = np.random.beta(mixup_alpha, mixup_beta)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Update the train_model function to include validation loss computation
def train_model(model, criterion, optimizer, num_epochs=args.num_epochs, ckpt_interval = args.ckpt_interval):
    if not args.final:
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
                if args.mixup:
                    mixed_inputs, targets_a, targets_b, lam = mixup_data(inputs, labels,
                                                                         mixup_alpha=args.mixup_alpha,
                                                                         mixup_beta=args.mixup_beta)
                    outputs = model(mixed_inputs)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
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
        train_f1 = f1_score(train_labels, train_preds, average='binary', pos_label=1)
        stats['train_loss'].append(epoch_loss)
        stats['train_acc'].append(epoch_acc.item())
        stats['train_f1'].append(train_f1)

        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, F1: {train_f1:.4f}')

        # Update the learning rate: Warm-up or Cosine Annealing based on the current epoch
        if epoch < args.warmup_epochs:  # First 10 epochs for warm-up
            warmup_scheduler.step()
        else:  # Cosine annealing for the remaining epochs
            scheduler.step()

        if not args.final:
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
                    probs = nn.functional.softmax(outputs, dim=1)[:,1]
                    val_corrects += torch.sum(preds == labels.data)
                    val_preds.extend(preds.view(-1).cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
                    val_probs.extend(probs.cpu().numpy())
            val_loss /= len(val_loader.dataset)
            val_acc = val_corrects.double() / len(val_loader.dataset)
            val_f1 = f1_score(val_labels, val_preds, average='binary', pos_label=1)
            val_auc = roc_auc_score(val_labels, val_probs)  # Calculate AUC based on labels and probabilities

            # Store the results
            stats['val_loss'].append(val_loss)
            stats['val_acc'].append(val_acc.item())
            stats['val_f1'].append(val_f1)
            stats['val_auc'].append(val_auc)

            # Print the results
            print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc: .4f}, '
                  f'Validation F1: {val_f1:.4f}, Validation AUC: {val_auc:.4f}')

            # Call early stopping
            early_stopping(val_auc, model, epoch)

            if args.earlystop and early_stopping.early_stop:
                print("Early stopping")
                break

        if (epoch + 1) % ckpt_interval == 0:
            checkpoint_path = os.path.join(full_output_path, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
        if args.final and (epoch + 1) == num_epochs:
            checkpoint_path = os.path.join(full_output_path, f'checkpoint.pt')
            torch.save(model.state_dict(), checkpoint_path)
    # Load the last checkpoint with the best model
    model.load_state_dict(torch.load(full_output_path+f'/checkpoint.pt'))
    if not args.final:
        print(f"Best model was saved at epoch {early_stopping.best_epoch} with val AUC = {early_stopping.val_auc_max:.4f}")

    return (model,stats) if args.final else (model,stats,early_stopping.best_epoch,early_stopping.val_auc_max)

def find_val_thresholds(model, val_loader, num_thresholds=201):
    """
    Find multiple thresholds in a single pass over a discrete set of threshold candidates.
    Returns a dict with keys:
      - default_0.5, maxF1, maxYouden, spec95, spec99
    """
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            prob = F.softmax(outputs, dim=1)[:,1]
            all_probs.extend(prob.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # We'll do a single pass through thresholds in [0,1]
    thresholds = np.linspace(0,1, num_thresholds, endpoint=True)

    # Initialize
    t_default   = 0.5
    best_f1     = -1.0
    t_f1        = 0.5
    best_youden = -1.0
    t_youden    = 0.5

    # We'll track the best specificity for 95% and 99% sensitivity
    best_spec_95  = -1.0
    t_spec95      = 0.5
    best_spec_99  = -1.0
    t_spec99      = 0.5

    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        cm = confusion_matrix(all_labels, preds, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()

        precision_ = tp/(tp+fp) if (tp+fp)>0 else 0
        recall_ = tp/(tp+fn) if (tp+fn)>0 else 0
        specificity_ = tn/(tn+fp) if (tn+fp)>0 else 0

        # F1
        f1_ = 2*precision_*recall_/(precision_+recall_) if (precision_+recall_)>0 else 0
        if f1_ > best_f1:
            best_f1 = f1_
            t_f1 = t

        # Youden
        youden_ = recall_ + specificity_ - 1
        if youden_ > best_youden:
            best_youden = youden_
            t_youden = t

        # Specificity at >=95% sensitivity
        if recall_ >= 0.95 and specificity_ > best_spec_95:
            best_spec_95 = specificity_
            t_spec95 = t

        # Specificity at >=99% sensitivity
        if recall_ >= 0.99 and specificity_ > best_spec_99:
            best_spec_99 = specificity_
            t_spec99 = t

    thresholds_dict = {
        'default_0.5': t_default,
        'maxF1': t_f1,
        'maxYouden': t_youden,
        'spec95': t_spec95,
        'spec99': t_spec99
    }
    return thresholds_dict


class GradCAM:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.features = None
        self.gradients = None
        self._target_layer()

    def _target_layer(self):
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                self.target_layer = module
                return
        raise ValueError("Layer name not found in the model")

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def _save_features(self, module, input, output):
        self.features = output

    def generate(self, input_tensor, class_idx):
        self.model.eval()  # Ensure model is in eval mode for consistent forward pass
        self.features = None
        self.gradients = None
        forward_handle = self.target_layer.register_forward_hook(self._save_features)
        backward_handle = self.target_layer.register_full_backward_hook(self._save_gradients)

        # Forward pass
        output = self.model(input_tensor)
        if isinstance(output, tuple):  # Model might return a tuple
            output = output[0]
        self.model.zero_grad()

        target = output[:, class_idx].sum()
        target.backward()

        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()

        # Gradient and feature processing
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.features.shape[1]):
            self.features[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.features, dim=1).squeeze()
        raw_cam = heatmap.cpu().detach().numpy()  # shape e.g. (7, 7) for typical ResNet-50
        raw_cam = np.maximum(raw_cam, 0)
        den = np.max(raw_cam) if np.max(raw_cam) != 0 else 1
        raw_cam /= den

        # Now do upsample
        up_cam = cv2.resize(raw_cam, (input_tensor.shape[3], input_tensor.shape[2]))
        up_cam = np.uint8(255 * up_cam)

        heatmap_colored = cv2.applyColorMap(up_cam, cv2.COLORMAP_JET)

        return raw_cam, heatmap_colored


def save_concatenated_images(original_image, heatmap, output_path, fname):
    """
    Combines the original image and the heatmap-superimposed image side by side,
    then saves to `output_path/fname`.
    """
    original_image = np.array(original_image)
    # Convert from RGB (PIL) to BGR (OpenCV)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

    # Superimpose heatmap
    superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

    # Concatenate side by side
    concatenated = cv2.hconcat([original_image, superimposed_img])

    # Ensure the output folder exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cv2.imwrite(os.path.join(output_path, fname), concatenated)

def generate_ig_attributions(model, input_tensor, class_idx, baseline=None, steps=100):
    """
    Uses Captum's IntegratedGradients to compute attributions.
    :param model: your PyTorch model (in eval mode)
    :param input_tensor: shape (1, 3, H, W)
    :param class_idx: integer class index, e.g. 0 or 1
    :param baseline: if None, uses a zero baseline (same shape as input_tensor)
    :param steps: number of steps for the IG approximation
    :return: a torch.Tensor of shape (1, 3, H, W) with IG attributions
    """
    model.eval()
    ig = IntegratedGradients(model)

    if baseline is None:
        baseline = torch.zeros_like(input_tensor)  # zero image

    # target=class_idx means: "explain how input_tensor influences logit for class_idx"
    attributions = ig.attribute(
        input_tensor,
        baselines=baseline,
        target=class_idx,
        n_steps=steps
    )
    return attributions


def save_ig_visualization(pil_img, ig_attr, output_path, fname):
    """
    Saves a 3-panel image:
      1) Original
      2) IG heatmap (grayscale or color-coded)
      3) 'Illuminated' overlay (original * IG intensities)
    """

    # Convert PIL image to BGR for consistent display with OpenCV
    original = np.array(pil_img)         # shape (H, W, 3) in RGB
    original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

    # ig_attr is shape (1, 3, H, W) from Captum
    # We want a 2D map, so sum across channels or take absolute value
    ig_attr_np = ig_attr[0].detach().cpu().numpy()  # shape (3, H, W)
    # Sum across channels (or take mean, or just use channel[0] if grayscale)
    ig_2d = np.abs(ig_attr_np).sum(axis=0)  # shape (H, W)

    # # Optionally take absolute value if you want magnitude only
    # ig_2d = np.abs(ig_2d)

    # Normalize to 0..1
    ig_min, ig_max = ig_2d.min(), ig_2d.max()
    denom = (ig_max - ig_min) if (ig_max - ig_min) != 0 else 1e-8
    ig_norm = (ig_2d - ig_min) / denom

    # (1) Panel 1 => Original BGR
    panel1 = original_bgr

    # (2) Panel 2 => IG heatmap as grayscale or color-coded
    #    Let's do a color-coded approach using OpenCV COLORMAP_JET
    ig_heatmap_8u = (ig_norm * 255).astype(np.uint8)        # shape (H, W)
    ig_heatmap_3c = cv2.applyColorMap(ig_heatmap_8u, cv2.COLORMAP_VIRIDIS)  # shape (H, W, 3) in BGR
    panel2 = ig_heatmap_3c

    # (3) Panel 3 => 'Illuminated' overlay
    #    We can do: illuminated = original_bgr * (0.2 + 0.8 * ig_norm) for example
    #    so that higher IG => brighter
    #    If your original is grayscale repeated in 3 channels, it still works
    #    but might appear somewhat uniform if your original has no color
    ig_expanded = np.stack([ig_norm]*3, axis=-1)  # shape (H, W, 3)
    # alpha = 0.2 + 0.8 * ig_expanded
    # illuminated = (original_bgr.astype(float) * alpha).astype(np.uint8)
    illuminated = (original_bgr.astype(float) * ig_expanded).astype(np.uint8)
    panel3 = illuminated

    # Concatenate horizontally: shape (H, W*3, 3)
    concatenated = cv2.hconcat([panel1, panel2, panel3])

    # Make sure output_path exists
    os.makedirs(output_path, exist_ok=True)

    # Save
    out_path = os.path.join(output_path, fname)
    cv2.imwrite(out_path, concatenated)


def save_shap_visualization(pil_img, shap_tensor, output_path, fname):
    """
    Saves a 3-panel image:
      1) Original
      2) SHAP heatmap (grayscale or color-coded)
      3) 'Illuminated' overlay (original * SHAP intensities)
      """

    # Convert PIL image to BGR (for OpenCV)
    original = np.array(pil_img)  # shape (H, W, 3) in RGB
    original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

    # If shap_tensor is a PyTorch tensor, convert to NumPy
    if hasattr(shap_tensor, 'cpu'):
        shap_tensor = shap_tensor.detach().cpu().numpy()  # shape (3, H, W)

    # 1) Sum across channels (or take mean). Then take absolute value to see magnitude
    shap_2d = np.abs(shap_tensor).sum(axis=0)  # shape (H, W)

    # 2) Normalize to [0,1]
    s_min, s_max = shap_2d.min(), shap_2d.max()
    denom = (s_max - s_min) if (s_max - s_min) != 0 else 1e-8
    shap_norm = (shap_2d - s_min) / denom

    # (A) Panel 1 => Original
    panel1 = original_bgr

    # (B) Panel 2 => SHAP heatmap (color-coded)
    shap_8u = (shap_norm * 255).astype(np.uint8)  # shape (H, W)
    shap_color = cv2.applyColorMap(shap_8u, cv2.COLORMAP_VIRIDIS)
    panel2 = shap_color

    # (C) Panel 3 => 'Illuminated' overlay
    #     We replicate your IG approach of pixelwise multiplication by shap_norm
    #     (or you can do alpha = 0.2 + 0.8 * shap_norm if you prefer)
    shap_expanded = np.stack([shap_norm]*3, axis=-1)  # shape (H, W, 3)
    illuminated = (original_bgr.astype(float) * shap_expanded).astype(np.uint8)
    panel3 = illuminated

    # Concatenate horizontally: shape (H, W*3, 3)
    concatenated = cv2.hconcat([panel1, panel2, panel3])

    # Make sure output_path exists
    os.makedirs(output_path, exist_ok=True)

    # Save
    out_path = os.path.join(output_path, fname)
    cv2.imwrite(out_path, concatenated)


def evaluate_on_test(model, test_loader, thresholds_dict, save_path):
    """
    1) Generate separate ROC and PR plots with "no-skill" lines
    2) For each threshold, compute metrics and save them to CSV
    3) Save test probabilities in a CSV file
    4) If --gradcam is passed, generate Grad-CAM heatmaps for both classes
    """
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            prob = F.softmax(outputs, dim=1)[:,1]   # Probability of the positive class
            all_probs.extend(prob.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # --- (A) ROC + PR Curves ---
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    pos_rate = np.mean(all_labels)

    # Separate ROC plot
    plt.figure(figsize=(4,4))
    plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC={roc_auc:.3f})')
    plt.plot([0,1],[0,1],'--', color='gray', label='No Skill')
    plt.xlabel('1-Specificity (FPR)')
    plt.ylabel('Sensitivity (TPR)')
    # plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    roc_path = os.path.join(save_path, "test_roc.png")
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    # Separate Precision-Recall plot
    plt.figure(figsize=(4,4))
    plt.plot(recall, precision, lw=2, label=f'PR (AUC={pr_auc:.3f})')
    plt.axhline(y=pos_rate, color='gray', linestyle='--', label=f'No Skill (AUC={pos_rate:.3f})')
    plt.xlabel('Sensitivity (Recall)')
    plt.ylabel('PPV (Precision)')
    # plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    pr_path = os.path.join(save_path, "test_pr.png")
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close()

    # --- (B) Threshold-based metrics ---
    metrics_filename = os.path.join(save_path, 'test_threshold_metrics.csv')
    confusion_matrix_filename = os.path.join(save_path, 'test_confusion_matrices.csv')
    classification_report_filename = os.path.join(save_path, 'test_classification_reports.csv')

    with open(metrics_filename, 'w', newline='') as metrics_file, \
          open(confusion_matrix_filename, 'w', newline='') as cm_file, \
          open(classification_report_filename, 'w', newline='') as report_file:

        metrics_writer = csv.writer(metrics_file)
        cm_writer = csv.writer(cm_file)
        report_writer = csv.writer(report_file)
        # Write headers
        metrics_writer.writerow(
            ['ThresholdName', 'ThresholdValue', 'Precision', 'Recall', 'Specificity', 'F1', 'Youden','AUROC','AUPRC'])
        cm_writer.writerow(['ThresholdName', 'ThresholdValue', 'TN', 'FP', 'FN', 'TP'])
        report_writer.writerow(
            ['ThresholdName', 'ThresholdValue', 'Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
        for t_name, t_val in thresholds_dict.items():
            preds = (all_probs >= t_val).astype(int)
            cm = confusion_matrix(all_labels, preds, labels=[0,1])
            tn, fp, fn, tp = cm.ravel()
            precision_ = tp/(tp+fp) if (tp+fp) else 0
            recall_ = tp/(tp+fn) if (tp+fn) else 0
            specificity_ = tn/(tn+fp) if (tn+fp) else 0
            f1_ = f1_score(all_labels, preds, average='binary', pos_label=1)
            youden_ = recall_ + specificity_ - 1

            # Save threshold metrics
            metrics_writer.writerow([t_name, f"{t_val:.3f}", f"{precision_:.3f}", f"{recall_:.3f}",f"{specificity_:.3f}",
                                     f"{f1_:.3f}", f"{youden_:.3f}",f"{roc_auc:.3f}",f"{pr_auc:.3f}"])

            # Save confusion matrix
            cm_writer.writerow([t_name, f"{t_val:.3f}", tn, fp, fn, tp])

            # Save classification report
            report = classification_report(all_labels, preds, target_names=["Negative", "Positive"], output_dict=True)
            for class_name, values in report.items():
                if isinstance(values, dict):  # Ensure we're only processing class-level metrics
                    report_writer.writerow([t_name, f"{t_val:.3f}", class_name,
                                            f"{values['precision']:.3f}", f"{values['recall']:.3f}",
                                            f"{values['f1-score']:.3f}", values['support']])

    # --- (C) Save probabilities to CSV (similar to evaluate_model) ---
    df_probs = pd.DataFrame({'Prob_Positive': all_probs})
    # If you'd also like to store Prob_Negative:
    df_probs['Prob_Negative'] = 1.0 - df_probs['Prob_Positive']
    df_probs['True_Label'] = all_labels
    # Save
    df_probs.to_csv(os.path.join(save_path, 'test_probabilities.csv'), index=False)

    if args.gradcam:
        print("Generating Grad-CAMs for both negative (0) and positive (1) classes ...")
        # 1) Instantiate a GradCAM object. Modify "layer_name" to the correct layer in your model (e.g., layer4).
        gradcam = GradCAM(model, layer_name="layer4")  # Example: for resnet50, "layer4" is the final conv block.

        # 2) Create the negative/positive save directories
        negative_cam_dir = os.path.join(save_path, "GradCAM_Negative")
        positive_cam_dir = os.path.join(save_path, "GradCAM_Positive")
        negative_cam_scores_dir = os.path.join(save_path, "GradCAM_Negative_Scores")
        positive_cam_scores_dir = os.path.join(save_path, "GradCAM_Positive_Scores")
        os.makedirs(negative_cam_dir, exist_ok=True)
        os.makedirs(positive_cam_dir, exist_ok=True)
        os.makedirs(negative_cam_scores_dir, exist_ok=True)
        os.makedirs(positive_cam_scores_dir, exist_ok=True)

        # 3) We need to run through the test set again, but one image at a time
        #    so we can produce a Grad-CAM heatmap for each of the 2 classes.
        #    We'll also need the *original* untransformed PIL image to overlay the heatmap on.
        #    The best way is to re-iterate through your test dataset in index order.
        model.eval()

        test_indices = range(len(test_dataset))  # if you want indices
        for idx in tqdm(test_indices, desc="Grad-CAM"):
            # (a) Load the original PIL image (without transform) for visualization
            #     Then apply the same transforms (besides normalization) to feed the model
            #     OR just re-use the test_dataset logic carefully.

            pil_img = Image.open(test_dataset.paths[idx]).convert('RGB')  # original untransformed image
            label = test_dataset.labels[idx].item()

            # (b) Now create the input tensor the same way the DataLoader does
            #     using val_transform. This matches your forward pass from test_loader
            input_tensor = val_transform(pil_img).unsqueeze(0).to(device)

            # (c) Generate Grad-CAM heatmaps for both class 0 and class 1:
            model.zero_grad()
            raw_cam_neg, colored_cam_neg = gradcam.generate(input_tensor, class_idx=0)
            np.save(os.path.join(negative_cam_scores_dir, f"{idx}_raw.npy"), raw_cam_neg)
            # IMPORTANT: We need to zero out grads or set `retain_graph=True`
            # Actually, in the updated code, just do:
            model.zero_grad()
            raw_cam_pos, colored_cam_pos = gradcam.generate(input_tensor, class_idx=1)
            np.save(os.path.join(positive_cam_scores_dir, f"{idx}_raw.npy"), raw_cam_pos)

            # (d) Save them (concatenate + overlay with the original image)
            save_concatenated_images(pil_img, colored_cam_neg, negative_cam_dir, f"{idx}.png")
            save_concatenated_images(pil_img, colored_cam_pos, positive_cam_dir, f"{idx}.png")

    if args.IG:
        print("Generating Integrated Gradients for both negative (0) and positive (1) classes ...")

        # Create output folders
        ig_negative_dir = os.path.join(save_path, "IG_Negative")
        ig_positive_dir = os.path.join(save_path, "IG_Positive")
        ig_neg_scores_dir = os.path.join(save_path, "IG_Negative_Scores")
        ig_pos_scores_dir = os.path.join(save_path, "IG_Positive_Scores")
        os.makedirs(ig_negative_dir, exist_ok=True)
        os.makedirs(ig_positive_dir, exist_ok=True)
        os.makedirs(ig_neg_scores_dir, exist_ok=True)
        os.makedirs(ig_pos_scores_dir, exist_ok=True)

        # Loop over each sample by index
        model.eval()
        for idx in tqdm(range(len(test_dataset)), desc="Integrated Gradients"):
            # Load the original PIL image (untransformed)
            pil_img = Image.open(test_dataset.paths[idx]).convert('RGB')
            # Possibly your data is grayscale repeated in 3 channels. That's okay if your model expects 3-channel input.

            # Create the input tensor
            input_tensor = val_transform(pil_img).unsqueeze(0).to(device)

            # For class 0 => negative
            model.zero_grad()
            ig_attr_neg = generate_ig_attributions(model, input_tensor, class_idx=0)
            # Save
            np.save(os.path.join(ig_neg_scores_dir, f"{idx}.npy"), ig_attr_neg[0].cpu().numpy())
            save_ig_visualization(pil_img, ig_attr_neg, ig_negative_dir, f"{idx}.png")

            # For class 1 => positive
            model.zero_grad()
            ig_attr_pos = generate_ig_attributions(model, input_tensor, class_idx=1)
            # Save
            np.save(os.path.join(ig_pos_scores_dir, f"{idx}.npy"), ig_attr_pos[0].cpu().numpy())
            save_ig_visualization(pil_img, ig_attr_pos, ig_positive_dir, f"{idx}.png")

    if args.shap:
        # 1) Create a small background set from the training dataset
        background_indices = np.random.choice(len(train_dataset), size=100, replace=False)
        background_data = []
        for b_idx in background_indices:
            pil_bimg = Image.open(train_dataset.paths[b_idx]).convert('RGB')
            # transform it similarly to val_transform
            tensor_b = val_transform(pil_bimg).to(device)
            background_data.append(tensor_b.cpu().numpy())  # store in CPU memory

        background_data = np.stack(background_data, axis=0)  # shape (10, C, H, W)
        shap_background = torch.tensor(background_data, dtype=torch.float).to(device)

        # 2) Initialize DeepExplainer (fall back to GradientExplainer if not supported)
        # try:
        #     shap_explainer = shap.DeepExplainer(model, shap_background)
        #     print("Using DeepExplainer for SHAP.")
        # except:
        shap_explainer = shap.GradientExplainer(model, shap_background)
        print("Using GradientExplainer (fallback).")

        # Create output folders for SHAP
        shap_negative_dir = os.path.join(save_path, "SHAP_Negative")
        shap_positive_dir = os.path.join(save_path, "SHAP_Positive")
        os.makedirs(shap_negative_dir, exist_ok=True)
        os.makedirs(shap_positive_dir, exist_ok=True)

        # Also create output folders to store raw shap scores (if you want them)
        shap_neg_scores_dir = os.path.join(save_path, "SHAP_Negative_Scores")
        shap_pos_scores_dir = os.path.join(save_path, "SHAP_Positive_Scores")
        os.makedirs(shap_neg_scores_dir, exist_ok=True)
        os.makedirs(shap_pos_scores_dir, exist_ok=True)

        # Loop over test samples
        for idx in tqdm(range(len(test_dataset)), desc="SHAP"):
            pil_img = Image.open(test_dataset.paths[idx]).convert('RGB')
            input_tensor = val_transform(pil_img).unsqueeze(0).to(device)

            # shap_values is a list of length num_classes (2 for binary)
            shap_values = shap_explainer.shap_values(input_tensor)

            # For class 0:
            shap_val_class0 = shap_values[0]  # shape (1, C, H, W)
            # save raw scores as .npy
            np.save(os.path.join(shap_neg_scores_dir, f"{idx}.npy"), shap_val_class0[0])

            # Possibly do a shap.image_plot or custom overlay
            # e.g., shap.image_plot([shap_val_class0], [input_tensor.cpu().numpy()])

            # For class 1:
            shap_val_class1 = shap_values[1]
            np.save(os.path.join(shap_pos_scores_dir, f"{idx}.npy"), shap_val_class1[0])

            # We just want the array of shape (3, H, W)
            shap_val_class0_np = shap_val_class0[0]  # shape (3, H, W)
            shap_val_class1_np = shap_val_class1[0]

            # Suppose we do for class 0 => negative
            save_shap_visualization(pil_img, shap_val_class0_np,
                                    shap_negative_dir, f"{idx}.png")

            # Class 1 => positive
            save_shap_visualization(pil_img, shap_val_class1_np,
                                    shap_positive_dir, f"{idx}.png")


if args.mode == 'train':
    # Train and Evaluate
    if args.final:
        metadata_file = os.path.join(full_metadata_path, "train_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            thresholds_dict = metadata["thresholds"]
            best_epoch = metadata["best_epoch"]
            print(f"Loaded thresholds from {metadata_file}")
        model,history = train_model(model, criterion, optimizer, num_epochs=best_epoch+1)
    else:
        model,history,best_epoch,best_val_auc = train_model(model, criterion, optimizer)

    # Number of epochs
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot for accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_f1'], label='Training F1 Score')
    if not args.final:
        plt.plot(epochs, history['val_auc'], label='Validation AUC')
        plt.title('Training F1/Validation AUROC')
    else:
        plt.title('Training F1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    # Plot for loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    if not args.final:
        plt.plot(epochs, history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
    else:
        plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(full_output_path+'/history.png')

    if not args.final:
        # Create thresholds JSON from validation set
        thresholds_dict = find_val_thresholds(model, val_loader, num_thresholds=201)
        # Save thresholds + best epoch for future usage
        metadata_file = os.path.join(full_output_path, "train_metadata.json")
        metadata = {
            "best_epoch": best_epoch,
            "thresholds": thresholds_dict,
            "val_auc": best_val_auc
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved thresholds + best epoch to {metadata_file}")

    # Evaluate on test set
    evaluate_on_test(model, test_loader, thresholds_dict, full_output_path)

elif args.mode == 'eval':
    # Suppose we just want to load the model checkpoint + thresholds
    # from a previous run. We do something like:
    model.load_state_dict(torch.load(os.path.join(full_output_path, args.checkpoint)))
    if args.final:
        # Load the thresholds + best_epoch if we want
        metadata_file = os.path.join(full_metadata_path, "train_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            thresholds_dict = metadata["thresholds"]
            print(f"Loaded thresholds from {metadata_file}")
    else:
        # Load the thresholds + best_epoch if we want
        metadata_file = os.path.join(full_output_path, "train_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            thresholds_dict = metadata["thresholds"]
            print(f"Loaded thresholds from {metadata_file}")
        else:
            thresholds_dict = {
                "default_0.5":0.5,
                "maxF1":0.5,
                "maxYouden":0.5,
                "spec95":0.5,
                "spec99":0.5
            }

    # Now do evaluations
    evaluate_on_test(model, test_loader, thresholds_dict, full_output_path)