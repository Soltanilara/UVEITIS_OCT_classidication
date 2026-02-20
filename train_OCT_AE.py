import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import f1_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from pytorch_pretrained_vit import ViT

# Create the parser
parser = argparse.ArgumentParser()

# Define arguments
parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train', help='Mode to run the script in: "train" or "eval"')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pt', help='Path to the model checkpoint for evaluation')
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
parser.add_argument('--loss', type=str, default='CE', choices=['CE','focal'], help='Loss function (default: CE)')
parser.add_argument('--unweighted', action='store_true', help='Enable for unweighted loss function')
parser.add_argument('--gamma', type=float, default=2.0, help='Focus parameter for focal loss (default: 2.0)')
parser.add_argument('--protocol', type=str, default='finetune', choices=['finetune','lin_eval','scratch','complement'], help='Training protocol (default: finetune)')
parser.add_argument('--model', type=str, default='resnet50', help='backbone model')
parser.add_argument('--pretraining', type=str, default='swav', choices=['swav','barlowtwins','supervised','vit','dino'], help='Training protocol (default: swav)')
parser.add_argument('--beta', type=float, default=0.0, help='Orthogonalization intensity for complement tuning (default: 0)')
parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
parser.add_argument('--csvpath', type=str, default='split_1', help='Path to input csv files')
parser.add_argument('--dataset_path', type=str, default='Dataset 01032025', help='Path to images')
parser.add_argument('--output_path', type=str, default='output_AE_split_1', help='Path to save the output')
parser.add_argument('--gpu', type=int, default=0, help='GPU device index to use (default: 1)')

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
               f"{args.loss}_loss_{str(args.gamma)+'_' if args.loss=='focal' else ''}"
               f"{f'unweighted_' if args.unweighted else ''}"
               f"batch_{args.batch_size}_lr_{args.lr}_{args.warmup_epochs}_{args.T_0}_{args.T_multi}_{args.eta_min}"
               f"_epochs_{args.num_epochs}{f'_hflip' if args.hflip else ''}{f'_elastic' if args.elastic else ''}"
               f"{f'_brightness' if args.brightness else ''}{f'_contrast' if args.contrast else ''}"
               f"{f'_gnoise' if args.gnoise else ''}_beta_{args.beta}"
               f"_seed_{args.seed}")
full_output_path = os.path.join(args.output_path, folder_name)

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
        transforms.Resize((512, 512)),  # ResNet-50 requires 224x224 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5]),
        # transforms.Normalize(mean=train_mean, std=train_std),
    ])
else:
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),  # ResNet-50 requires 224x224 input size
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
    transform_list.append(transforms.Resize((512, 512)))
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


train_dataset = CustomImageDataset('train.csv', args.csvpath, args.dataset_path, transform=transform)
train_loader = DataLoader(train_dataset, # dataset to turn into iterable
    batch_size=args.batch_size, # how many samples per batch?
    shuffle=True # shuffle data every epoch?
)

# Test data
test_dataset = CustomImageDataset('test.csv', args.csvpath, args.dataset_path, transform=val_transform)
test_loader = DataLoader(test_dataset,
    batch_size=args.batch_size,
    shuffle=False # don't necessarily have to shuffle the testing data
)

# Validation data
val_dataset = CustomImageDataset('val.csv', args.csvpath, args.dataset_path, transform=val_transform)
val_loader = DataLoader(val_dataset,
    batch_size=args.batch_size,
    shuffle=False # don't necessarily have to shuffle the testing data
)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        if args.protocol=='scratch':
            self.encoder = models.resnet50(pretrained=False)
        elif args.protocol=='finetune' and args.pretraining=='swav':
            self.encoder = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        self.encoder.avgpool = nn.Identity()
        self.encoder.fc = nn.Identity()
    def forward(self, x):
        x = self.encoder(x).view(-1,2048,16,16)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 2),
        )
    def forward(self, x):
        return self.classifier(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            # Input: 2048 x 16 x 16
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),  # Output: 1024 x 32 x 32
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),   # Output: 512 x 64 x 64
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),    # Output: 256 x 128 x 128
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),    # Output: 128 x 256 x 256
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),     # Output: 64 x 512 x 512
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),                # Output: 3 x 512 x 512
            nn.Sigmoid()  # Output range [0, 1] for normalized images
        )
    def forward(self, x):
        return self.decoder(x)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.classifier = Classifier()
        self.decoder = Decoder()
    def forward(self, x):
        encoded = self.encoder(x)
        logits = self.classifier(encoded)
        decoded = self.decoder(encoded)
        return decoded, logits

model = Autoencoder()

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


classWeights = list(np.load(args.csvpath +'/classWeights.npy').astype(np.float32))
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

recon_criterion = nn.MSELoss()

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
        self.val_auc_max = 0
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
            print(f'Validation AUC increased ({self.val_auc_max:.6f} --> {val_auc:.6f}).  Saving model ...')
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
            decoded, outputs = model(inputs)
            loss_classification = criterion(outputs, labels)
            loss_reconstruction = recon_criterion(decoded, torch.sigmoid(inputs))
            # print('regression: ', loss_classification.item(), 'reconstruction: ', loss_reconstruction.item())
            loss = loss_classification + args.beta * loss_reconstruction
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
                decoded, outputs = model(inputs)
                loss = criterion(outputs, labels) + args.beta * recon_criterion(decoded, torch.sigmoid(inputs))
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
        early_stopping(val_auc, model)

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
    true_inputs = np.zeros((len(test_loader.dataset), 3, 512, 512))
    recon_images = np.zeros((len(test_loader.dataset), 3, 512, 512))

    eval_progress_bar = tqdm(test_loader, desc='Evaluating', leave=False)
    with torch.no_grad():
        for i,(inputs, labels) in enumerate(eval_progress_bar):
            true_inputs[args.batch_size * i:args.batch_size * (i + 1), :, :, :] = inputs.numpy()
            inputs = inputs.to(device)
            labels = labels.to(device)

            decoded, outputs = model(inputs)
            recon_images[args.batch_size * i:args.batch_size * (i + 1), :, :, :] = torch.logit(decoded).cpu().numpy()
            _, preds = torch.max(outputs, 1)
            probs = nn.functional.softmax(outputs, dim=1)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = running_corrects.double() / len(test_loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1)
    confusion = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    print(f'Test Accuracy: {acc:.4f}')
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
        csvwriter.writerow([acc, f1])
    # Optionally, save the classification report to a file
    with open(eval_output_path + "/confusion_matrix.txt", 'w') as confusion_file:
        confusion_file.write(np.array2string(confusion))
    with open(eval_output_path + "/classification_report.txt", 'w') as report_file:
        report_file.write(report)
    fpr, tpr, _ = roc_curve(all_labels, [p[1] for p in all_probs])
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

    # Save probabilities to a CSV file
    df_probs = pd.DataFrame(all_probs, columns=['Prob_Negative', 'Prob_Positive'])
    df_probs.insert(2, 'True_Label', all_labels)
    df_probs.to_csv(os.path.join(eval_output_path,'test_probabilities.csv'), index=False)

    # Visualize reconstructions

    if args.protocol == 'scratch':
        mean, sd = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    elif args.protocol == 'finetune' and args.pretraining == 'swav':
        mean, sd = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # Select 10 random sample indices
    vis_indices = list(np.random.choice(range(len(true_inputs)), 10, replace=False))

    # Create a plot to visualize the images
    fig, axes = plt.subplots(2, len(vis_indices), figsize=(32, 8))
    # fig.suptitle('Input And Reconstructed Scans', fontsize=24, y=0.8)

    # Plot each pair of images with proper resolution and spacing
    for i, index in enumerate(vis_indices):
        true_im = (((true_inputs[index] * np.array(sd).reshape(3, 1, 1)) + np.array(
            mean).reshape(3, 1, 1)) * 255).astype(np.uint8).transpose(1, 2, 0)
        pred_im = (((recon_images[index] * np.array(sd).reshape(3, 1, 1)) + np.array(
            mean).reshape(3, 1, 1)) * 255).astype(np.uint8).transpose(1, 2, 0)

        # Plot true input image
        axes[0, i].imshow(true_im)
        if i == 0:
            axes[0, i].text(-100, 256, 'Input', fontsize=20, rotation=90, verticalalignment='center')
        axes[0, i].axis('off')
        axes[0, i].set_anchor('S')

        # Plot reconstructed image
        axes[1, i].imshow(pred_im)
        if i == 0:
            axes[1, i].text(-100, 256, 'Reconstruction', fontsize=20, rotation=90, verticalalignment='center')
        axes[1, i].axis('off')
        axes[1, i].set_anchor('N')

    # Adjust the layout to prevent overlapping
    plt.subplots_adjust(hspace=0.1, wspace=0.2)

    # Display the plot
    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(eval_output_path + "/sample_reconstructions.png", dpi=216)

if args.mode == 'train':
    # Train and Evaluate
    model,history = train_model(model, criterion, optimizer)

    # Number of epochs
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot for accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_f1'], label='Training F1 Score')
    plt.plot(epochs, history['val_auc'], label='Validation AUC')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
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