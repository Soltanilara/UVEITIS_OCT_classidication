import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
from sklearn.metrics import f1_score, confusion_matrix, classification_report, roc_curve, auc
from pytorch_pretrained_vit import ViT
from monai.losses.ssim_loss import SSIMLoss

# Create the parser
parser = argparse.ArgumentParser()

# Define arguments
parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train', help='Mode to run the script in: "train" or "eval"')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pt', help='Path to the model checkpoint for evaluation')
parser.add_argument('--ckpt_interval', type=int, default=10, help='Interval of epochs to save checkpoints')
parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training (default: 128)')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate (default: 0.01)')
parser.add_argument('--num_epochs', type=int, default=13, help='Number of epochs to train (default: 100)')
parser.add_argument('--earlystop', action='store_true', help='Enable early stopping')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping (default: 10)')
parser.add_argument('--protocol', type=str, default='AE', choices=['AE','finetune','lin_eval','scratch','complement'], help='Training protocol (default: finetune)')
parser.add_argument('--model', type=str, default='resnet50', help='backbone model')
parser.add_argument('--pretraining', type=str, default='swav', choices=['swav','barlowtwins','supervised','vit'], help='Training protocol (default: swav)')
parser.add_argument('--beta', type=float, default=1, help='Orthogonalization intensity for complement tuning (default: 0)')
parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
parser.add_argument('--csvpath', type=str, default='split_1', help='Path to input csv files')
parser.add_argument('--dataset_path', type=str, default='/home/amin/Datasets/New Uveitis Code/Corrected Labeling', help='Path to images')
parser.add_argument('--output_path', type=str, default='./output (split_1)', help='Path to save the output')
parser.add_argument('--loss', type=str, default='mse', choices=['ssim','mse'], help='Loss function')
parser.add_argument('--minreduce', action='store_true', help='subtract the mean of each sample from it')
parser.add_argument('--gpu', type=int, default=1, help='GPU device index to use (default: 1)')

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

folder_name = f"{args.protocol}_{args.model}_autoencoder_{f'pretraining_{args.pretraining}_' if args.protocol!='scratch' else ''}batch_{args.batch_size}_{args.loss}_lr_{args.lr}{f'_minreduce' if args.minreduce else ''}_epochs_{args.num_epochs}{f'_earlystop_patience_{args.patience}' if args.earlystop else ''}{f'_beta_{args.beta}' if args.protocol=='AE' else ''}_seed_{args.seed}"
full_output_path = os.path.join(args.output_path, folder_name)

# Check if the full_output_path exists, if not, create it
if not os.path.exists(full_output_path):
    os.makedirs(full_output_path)


def load_data(csv_file, csvpath, folder):
    df = pd.read_csv(os.path.join(csvpath,csv_file))
    df['Intensity Values'] = df['Intensity Values'].map(lambda x: eval(x))
    paths = df['Image File'].apply(lambda x: os.path.join(folder, x))
    labels = df['Intensity Values'].values
    labels = torch.tensor(np.vstack(labels)/255, dtype=torch.float)
    if args.minreduce:
        labels = labels - labels.min(dim=1).values.unsqueeze(1)
    return paths, labels

train_mean = []
train_std = []

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

if args.protocol=='scratch':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean, std=train_std),
    ])
else:
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # ResNet-50 requires 224x224 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
if args.model in {'B_16_imagenet1k'} or args.pretraining == 'vit':
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5]),
    ])

train_dataset = CustomImageDataset('train.csv', args.csvpath, args.dataset_path, transform=transform)
train_loader = DataLoader(train_dataset, # dataset to turn into iterable
    batch_size=args.batch_size, # how many samples per batch?
    shuffle=True # shuffle data every epoch?
)

# Test data
test_dataset = CustomImageDataset('test.csv', args.csvpath, args.dataset_path, transform=transform)
test_loader = DataLoader(test_dataset,
    batch_size=args.batch_size,
    shuffle=False # don't necessarily have to shuffle the testing data
)

# Validation data
val_dataset = CustomImageDataset('val.csv', args.csvpath, args.dataset_path, transform=transform)
val_loader = DataLoader(val_dataset,
    batch_size=args.batch_size,
    shuffle=False # don't necessarily have to shuffle the testing data
)


if args.protocol == 'AE':
    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            # self.encoder = models.resnet50(pretrained=False)
            self.encoder = torch.hub.load('facebookresearch/swav:main', 'resnet50')
            self.encoder.avgpool = nn.Identity()
            self.encoder.fc = nn.Identity()
            # print(self.encoder)

        def forward(self, x):
            x = self.encoder(x).view(-1,2048,16,16)
            # print(x.size())
            return x


    # class Regressor(nn.Module):
    #     def __init__(self):
    #         super(Regressor, self).__init__()
    #         self.conv = nn.Conv2d(2048, 64, kernel_size=(1, 16))
    #         self.fc = nn.Linear(1024, 333)
    #
    #     def forward(self, x):
    #         # print(x.size())
    #         x = self.conv(x)
    #         # print(x.size())
    #         x = F.relu(x)
    #         x = x.view(x.size(0), -1)  # Flatten the feature map
    #         x = self.fc(x)
    #         return x


    class Regressor(nn.Module):
        def __init__(self):
            super(Regressor, self).__init__()
            self.upconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
            self.conv1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
            self.upconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.upconv4 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
            self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.upconv5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
            self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.conv6 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
            # self.conv6 = nn.Conv2d(64, 1, kernel_size=(1,512))
            # self.conv6 = nn.Conv2d(64, 64, kernel_size=(512,1))
            # self.conv6 = nn.Conv2d(64, 1, kernel_size=(512,1))
            self.fc = nn.Linear(262144, 333)
            # self.fc = nn.Linear(32768, 333)

        def forward(self, x):
            x = self.upconv1(x)
            x = F.relu(self.conv1(x))
            x = self.upconv2(x)
            x = F.relu(self.conv2(x))
            x = self.upconv3(x)
            x = F.relu(self.conv3(x))
            x = self.upconv4(x)
            x = F.relu(self.conv4(x))
            # x = F.relu(self.upconv5(x))
            # print(x.size())
            x = self.upconv5(x)
            x = F.relu(self.conv5(x))
            x = F.relu(self.conv6(x))
            # print(x.size())
            x = x.view(x.size(0), -1)  # Flatten the feature map
            x = torch.sigmoid(self.fc(x))
            # x = torch.sigmoid(self.conv5(x))  # Use sigmoid activation to get the output in range [0, 1]
            return x

    # class Regressor(nn.Module):
    #     def __init__(self):
    #         super(Regressor, self).__init__()
    #         # self.upconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
    #         # self.conv1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
    #         # self.upconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
    #         # self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    #         # self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
    #         # self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
    #         # self.upconv4 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
    #         # self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    #         # self.upconv5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
    #         # self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    #         # self.conv6 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
    #         # # self.conv6 = nn.Conv2d(64, 1, kernel_size=(1,512))
    #         self.fc1 = nn.Linear(524288, 333)
    #         self.fc2 = nn.Linear(333,333)
    #
    #     def forward(self, x):
    #         # x = self.upconv1(x)
    #         # x = F.relu(self.conv1(x))
    #         # x = self.upconv2(x)
    #         # x = F.relu(self.conv2(x))
    #         # x = self.upconv3(x)
    #         # x = F.relu(self.conv3(x))
    #         # x = self.upconv4(x)
    #         # x = F.relu(self.conv4(x))
    #         # # x = F.relu(self.upconv5(x))
    #         # # print(x.size())
    #         # x = self.upconv5(x)
    #         # x = F.relu(self.conv5(x))
    #         # x = F.relu(self.conv6(x))
    #         # # print(x.size())
    #         x = x.view(x.size(0), -1)  # Flatten the feature map
    #         x = F.relu(self.fc1(x))
    #         x = torch.sigmoid(self.fc2(x))
    #         # x = torch.sigmoid(self.conv5(x))  # Use sigmoid activation to get the output in range [0, 1]
    #         return x


    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.upconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
            self.conv1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
            self.upconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.upconv4 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
            self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.upconv5 = nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)
            self.conv5 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

        def forward(self, x):
            x = self.upconv1(x)
            x = F.relu(self.conv1(x))
            x = self.upconv2(x)
            x = F.relu(self.conv2(x))
            x = self.upconv3(x)
            x = F.relu(self.conv3(x))
            x = self.upconv4(x)
            x = F.relu(self.conv4(x))
            x = self.upconv5(x)
            # x = self.conv5(x)
            x = torch.sigmoid(self.conv5(x))  # Use sigmoid activation to get the output in range [0, 1]
            return x


    class Autoencoder(nn.Module):
        def __init__(self):
            super(Autoencoder, self).__init__()
            self.encoder = Encoder()
            self.regressor = Regressor()
            self.decoder = Decoder()

        def forward(self, x):
            encoded = self.encoder(x)
            representation = self.regressor(encoded)
            decoded = self.decoder(encoded)
            return decoded, representation

    model = Autoencoder()

elif args.protocol == 'complement':
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
    model = models.resnet50(pretrained=False)
    # model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Adjusting for CIFAR-10 classes
else:
    if args.pretraining == 'swav':
        if args.model=='resnet50':
            model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        elif args.model=='resnet50w5':
            model = nn.Sequential(
                torch.hub.load('facebookresearch/swav:main', 'resnet50w5'),
                nn.Linear(10240,333)
            )
    elif args.pretraining == 'barlowtwins':
        model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
        # saved_weights = torch.nn.Parameter(model.conv1.weight[:, :2, :, :])
        # model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # model.conv1.weight = saved_weights
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
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 333)
        # print(model)

    if args.protocol == 'lin_eval':
        for param in model.parameters():
            param.requires_grad = False
    if args.model not in {'resnet50w5'}:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 333)  # Adjusting for CIFAR-10 classes

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# Step 3: Define Loss Function and Optimizer
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.4912,1.5088]).to(device))       # split_1
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.4821,1.5179]).to(device))       # split_2
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.4948,1.5052]).to(device))       # split_3
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.4936,1.5064]).to(device))       # split_5
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.4602,1.5398]).to(device))       # split_9
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5425,4.4575]).to(device))     # (H6P36,H12P19P43)
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.6683,4.3317]).to(device))     # (H6P36,H12P19P43) randomized trainval
if args.loss == 'ssim':
    criterion = SSIMLoss(spatial_dims=2, data_range=1.0)
elif args.loss == 'mse':
    criterion = nn.MSELoss()



# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
        self.val_f1_max = 0
        self.delta = delta

    def __call__(self, val_f1, model):
        score = val_f1

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_f1, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_f1, model)
            self.counter = 0

    def save_checkpoint(self, val_f1, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation F1 score increased ({self.val_f1_max:.6f} --> {val_f1:.6f}).  Saving model ...')
        torch.save(model.state_dict(), full_output_path+f'/checkpoint.pt')
        self.val_f1_max = val_f1

# Update the train_model function to include validation loss computation
def train_model(model, criterion, optimizer, num_epochs=args.num_epochs, ckpt_interval = args.ckpt_interval):
    early_stopping = EarlyStopping(verbose=True)
    stats = {'train_loss':[], 'train_acc':[], 'train_f1':[], 'val_loss':[], 'val_acc':[], 'val_f1':[]}
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_ssim = 0
        # train_preds, train_labels = [], []

        train_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs - 1}', leave=False)
        for inputs, labels in train_progress_bar:
            inputs = inputs.to(device)
            if args.loss == 'ssim':
                labels = labels.unsqueeze(1).expand(-1,11,-1).unsqueeze(1).to(device) #.view(labels.size(0),1,11,-1)
            elif args.loss == 'mse':
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
            elif args.protocol == 'AE':
                decoded, outputs = model(inputs)
                if args.loss == 'mse':
                    loss_regression = criterion(outputs,labels)
                    loss = loss_regression+args.beta*criterion(decoded, torch.sigmoid(inputs))
                    # print('regression: ', criterion(outputs,labels).item(), 'reconstruction: ', criterion(decoded, torch.sigmoid(inputs)).item())
                else:
                    raise Exception('Loss should be mse.')
            else:
                outputs = torch.sigmoid(model(inputs))
                if args.loss =='ssim':
                    loss = criterion(outputs.unsqueeze(1).expand(-1,11,-1).unsqueeze(1), labels)
                elif args.loss == 'mse':
                    loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            if args.loss == 'ssim':
                running_ssim += (1-loss.item())*inputs.size(0)
            elif args.loss == 'mse':
                running_ssim += inputs.size(0)/(1+loss.item())
                # if args.protocol == 'AE':
                #

            # Update progress bar
            train_progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_ssim / len(train_dataset)
        # train_f1 = f1_score(train_labels, train_preds, average='binary', pos_label=1)
        stats['train_loss'].append(epoch_loss)
        stats['train_acc'].append(epoch_acc)
        # stats['train_f1'].append(train_f1)

        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

        # Calculate the validation loss after each epoch
        val_loss = 0.0
        val_ssim = 0
        # val_preds, val_labels = [], []
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                if args.loss == 'ssim':
                    labels = labels.unsqueeze(1).expand(-1,11,-1).unsqueeze(1).to(device)
                elif args.loss == 'mse':
                    labels = labels.to(device)
                if args.protocol == 'complement':
                    outputs,_,_ = model(inputs)
                elif args.protocol == 'AE':
                    decoded, outputs = model(inputs)
                    loss_regression = criterion(outputs, labels)
                    loss = loss_regression + args.beta*criterion(decoded, torch.sigmoid(inputs))
                    val_loss += loss.item() * inputs.size(0)
                    val_ssim += inputs.size(0) / (1 + loss.item())
                else:
                    outputs = torch.sigmoid(model(inputs))
                    if args.loss == 'ssim':
                        loss = criterion(outputs.unsqueeze(1).expand(-1,11,-1).unsqueeze(1), labels)
                        val_loss += loss.item() * inputs.size(0)
                        val_ssim += (1-loss.item())*inputs.size(0)
                    elif args.loss == 'mse':
                        loss = criterion(outputs,labels)
                        val_loss += loss.item() * inputs.size(0)
                        val_ssim += inputs.size(0)/(1+loss.item())
        val_loss /= len(val_loader.dataset)
        val_acc = val_ssim / len(val_loader.dataset)
        stats['val_loss'].append(val_loss)
        stats['val_acc'].append(val_acc)
        print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc: .4f}')

        # Call early stopping
        early_stopping(val_acc, model)
        #
        # if args.earlystop and early_stopping.early_stop:
        #     print("Early stopping")
        #     break

        if (epoch + 1) % ckpt_interval == 0:
            checkpoint_path = os.path.join(full_output_path, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
    # Load the last checkpoint with the best model
    model.load_state_dict(torch.load(full_output_path+f'/checkpoint.pt'))

    return model,stats

def evaluate_model(model, test_loader):
    model.eval()
    running_ssim = 0
    true_images = np.zeros((len(test_loader.dataset),333))
    pred_images = np.zeros((len(test_loader.dataset),333))
    if args.protocol == 'AE':
        true_inputs = np.zeros((len(test_loader.dataset), 3,512,512))
        recon_images = np.zeros((len(test_loader.dataset), 3,512,512))
    eval_progress_bar = tqdm(test_loader, desc='Evaluating', leave=False)
    with torch.no_grad():
        for i,(inputs, labels) in enumerate(eval_progress_bar):
            # if i>10:
            #     break
            if args.protocol == 'AE':
                true_inputs[args.batch_size * i:args.batch_size * (i + 1), :,:,:] = inputs.numpy()
            # print(torch.max(labels))
            inputs = inputs.to(device)
            true_images[args.batch_size*i:args.batch_size*(i+1),:] = labels.numpy()
            if args.loss == 'ssim':
                labels = labels.unsqueeze(1).expand(-1,11,-1).unsqueeze(1).to(device)
            elif args.loss == 'mse':
                labels = labels.to(device)

            if args.protocol == 'complement':
                outputs, _, _ = model(inputs)
            elif args.protocol == 'AE':
                decoded,outputs = model(inputs)
                recon_images[args.batch_size * i:args.batch_size * (i + 1), :,:,:] = torch.logit(decoded).cpu().numpy()
            else:
                outputs = torch.sigmoid(model(inputs))
            pred_images[args.batch_size*i:args.batch_size*(i+1),:] = outputs.cpu().numpy()
            if args.loss == 'ssim':
                running_ssim += (1-criterion(outputs.unsqueeze(1).expand(-1,11,-1).unsqueeze(1),labels).item())*inputs.size(0)
            elif args.loss == 'mse':
                running_ssim += inputs.size(0)/(1+criterion(outputs,labels).item())

    acc = running_ssim / len(test_loader.dataset)
    print(f'Test Accuracy: {acc:.4f}')

    if not os.path.exists(os.path.join(full_output_path,args.checkpoint[:-3])):
        os.makedirs(os.path.join(full_output_path,args.checkpoint[:-3]))
    eval_output_path = os.path.join(full_output_path,args.checkpoint[:-3])
    filename = eval_output_path + "/test_metrics.csv"

    # Write metrics to CSV
    with open(filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([acc])

    # Visualize examples

    vis_indices = list(np.random.choice(range(len(true_images)),10,replace=False))
    # vis_indices = list(np.random.choice(range(10*args.batch_size),10,replace=False))
    fig, axes = plt.subplots(len(vis_indices), 2, figsize=(24, 3 * len(vis_indices)))
    fig.suptitle('Predicted and true FA intensity strips')

    for i, index in enumerate(vis_indices):
        true_im = (true_images[index]*255).astype(np.uint8).reshape(1, -1)
        pred_im = (pred_images[index]*255).astype(np.uint8).reshape(1, -1)
        #
        # # Convert 1D arrays to 2D images for visualization
        # original_image = np.array(original_values, dtype=np.uint8).reshape(1, -1)
        # resized_image = np.array(resized_values, dtype=np.uint8).reshape(1, -1)

        axes[i, 0].imshow(pred_im, cmap='gray', aspect='auto')
        # axes[i, 0].set_title(f'Original (Size: {size})')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(true_im, cmap='gray', aspect='auto')
        # axes[i, 1].set_title(f'Resized (Size: 333)')
        axes[i, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()
    plt.savefig(eval_output_path + "/sample FA strip constructions.png")

    # Visualize reconstructions

    vis_indices = list(np.random.choice(range(len(true_images)), 10, replace=False))
    fig, axes = plt.subplots(len(vis_indices), 2, figsize=(24, 3 * len(vis_indices)))
    fig.suptitle('Reconstructed and input scans')

    for i, index in enumerate(vis_indices):
        true_im = (((true_inputs[index] * np.array([0.229,0.224,0.225]).reshape(3,1,1))+np.array([0.485,0.456,0.406]).reshape(3,1,1)) * 255).astype(np.uint8).transpose(1,2,0)
        pred_im = (((recon_images[index] * np.array([0.229,0.224,0.225]).reshape(3,1,1))+np.array([0.485,0.456,0.406]).reshape(3,1,1)) * 255).astype(np.uint8).transpose(1,2,0)
        #
        # # Convert 1D arrays to 2D images for visualization
        # original_image = np.array(original_values, dtype=np.uint8).reshape(1, -1)
        # resized_image = np.array(resized_values, dtype=np.uint8).reshape(1, -1)

        axes[i, 0].imshow(pred_im, aspect='auto')
        # axes[i, 0].set_title(f'Original (Size: {size})')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(true_im, aspect='auto')
        # axes[i, 1].set_title(f'Resized (Size: 333)')
        axes[i, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()
    plt.savefig(eval_output_path + "/sample reconstructions.png")



    # # Optionally, save the classification report to a file
    # with open(eval_output_path + "/confusion_matrix.txt", 'w') as confusion_file:
    #     confusion_file.write(np.array2string(confusion))
    # with open(eval_output_path + "/classification_report.txt", 'w') as report_file:
    #     report_file.write(report)
    # fpr, tpr, _ = roc_curve(all_labels, [p[1] for p in all_probs])
    # roc_auc = auc(fpr, tpr)
    #
    # # Plot ROC curve
    # plt.figure()
    # plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    # plt.savefig(os.path.join(eval_output_path, 'ROC.png'))
    # # plt.show()
    #
    # # Save probabilities to a CSV file
    # df_probs = pd.DataFrame(all_probs, columns=['Prob_Negative', 'Prob_Positive'])
    # df_probs.insert(2, 'True_Label', all_labels)
    # df_probs.to_csv(os.path.join(eval_output_path,'test_probabilities.csv'), index=False)

if args.mode == 'train':
    # Train and Evaluate
    model,history = train_model(model, criterion, optimizer)

    # Number of epochs
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot for accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], label='Training SSIM')
    plt.plot(epochs, history['val_acc'], label='Validation SSIM')
    plt.title('Training and Validation SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()

    # Plot for loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], label='Training SSIM Loss')
    plt.plot(epochs, history['val_loss'], label='Validation SSIM Loss')
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