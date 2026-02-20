import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import random
import os
import copy  # To save the best model
import csv
import cv2
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import f1_score, confusion_matrix, classification_report, roc_curve, auc

# Create the parser
parser = argparse.ArgumentParser()

# Define arguments
parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='eval', help='Mode to run the script in: "train" or "eval"')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pt', help='Path to the model checkpoint for evaluation')
parser.add_argument('--ckpt_interval', type=int, default=10, help='Interval of epochs to save checkpoints')
parser.add_argument('--batch_size', type=int, default=64, help='Input batch size for training (default: 128)')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate (default: 0.01)')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
parser.add_argument('--earlystop', action='store_true', help='Enable early stopping')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping (default: 10)')
parser.add_argument('--protocol', type=str, default='finetune', choices=['finetune','lin_eval','scratch','complement'], help='Training protocol (default: finetune)')
parser.add_argument('--model', type=str, default='resnet50', help='backbone model')
parser.add_argument('--pretraining', type=str, default='swav', choices=['swav','barlowtwins','supervised'], help='Training protocol (default: lin_eval)')
parser.add_argument('--beta', type=float, default=0.0, help='Orthogonalization intensity for complement tuning (default: 0)')
parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
parser.add_argument('--csvpath', type=str, default='split_1', help='Path to input csv files')
parser.add_argument('--dataset_path', type=str, default='/mnt/18ee5ff4-5aaf-495c-b305-9b9698c8d053/Amin/Datasets/New Uveitis Code/Dataset 09192024', help='Path to images')
parser.add_argument('--output_path', type=str, default='./output (split_1)', help='Path to save the output')
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

folder_name = f"{args.protocol}_{args.model}_{f'pretraining_{args.pretraining}_' if args.protocol!='scratch' else ''}batch_{args.batch_size}_lr_{args.lr}_epochs_{args.num_epochs}{f'_earlystop_patience_{args.patience}' if args.earlystop else ''}{f'_beta_{args.beta}' if args.protocol=='complement' else ''}_seed_{args.seed}"
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
        # transforms.Resize((224, 224)),  # ResNet-50 requires 224x224 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
                nn.Linear(10240,2)
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

    if args.protocol == 'lin_eval':
        for param in model.parameters():
            param.requires_grad = False
    if args.model not in {'resnet50w5'}:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # Adjusting for CIFAR-10 classes

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# Step 3: Define Loss Function and Optimizer
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.4912,1.5088]).to(device))       # split_1
criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.4821,1.5179]).to(device))       # split_2
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.4948,1.5052]).to(device))       # split_3
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.4936,1.5064]).to(device))       # split_5
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.4602,1.5398]).to(device))       # split_9
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5425,4.4575]).to(device))     # (H6P36,H12P19P43)
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.6683,4.3317]).to(device))     # (H6P36,H12P19P43) randomized trainval
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.55,4.45]).to(device))
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.434,9.567]).to(device))
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
        train_f1 = f1_score(train_labels, train_preds, average='binary', pos_label=1)
        stats['train_loss'].append(epoch_loss)
        stats['train_acc'].append(epoch_acc.item())
        stats['train_f1'].append(train_f1)

        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, F1: {train_f1:.4f}')

        # Calculate the validation loss after each epoch
        val_loss = 0.0
        val_corrects = 0
        val_preds, val_labels = [], []
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
                val_corrects += torch.sum(preds == labels.data)
                val_preds.extend(preds.view(-1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        val_f1 = f1_score(val_labels, val_preds, average='binary', pos_label=1)
        stats['val_loss'].append(val_loss)
        stats['val_acc'].append(val_acc.item())
        stats['val_f1'].append(val_f1)
        print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc: .4f}, Validation F1: {val_f1:.4f}')

        # Call early stopping
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

# class GradCAM:
#     def __init__(self, model, layer_name):
#         self.model = model
#         self.layer_name = layer_name
#         self._target_layer()
#         self.features = None
#         self.gradients = None
#
#     def _target_layer(self):
#         for name, module in self.model.named_modules():
#             if name == self.layer_name:
#                 self.target_layer = module
#                 return
#         raise ValueError("Layer name not found in the model")
#
#     def _get_gradients(self, handler, input, output, output_grad):
#         self.gradients = output_grad[0]
#
#     def _forward_hook(self, handler, input, output):
#         self.features = output[0]
#
#     def generate(self, input_tensor):
#         # self.model.train()
#         forward_hook = self.target_layer.register_forward_hook(self._forward_hook)
#         backward_handler = self.target_layer.register_backward_hook(self._get_gradients)
#         output = self.model(input_tensor)
#         self.model.zero_grad()
#         if isinstance(output, tuple):
#             output[0].max().backward()  # For models returning multiple outputs such as the complement protocol
#         else:
#             output.max().backward()
#         forward_hook.remove()
#         backward_handler.remove()
#         # self.model.eval()
#         gradients = self.gradients
#         pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
#         for i in range(self.features.shape[1]):
#             self.features[:, i, :, :] *= pooled_gradients[i]
#         heatmap = torch.mean(self.features, dim=1).squeeze()
#         heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
#         heatmap /= np.max(heatmap)
#         heatmap = cv2.resize(heatmap, (input_tensor.shape[-1], input_tensor.shape[-2]))
#         heatmap = np.uint8(255 * heatmap)
#         heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#         return heatmap

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

    def generate(self, input_tensor):
        self.model.eval()  # Ensure model is in eval mode for consistent forward pass
        self.features = None
        self.gradients = None
        forward_handle = self.target_layer.register_forward_hook(self._save_features)
        backward_handle = self.target_layer.register_backward_hook(self._save_gradients)

        # Forward pass
        output = self.model(input_tensor)
        if isinstance(output, tuple):  # Model might return a tuple
            output = output[0]
        self.model.zero_grad()

        # Trigger the backward pass
        output_max = output.max()
        output_max.backward()

        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()

        # Gradient and feature processing
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.features.shape[1]):
            self.features[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.features, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)
        # print(heatmap.shape,input_tensor.shape)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
        heatmap = cv2.resize(heatmap, (input_tensor.shape[3], input_tensor.shape[2]))  # note the dimension ordering
        heatmap = np.uint8(255 * heatmap)
        # print(heatmap.shape)
        if len(heatmap.shape) != 2 or heatmap.dtype != np.uint8:
            raise ValueError("Heatmap is not in the expected format.")
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # print(heatmap.transpose((2,0,1)).shape)
        return heatmap
        # return heatmap.transpose((2,0,1))

def save_concatenated_images(original_image, heatmap, output_path, fname):
    # heatmap = np.array(transforms.functional.to_pil_image(heatmap))
    # print(heatmap.shape,np.array(original_image).shape)
    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    original_image = np.array(original_image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap, 0.4,0)
    # print(original_image.shape,superimposed_img.shape)
    # print(np.max(original_image),np.max(superimposed_img),original_image.dtype,superimposed_img.dtype)
    concatenated = cv2.hconcat([original_image, superimposed_img])
    cv2.imwrite(output_path + "/" + fname, concatenated)

def evaluate_model(model, test_loader):
    if not os.path.exists(os.path.join(full_output_path,args.checkpoint[:-3])):
        os.makedirs(os.path.join(full_output_path,args.checkpoint[:-3]))
    eval_output_path = os.path.join(full_output_path,args.checkpoint[:-3])
    if not os.path.exists(os.path.join(eval_output_path,'GradCAM')):
        os.makedirs(os.path.join(eval_output_path,'GradCAM'))
    gradcam_output_path = os.path.join(eval_output_path,'GradCAM')
    filename = eval_output_path + "/test_metrics.csv"
    model.eval()
    grad_cam = GradCAM(model, 'layer4')  # Assuming you are using ResNet50, and want the last layer before FC
    running_corrects = 0
    all_preds = []
    all_labels = []
    all_probs = []
    # transformback = transforms.Compose([
    #     transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[-0.229, -0.224, -0.225]),
        # transforms.ToPILImage
    # ])

    eval_progress_bar = tqdm(test_loader, desc='Evaluating', leave=False)
    # with torch.no_grad():
    for i, (inputs, labels) in enumerate(eval_progress_bar):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # inputs.requires_grad_()

        if args.protocol == 'complement':
            outputs, _, _ = model(inputs)
        else:
            outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        probs = nn.functional.softmax(outputs, dim=1)
        running_corrects += torch.sum(preds == labels.data)
        all_preds.extend(preds.view(-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        for j in range(inputs.size(0)):
            # print(inputs[j].size())
            heatmap = grad_cam.generate(inputs[j].unsqueeze(0))
            img = (inputs[j].cpu() * torch.tensor([0.229,0.224,0.225]).reshape(3,1,1))+torch.tensor([0.485,0.456,0.406]).reshape(3,1,1)
            # img = transformback(inputs[j].cpu())
            # img = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[-0.229, -0.224, -0.225])
            img = transforms.functional.to_pil_image(img)
            # img = transforms.functional.to_pil_image(inputs[j].cpu())
            save_concatenated_images(img, heatmap, gradcam_output_path, f"{i*test_loader.batch_size+j}.png")

    acc = running_corrects.double() / len(test_loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1)
    confusion = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    print(f'Test Accuracy: {acc:.4f}')
    print(f'Test F1 Score: {f1:.4f}')
    print('Confusion Matrix:\n', confusion)
    print('Classification Report:\n', report)

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

if args.mode == 'train':
    # Train and Evaluate
    model,history = train_model(model, criterion, optimizer)

    # Number of epochs
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot for accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_f1'], label='Training F1 Score')
    plt.plot(epochs, history['val_f1'], label='Validation F1 Score')
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