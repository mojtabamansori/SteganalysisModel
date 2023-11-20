import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50
from sklearn.metrics import accuracy_score
import time
import os
import glob
from PIL import Image

# Check for GPU availability
if torch.cuda.is_available():
    # Get the number of available GPUs
    gpu_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {gpu_count}")

    # Get the name of the GPU
    gpu_name = torch.cuda.get_device_name(0)  # Here, 0 represents the first GPU
    print(f"GPU Name: {gpu_name}")
else:
    print("GPU not found. PyTorch will use the CPU.")

payload = 0.1

ste_model_cover = 'cover'
ste_model_LSB = 'LSB'
ste_model_HOGO = 'HOGO'
ste_model_WOW = 'WOW'
ste_model_UNIWARD = 'UNIWARD'

base_folder_cover = f'C:/Users/ns/Desktop/paksersht/Code Email/{ste_model_cover}/{payload}'
base_folder_LSB = f'C:/Users/ns/Desktop/paksersht/Code Email/{ste_model_LSB}/{payload}'
base_folder_HOGO = f'C:/Users/ns/Desktop/paksersht/Code Email/{ste_model_HOGO}/{payload}'
base_folder_WOW = f'C:/Users/ns/Desktop/paksersht/Code Email/{ste_model_WOW}/{payload}'
base_folder_UNIWARD = f'C:/Users/ns/Desktop/paksersht/Code Email/{ste_model_UNIWARD}/{payload}'

file_list_cover = glob.glob(os.path.join(base_folder_cover, '*'))
file_list_LSB = glob.glob(os.path.join(base_folder_LSB, '*'))
file_list_HOGO = glob.glob(os.path.join(base_folder_HOGO, '*'))
file_list_WOW = glob.glob(os.path.join(base_folder_WOW, '*'))
file_list_UNIWARD = glob.glob(os.path.join(base_folder_UNIWARD, '*'))

class UnifiedSteganographyDataset(Dataset):
    def __init__(self, file_lists, labels, transform=None):
        self.file_lists = file_lists
        self.labels = labels
        self.transform = transform

        # Combine all file lists into a single list
        self.all_files = []
        for i, file_list in enumerate(file_lists):
            self.all_files.extend([(file_path, i) for file_path in file_list])

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        img_path, label = self.all_files[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet-50 input size
    transforms.ToTensor(),
])

all_file_lists = [file_list_cover, file_list_LSB, file_list_HOGO, file_list_WOW, file_list_UNIWARD]

labels = list(range(len(all_file_lists)))

unified_dataset = UnifiedSteganographyDataset(all_file_lists, labels, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(unified_dataset))
val_size = len(unified_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(unified_dataset, [train_size, val_size])

# Create data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Define the SteganalysisModel using ResNet-50
class SteganalysisModel(nn.Module):
    def __init__(self, num_classes=len(all_file_lists)):
        super(SteganalysisModel, self).__init__()
        resnet = resnet50(pretrained=True)
        # Remove the fully connected layers of ResNet-50
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])

        # Add custom fully connected layers for classification
        self.fc1 = nn.Linear(2048, 512)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet_features(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
steganalysis_model = SteganalysisModel()
steganalysis_model = steganalysis_model.to('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(steganalysis_model.parameters(), lr=0.001)

# Training and Validation loop
for epoch in range(5):
    # Training
    steganalysis_model.train()

    # Record the start time
    epoch_start_time = time.time()

    for images, labels in train_loader:
        images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer.zero_grad()
        outputs = steganalysis_model(images)
        loss = criterion(outputs, labels)
        print(f'\rloss = {loss}', end='')
        loss.backward()
        optimizer.step()

    # Calculate the training time for the epoch
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time

    # Validation
    steganalysis_model.eval()
    val_predictions, val_labels = [], []
    with torch.no_grad():
        for val_images, val_true_labels in val_loader:
            val_images, val_true_labels = val_images.to('cuda' if torch.cuda.is_available() else 'cpu'), val_true_labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            val_outputs = steganalysis_model(val_images)
            val_predictions.extend(torch.argmax(val_outputs, 1).cpu().tolist())
            val_labels.extend(val_true_labels.cpu().tolist())

    # Calculate and print accuracy
    val_accuracy = accuracy_score(val_labels, val_predictions)
    print(
        f'Epoch {epoch + 1}/{5} - Loss: {loss.item():.4f} - Validation Accuracy: {val_accuracy * 100:.2f}% - Epoch Time: {epoch_duration:.2f} seconds')

# Save the trained model if needed
torch.save(steganalysis_model.state_dict(), 'steganalysis_model_resnet.pth')
