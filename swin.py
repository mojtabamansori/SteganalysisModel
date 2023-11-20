import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import time
import os
import glob
from PIL import Image
import timm  # Install using: pip install timm
from tqdm import tqdm
import numpy as np

payload = 0.4
batch_size = 8

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

# Add some preprocessing methods

class CustomTransform:
    def __call__(self, image):
        image = self.add_neighbor_difference(image)

        # Your custom preprocessing logic here
        # For example, you can add normalization, data augmentation, etc.
        transform = transforms.Compose([
            transforms.Resize((384, 384)),  # Swin Transformer input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image)

    def add_neighbor_difference(self, image):
        img_array = np.array(image)

        diff_matrix = np.zeros_like(img_array, dtype=np.uint8)
        for i in range(1, img_array.shape[0] - 1):
            for j in range(1, img_array.shape[1] - 1):
                diff_matrix[i, j] = (
                        abs(img_array[i, j] - img_array[i - 1, j]) +
                        abs(img_array[i, j] - img_array[i + 1, j]) +
                        abs(img_array[i, j] - img_array[i, j - 1]) +
                        abs(img_array[i, j] - img_array[i, j + 1])
                )

        for channel in range(diff_matrix.shape[2]):
            img_array[:, :, channel] += diff_matrix[:, :, channel]

        return Image.fromarray(np.uint8(img_array))

transform = CustomTransform()

all_file_lists = [file_list_cover, file_list_LSB, file_list_HOGO, file_list_WOW, file_list_UNIWARD]

labels = list(range(len(all_file_lists)))

unified_dataset = UnifiedSteganographyDataset(all_file_lists, labels, transform=transform)

train_dataset = UnifiedSteganographyDataset(all_file_lists, labels, transform=transform)
val_dataset = UnifiedSteganographyDataset(all_file_lists, labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


class SwinSteganalysisModel(nn.Module):
    def __init__(self, num_classes=len(all_file_lists), pretrained=True):
        super(SwinSteganalysisModel, self).__init__()
        self.swin_transformer = timm.create_model('swin_large_patch4_window12_384', pretrained=pretrained)
        self.swin_transformer.head = nn.Linear(self.swin_transformer.head.in_features, num_classes)

    def forward(self, x):
        x = self.swin_transformer(x)
        return x

swin_steganalysis_model = SwinSteganalysisModel()
swin_steganalysis_model = swin_steganalysis_model.to('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(swin_steganalysis_model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)

for epoch in range(15):
    # Training
    swin_steganalysis_model.train()

    # Record the start time
    epoch_start_time = time.time()

    for images, labels in tqdm(train_loader):
        images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'),\
            labels.to('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer.zero_grad()
        outputs = swin_steganalysis_model(images)
        loss = criterion(outputs, labels)
        print(loss)
        loss.backward()
        optimizer.step()

    # Calculate the training time for the epoch
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time

    # Validation
    swin_steganalysis_model.eval()
    val_predictions, val_labels = [], []
    with torch.no_grad():
        for val_images, val_true_labels in val_loader:
            val_images, val_true_labels = val_images.to('cuda' if torch.cuda.is_available() else 'cpu'),\
                val_true_labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            val_outputs = swin_steganalysis_model(val_images)
            val_predictions.extend(torch.argmax(val_outputs, 1).cpu().tolist())
            val_labels.extend(val_true_labels.cpu().tolist())

    # Calculate and print accuracy
    val_accuracy = accuracy_score(val_labels, val_predictions)
    print(
        f'Epoch {epoch + 1}/{5} - Loss: {loss.item():.4f} - Validation Accuracy: {val_accuracy * 100:.2f}% - Epoch Time: {epoch_duration:.2f} seconds')

    # Adjust learning rate based on validation accuracy
    scheduler.step(val_accuracy)

# Save the trained model if needed
torch.save(swin_steganalysis_model.state_dict(), 'swin_steganalysis_model.pth')
