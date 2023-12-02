import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
import time
from PIL import Image
from tqdm import tqdm
from function_gpu_accessibility import gpu_acces
from function_dataload import dataload
import matplotlib.pyplot as plt
from model import YeNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
gpu_acces()
payload = 0.4
file_list_cover, file_list_WOW, file_list_UNIWARD = dataload(payload)


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
        image = np.array(Image.open(img_path), dtype=np.float32)
        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.ToTensor(),
])

all_file_lists = [file_list_cover, file_list_WOW, file_list_UNIWARD]
labels = list(range(len(all_file_lists)))
unified_dataset = UnifiedSteganographyDataset(all_file_lists, labels, transform=transform)
train_size = int(0.8 * len(unified_dataset))
val_size = len(unified_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(unified_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

steganalysis_model = YeNet()
steganalysis_model = steganalysis_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(steganalysis_model.parameters(), lr=0.001, momentum=0.95)

early_stopping_counter = 0
best_val_accuracy = 0.0

for epoch in range(100):

    steganalysis_model.train()
    epoch_start_time = time.time()
    total_loss = 0.0
    correct = [0] * len(all_file_lists)
    total_samples = [0] * len(all_file_lists)

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = steganalysis_model(images)
        loss = criterion(outputs, labels)
        print(f'    loss = {loss}')
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calculate training accuracy for each class
        _, predicted = torch.max(outputs.data, 1)
        for i in range(len(all_file_lists)):
            total_samples[i] += (labels == i).sum().item()
            correct[i] += (predicted == labels).logical_and(labels == i).sum().item()

        # Print training accuracy for each class
        for i in range(len(all_file_lists)):
            accuracy = correct[i] / total_samples[i] if total_samples[i] != 0 else 0


    average_loss = total_loss / len(train_loader)
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time

    steganalysis_model.eval()
    val_predictions, val_labels = [], []
    class_correct = [0] * len(all_file_lists)
    class_total = [0] * len(all_file_lists)

    with torch.no_grad():
        for val_images, val_true_labels in val_loader:
            val_images, val_true_labels = val_images.to(device), val_true_labels.to(device)
            val_outputs = steganalysis_model(val_images)
            val_predictions.extend(torch.argmax(val_outputs, 1).cpu().tolist())
            val_labels.extend(val_true_labels.cpu().tolist())

            # Compute accuracy for each class
            for i in range(len(all_file_lists)):
                class_correct[i] += ((torch.argmax(val_outputs, 1) == val_true_labels) & (val_true_labels == i)).sum().item()
                class_total[i] += (val_true_labels == i).sum().item()

    val_accuracy = accuracy_score(val_labels, val_predictions)
    print(f'Epoch {epoch + 1}/{100} - Loss: {average_loss:.4f} - Training Accuracy: {accuracy * 100:.2f}% - Validation Accuracy: {val_accuracy * 100:.2f}% - Epoch Time: {epoch_duration:.2f} seconds')

    for i in range(len(all_file_lists)):
        class_acc = class_correct[i] / class_total[i] if class_total[i] != 0 else 0
        print(f'Class {i} Accuracy: {class_acc * 100:.2f}%')

    # Check for early stopping
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    # Save model if validation accuracy does not improve for 30 consecutive epochs
    if early_stopping_counter >= 30:
        print(f'Early stopping after {epoch + 1} epochs due to no improvement in validation accuracy.')
        break

    # Save model checkpoint every 30 epochs
    if (epoch + 1) % 30 == 0:
        torch.save(steganalysis_model.state_dict(), f'steganalysis_model_resnet_epoch_{epoch + 1}.pth')

torch.save(steganalysis_model.state_dict(), 'steganalysis_model_resnet.pth')
