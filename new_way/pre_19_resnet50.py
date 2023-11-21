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
from function_model import SteganalysisModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import one_hot


device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpu_acces()
payload = 0.1
file_list_cover, file_list_LSB, file_list_HOGO, file_list_WOW, file_list_UNIWARD = dataload(payload)

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
    transforms.ToTensor(),
])

all_file_lists = [file_list_cover, file_list_LSB, file_list_HOGO, file_list_WOW, file_list_UNIWARD]
labels = list(range(len(all_file_lists)))
unified_dataset = UnifiedSteganographyDataset(all_file_lists, labels, transform=transform)
train_size = int(0.8 * len(unified_dataset))
val_size = len(unified_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(unified_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

steganalysis_model = SteganalysisModel()
steganalysis_model = steganalysis_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(steganalysis_model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.1, verbose=True)

early_stopping_counter = 0
best_val_accuracy = 0.0

for epoch in range(100):
    steganalysis_model.train()
    epoch_start_time = time.time()
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)  # Move data to the same device as the model
        optimizer.zero_grad()
        outputs = steganalysis_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'        loss = {loss:.3f}')

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time

    steganalysis_model.eval()
    val_predictions, val_labels = [], []
    with torch.no_grad():
        for val_images, val_true_labels in val_loader:
            val_images, val_true_labels = val_images.to(device), val_true_labels.to(device)  # Move validation data to the GPU
            val_outputs = steganalysis_model(val_images)
            val_predictions.extend(torch.argmax(val_outputs, 1).cpu().tolist())
            val_labels.extend(val_true_labels.cpu().tolist())
    val_accuracy = accuracy_score(val_labels, val_predictions)
    print(
        f'Epoch {epoch + 1}/{100} - Loss: {loss.item():.4f} - Validation Accuracy: {val_accuracy * 100:.2f}% - Epoch Time: {epoch_duration:.2f} seconds')

    scheduler.step(val_accuracy)

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
