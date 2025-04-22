import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import os
from PIL import Image
import numpy as np
from torchvision import transforms

# Define the CNN model class
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # Assuming input is 64x64
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> [32x31x31]
        x = self.pool(F.relu(self.conv2(x)))  # -> [64x14x14]
        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Custom Dataset class compatible with directory structure
class SignLanguageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []

        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_folder):
                img_path = os.path.join(cls_folder, img_name)
                self.images.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image and an invalid label (we'll filter these later)
            img = Image.new('RGB', (64, 64), color=(0, 0, 0))
            label = -1  # Invalid label to indicate skipped image

        if self.transform:
            img = self.transform(img)

        return img, label

# Number of classes (based on subfolder names)
data_dir = 'Image'
class_names = sorted(os.listdir(data_dir))
num_classes = len(class_names)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = CNNModel(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Create dataset and DataLoader
dataset = SignLanguageDataset(data_dir, transform=transform)

# Split into train and validation
val_split = 0.2
val_size = int(val_split * len(dataset))
train_size = len(dataset) - val_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Training loop
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        # Skip invalid labels (those with value -1)
        valid_mask = labels != -1
        images, labels = images[valid_mask], labels[valid_mask]

        if len(images) == 0:  # If the batch has no valid data, skip it
            continue

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "hindi_sign_model.pth")

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        # Skip invalid labels (those with value -1)
        valid_mask = labels != -1
        images, labels = images[valid_mask], labels[valid_mask]

        if len(images) == 0:  # If the batch has no valid data, skip it
            continue

        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metrics
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

print("\n🧩 Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))