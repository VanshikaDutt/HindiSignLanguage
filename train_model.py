
import os
import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # Adjust based on input image size
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 14 * 14)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Sign Language Dataset class
class SignLanguageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.images = []
        
        for label, cls in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if img_name.endswith('.png') or img_name.endswith('.jpg'):  # Ensure image files
                    try:
                        # Try reading the image to check its validity
                        image = cv2.imread(img_path)
                        if image is None:  # If the image cannot be read, skip it
                            print(f"Warning: Unable to read {img_path}. Skipping this file.")
                            continue
                        self.images.append((img_path, label))
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        continue

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = cv2.imread(img_path)
        if image is None:  # If the image is invalid, skip
            raise ValueError(f"Image {img_path} could not be loaded.")
        image = cv2.resize(image, (64, 64))  # Resize to a standard size
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load dataset
dataset = SignLanguageDataset('Image', transform=transform)

# Split dataset into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoader for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Get number of classes
num_classes = len(dataset.classes)

# Initialize the model
model = CNNModel(num_classes).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10  # Set the number of epochs
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")

# Save the trained model
torch.save(model.state_dict(), "hindi_sign_model.pth")
print("Model saved as hindi_sign_model.pth")
