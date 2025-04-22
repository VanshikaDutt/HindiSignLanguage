import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# Define the number of classes (e.g., 11 for your task)
num_classes = 11
# num_classes = 10

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> [32x31x31]
        x = self.pool(F.relu(self.conv2(x)))  # -> [64x14x14]
        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the custom dataset for loading images and labels
class CustomDataset(Dataset):
    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform
        self.image_files = os.listdir(image_dir)  # List of image file names

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name)  # Open image
        label = self.labels[idx]     # Get the label for the image

        if self.transform:
            image = self.transform(image)  # Apply transformations

        return image, label

# Assuming images are in 'train_data' directory and labels are available
image_dir = 'path_to_your_images'  # Replace with actual path to images

# Correct the label list to ensure it aligns with the images in the directory
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Example labels for each image, update as needed

# Define the transformations to apply to the images
transform = transforms.Compose([
    transforms.Resize((64, 64)),        # Resize the image to 64x64
    transforms.ToTensor(),              # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Create a DataLoader for the dataset
train_dataset = CustomDataset(image_dir, labels, transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()        # Zero the gradients
        outputs = model(images)      # Forward pass
        loss = criterion(outputs, labels)  # Calculate the loss
        loss.backward()              # Backward pass (compute gradients)
        optimizer.step()             # Update model parameters
        running_loss += loss.item()  # Accumulate loss

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "hindi_sign_model.pth")
