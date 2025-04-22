import cv2
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the CNN model (same as in your training script)
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

# Load the trained model
# model = CNNModel(num_classes=11).to(device)  # Adjust num_classes based on your dataset
model = CNNModel(num_classes=11).to(device)
model.load_state_dict(torch.load("hindi_sign_model.pth"))
model.eval()  # Set the model to evaluation mode

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to match the model's input size
    transforms.ToTensor(),
])

# Initialize webcam or video source
cap = cv2.VideoCapture(0)  # 0 for the default webcam

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to RGB (OpenCV uses BGR by default)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the image (numpy array) to a PIL Image
    pil_image = Image.fromarray(rgb_frame)

    # Apply the transformations
    image = transform(pil_image).unsqueeze(0).to(device)

    # Predict the class
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # Get the predicted label
    label = predicted.item()

    # Display the predicted label on the frame
    cv2.putText(frame, f"Predicted: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with the prediction
    cv2.imshow("Prediction", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()