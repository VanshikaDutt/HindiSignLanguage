# import cv2
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms
# from PIL import Image

# # Set the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # List of gesture class names (make sure it's exactly 11 if num_classes=11)
# class_names = ['A', 'AA', 'E', 'EE', 'I', 'O', 'OO', 'U', 'ae', 'au', 'anuswar']

# # Define the CNN model (same as used during training)
# class CNNModel(nn.Module):
#     def __init__(self, num_classes):
#         super(CNNModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, 3)
#         self.fc1 = nn.Linear(64 * 14 * 14, 128)
#         self.fc2 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 14 * 14)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # Load the model
# model = CNNModel(num_classes=11)
# model.load_state_dict(torch.load("hindi_sign_model.pth", map_location=device))
# model.to(device)
# model.eval()

# # Define transformation (must match training)
# transform = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# print("Press 'q' to quit.")
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Flip frame (optional, feels more natural)
#     frame = cv2.flip(frame, 1)

#     # Define ROI (Region of Interest) for hand
#     x1, y1, x2, y2 = 100, 100, 300, 300
#     roi = frame[y1:y2, x1:x2]
#     roi_display = roi.copy()

#     # Preprocess ROI for prediction
#     img_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
#     img_tensor = transform(img_pil).unsqueeze(0).to(device)

#     # Make prediction
#     with torch.no_grad():
#         output = model(img_tensor)
#         _, predicted = torch.max(output, 1)
#         predicted_label = class_names[predicted.item()]
#         confidence = torch.softmax(output, dim=1).max().item()

#     # Display results
#     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#     cv2.putText(frame, f"Prediction: {predicted_label} ({confidence*100:.1f}%)",
#                 (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)

#     cv2.imshow("Hindi Sign Language Recognition", frame)

#     # Press 'q' to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()












import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For 10 classes (matches your trained model)
class_names = ['A', 'AA', 'E', 'EE', 'I', 'O', 'OO', 'U', 'ae', 'au']

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNNModel(num_classes=10)
model.load_state_dict(torch.load("hindi_sign_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    img_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        predicted_label = class_names[predicted.item()]
        confidence = torch.softmax(output, dim=1).max().item()

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, f"Prediction: {predicted_label} ({confidence*100:.1f}%)",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)

    cv2.imshow("Hindi Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
