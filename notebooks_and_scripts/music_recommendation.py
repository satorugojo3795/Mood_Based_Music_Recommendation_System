import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import pandas as pd

# ------------------------------
# 🔹 Load the Songs Dataset
# ------------------------------
df_with_clusters = pd.read_csv("../data/df_with_clusters.csv")
# Make sure df_with_clusters has at least the columns: 'mood' and 'song_name'

# ------------------------------
# 🔹 Define the CNN Model (MUST MATCH TRAINED MODEL)
# ------------------------------
class CNNModel(nn.Module):
    def __init__(self, num_classes=7):  # FER2013 has 7 emotions
        super(CNNModel, self).__init__()

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.25)

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout3 = nn.Dropout(0.25)

        # Fully Connected Layers
        self.fc1 = nn.Linear(32 * 6 * 6, 256)  # Adjusted for 48x48 images
        self.fc_dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)  # Output layer

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)

        x = torch.flatten(x, start_dim=1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = self.fc2(x)  # Output logits
        return x

# ------------------------------
# 🔹 Load the Trained Model
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(num_classes=7).to(device)

# Load saved weights (note: we load the entire model as per your requirement)
# model = torch.load("../models/emotion_model_1_full.pth", map_location=device, weights_only=False)
# model.eval()
# Load only the state dict
model.load_state_dict(torch.load("../Backend/models/emotion_model_2_state.pth", map_location=device))
model.eval()

# Emotion labels (order must match your training)
emotion_classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

print("✅ Model Loaded Successfully!")

# ------------------------------
# 🔹 Initialize OpenCV Face Detector
# ------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ------------------------------
# 🔹 Define Image Preprocessing for CNN
# ------------------------------
transform = transforms.Compose([
    transforms.Grayscale(),             # Ensure single-channel input
    transforms.Resize((48, 48)),          # Resize for model input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
])

# ------------------------------
# 🔹 Initialize Variables for Emotion Capture Logic
# ------------------------------
captured_emotion = None
emotion_capture_time = None

# ------------------------------
# 🔹 Start Real-Time Webcam Feed
# ------------------------------
cap = cv2.VideoCapture(0)  # Open webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Process each detected face
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]  # Extract face ROI
        face_pil = Image.fromarray(face)  # Convert to PIL Image
        face_tensor = transform(face_pil).unsqueeze(0).to(device)  # Preprocess & add batch dimension

        # Predict Emotion
        with torch.no_grad():
            output = model(face_tensor)
            pred = torch.argmax(output, dim=1).item()

        # Get emotion label
        emotion_text = emotion_classes[pred]

        # Draw rectangle and emotion label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Capture the first detected emotion and record the time if not already captured
        if captured_emotion is None:
            captured_emotion = emotion_text
            emotion_capture_time = time.time()
            print("Captured first emotion:", captured_emotion)

    # Display the frame with face detection and emotion label
    cv2.imshow("Real-Time Emotion Detection", frame)

    # If an emotion was captured, check if 5 seconds have passed
    if captured_emotion is not None and (time.time() - emotion_capture_time) >= 5:
        detected_mood = captured_emotion  # Use the captured emotion as mood
        filtered_df = df_with_clusters[df_with_clusters['mood'] == detected_mood]

        if not filtered_df.empty:
            # In case there are fewer than 5 songs for a mood, sample the available ones
            num_samples = min(5, len(filtered_df))
            random_songs = filtered_df.sample(num_samples)['song_name']
            print("\nDetected mood:", detected_mood)
            print("Recommended songs:")
            for song in random_songs:
                print(" -", song)
        else:
            print("\nDetected mood:", detected_mood)
            print("No songs found for this mood in the dataframe.")

        # After recommending songs, break the loop to close the stream
        break

    # Option to quit early by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🎥 Webcam Stream Closed.")
