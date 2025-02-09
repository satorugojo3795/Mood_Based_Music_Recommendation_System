# backend/main.py
import io
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import base64
from io import BytesIO

# Import the model from utils/cnn_model.py
from utils.cnn_model import CNNModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Songs Dataset
df_with_clusters = pd.read_csv("data/df_with_clusters.csv")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(num_classes=7).to(device)

# Load only the state dict (make sure you saved your model as state dict)
model.load_state_dict(torch.load("models/emotion_model_2_state.pth", map_location=device))
model.eval()

# Emotion labels (must match training order)
emotion_classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Define Image Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Initialize OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@app.post("/detect")
async def detect_emotion(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR

    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return {"error": "No face detected. Please try again with a clear image."}

    # Use the first detected face
    (x, y, w, h) = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    face_pil = Image.fromarray(face_roi)
    face_tensor = transform(face_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(face_tensor)
        pred = torch.argmax(output, dim=1).item()

    detected_emotion = emotion_classes[pred]

    # Get song recommendations
    filtered_df = df_with_clusters[df_with_clusters['mood'] == detected_emotion]
    if not filtered_df.empty:
        num_samples = min(5, len(filtered_df))
        random_songs = list(filtered_df.sample(num_samples)['song_name'])
    else:
        random_songs = []

    # Convert the face ROI to a Base64-encoded JPEG image
    pil_face_image = Image.fromarray(face_roi)
    buffer = BytesIO()
    pil_face_image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{img_str}"

    return {
        "mood": detected_emotion,
        "songs": random_songs,
        "face_image": data_url
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
