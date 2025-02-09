# Mood Melody

**Mood Melody** is a full-stack emotion-based music recommendation web application. It uses your webcam to capture your facial expression, detects your emotion using a pretrained Convolutional Neural Network (CNN), and returns personalized song recommendations based on your detected mood. The backend is built with FastAPI, and the frontend is built with React.js—both designed with a modern, music-inspired aesthetic.

## Features

- **Emotion Detection:** Uses a CNN model to detect facial emotion from a webcam snapshot.
- **Personalized Recommendations:** Filters a dataset of songs to provide recommendations that match your mood.
- **Modern UI:** A beautifully designed, music-inspired interface built with React.
- **Full-Stack Implementation:** FastAPI serves as the backend API while React handles client-side interactions.

# Create new virtual environment
python -m venv .venv
python3 -m venv .venv


## Activate the virtual environment
.venv\Scripts\Activate
.venv\Scripts\activate.bat
source .venv/bin/activate

## Installed required Dependencies
pip install -r requirements.txt

## Run the backend
uvicorn main:app --reload

## run the frontend
npm start
