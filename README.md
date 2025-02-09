# Mood Melody

**Mood Melody** is a full-stack emotion-based music recommendation web application. It uses your webcam to capture your facial expression, detects your emotion using a pretrained Convolutional Neural Network (CNN), and returns personalized song recommendations based on your detected mood. The backend is built with FastAPI, and the frontend is built with React.js—both designed with a modern, music-inspired aesthetic.

## Features

- **Emotion Detection:** Uses a CNN model to detect facial emotion from a webcam snapshot.
- **Personalized Recommendations:** Filters a dataset of songs to provide recommendations that match your mood.
- **Modern UI:** A beautifully designed, music-inspired interface built with React.
- **Full-Stack Implementation:** FastAPI serves as the backend API while React handles client-side interactions.

## Project Structure

```
mood-melody/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── models/
│   │   └── emotion_model_2_state.pth
│   ├── data/
│   │   └── df_with_clusters.csv
│   └── utils/
│       └── cnn_model.py
└── frontend/
    ├── public/
    │   └── index.html
    ├── src/
    │   ├── App.js
    │   ├── App.css
    │   ├── index.js
    │   └── index.css
    └── package.json
```

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

- **Python 3.7+**
- **Node.js** (with npm)
- **Git**

---

## Setting Up the Backend

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/satorugojo3795/Mood_Based_Music_Recommendation_System.git
   cd Mood_Based_Music_Recommendation_System/backend
   ```

2. **Create a Virtual Environment:**

   On macOS/Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   On Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Backend Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Backend Server:**

   ```bash
   uvicorn main:app --reload
   ```

   The FastAPI backend will now be running at [http://localhost:8000](http://localhost:8000).

---

## Setting Up the Frontend

1. **Navigate to the Frontend Folder:**

   Open a new terminal window and change to the frontend directory:

   ```bash
   cd Mood_Based_Music_Recommendation_System/frontend
   ```

2. **Install Frontend Dependencies:**

   ```bash
   npm install
   ```

3. **Run the React Development Server:**

   ```bash
   npm start
   ```

   Your React app should automatically open in your default browser at [http://localhost:3000](http://localhost:3000).

---

## How It Works

1. **Capture Your Mood:**
   - On the landing page, the app requests access to your webcam.
   - Click the **"Capture My Mood"** button to take a snapshot from your video feed.

2. **Emotion Detection & Song Recommendation:**
   - The snapshot is sent to the FastAPI backend, where the CNN model processes the image to detect your facial emotion.
   - Based on the detected emotion, the backend filters a songs dataset and selects up to 5 recommendations.
   - The backend returns your detected mood, a Base64-encoded image of your face (the snapshot used for detection), and the recommended songs.

3. **Results Display:**
   - The frontend replaces the live video feed with the captured image.
   - It then displays your detected mood along with the recommended songs.

4. **Try Again:**
   - You can click the **"Try Again"** button to reload the page and capture a new mood.

---

## Deployment

For public deployment, consider using cloud platforms such as Heroku, Vercel, AWS, or DigitalOcean. Ensure that:
- CORS is configured appropriately.
- The React app is served over HTTPS.
- Environment variables are used for sensitive configuration (such as model paths).

---

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://reactjs.org/)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- https://medium.com/@UTMSBA24/mood-based-music-recommendation-system-5afb8bb90082
- Russell, J. A. (1980). A circumplex model of affect. Journal of Personality and Social Psychology, 39(6), 1161–1178.
- Russell, J. A. (2003). Core affect and the psychological construction of emotion. Psychological Review, 110(1), 145–172.
- Posner, J., Russell, J. A., & Peterson, B. S. (2005). The circumplex model of affect. Learning and Individual Differences, 39, 1161–1178.

