// frontend/src/App.js
import React, { useRef, useState, useEffect } from 'react';
import './App.css';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [mood, setMood] = useState(null);
  const [songs, setSongs] = useState([]);
  const [faceImage, setFaceImage] = useState(null);
  const [error, setError] = useState(null);
  const [capturing, setCapturing] = useState(false);

  // Request access to the webcam when the component mounts.
  useEffect(() => {
    async function getCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing camera", err);
        setError("Unable to access camera. Please check your device permissions.");
      }
    }
    getCamera();
  }, []);

  const captureAndDetect = async () => {
    setCapturing(true);
    setError(null);
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    // Set the canvas size to match the video dimensions.
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert the canvas image to a blob.
    canvas.toBlob(async (blob) => {
      if (!blob) {
        setError("Failed to capture image.");
        setCapturing(false);
        return;
      }
      // Prepare FormData with the captured image.
      const formData = new FormData();
      formData.append('file', blob, 'snapshot.jpg');

      try {
        const response = await fetch('http://localhost:8000/detect', {
          method: 'POST',
          body: formData,
        });
        const data = await response.json();
        if (data.error) {
          setError(data.error);
        } else {
          setMood(data.mood);
          setSongs(data.songs);
          setFaceImage(data.face_image);
        }
      } catch (err) {
        console.error(err);
        setError("Error detecting emotion.");
      }
      setCapturing(false);
    }, 'image/jpeg');
  };

  return (
    <div className="App">
      <div className="container">
        {!mood ? (
          // Landing / Capture view with live video feed
          <div className="landing">
            <h1>Mood Melody</h1>
            <p>
              Let the music match your soul. Capture your mood using your camera and receive personalized song recommendations.
            </p>
            <div className="video-container">
              <video ref={videoRef} autoPlay playsInline className="video-feed" />
              <canvas ref={canvasRef} style={{ display: 'none' }} />
            </div>
            {error && <p className="error">{error}</p>}
            <button onClick={captureAndDetect} disabled={capturing}>
              {capturing ? 'Detecting...' : 'Capture My Mood'}
            </button>
          </div>
        ) : (
          // Results view: show detected mood, the captured face image, and recommendations.
          <div className="results">
            <h1>Your Mood: {mood}</h1>
            {faceImage && (
              <div className="result-image">
                <img src={faceImage} alt="Detected face" />
              </div>
            )}
            {songs.length > 0 ? (
              <>
                <h2>Recommended Songs:</h2>
                <ul>
                  {songs.map((song, index) => (
                    <li key={index}>{song}</li>
                  ))}
                </ul>
              </>
            ) : (
              <p>No songs found for your mood.</p>
            )}
            <button onClick={() => window.location.reload()}>Try Again</button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
