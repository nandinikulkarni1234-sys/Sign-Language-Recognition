import React, { useRef, useState, useCallback } from "react";
import Webcam from "react-webcam";
import axios from "axios";

function WebcamCapture() {
  const webcamRef = useRef(null);
  const [result, setResult] = useState("");
  const [isPredicting, setIsPredicting] = useState(false);

  const capture = useCallback(async () => {
    if (!webcamRef.current) return;

    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return;

    try {
      setIsPredicting(true);

      // Convert base64 image to Blob
      const blob = await fetch(imageSrc).then((res) => res.blob());
      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setResult(`Predicted: ${response.data.prediction}`);
    } catch (error) {
      console.error(error);
      setResult("Error connecting to backend!");
    } finally {
      setIsPredicting(false);
    }
  }, [webcamRef]);

  return (
    <div className="upload-container">
      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        width={400}
        videoConstraints={{ facingMode: "user" }}
      />
      <br />
      <button onClick={capture} disabled={isPredicting}>
        {isPredicting ? "Predicting..." : "Capture & Predict"}
      </button>
      <p>{result}</p>
    </div>
  );
}

export default WebcamCapture;
