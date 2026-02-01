import React, { useRef, useState, useEffect } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import { Hands } from "@mediapipe/hands";
import * as cam from "@mediapipe/camera_utils";

export default function WebcamPredict() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [prediction, setPrediction] = useState("...");
  const [voiceEnabled, setVoiceEnabled] = useState(true);

  // ---------- SPEAK ----------
  const speak = (text) => {
    if (!voiceEnabled) return;
    let msg = new SpeechSynthesisUtterance(text);
    msg.lang = "en-US";

    const voices = window.speechSynthesis.getVoices();
    const female = voices.find(v =>
      v.name.toLowerCase().includes("female") ||
      v.name.toLowerCase().includes("woman") ||
      v.name.toLowerCase().includes("zira")
    );
    if (female) msg.voice = female;

    window.speechSynthesis.speak(msg);
  };

  // --------- MEDIAPIPE HANDS: DRAW BOX ----------
  useEffect(() => {
    const hands = new Hands({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 0,
      minDetectionConfidence: 0.4,
      minTrackingConfidence: 0.4,
    });

    hands.onResults((results) => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (!results.multiHandLandmarks?.length) return;

      const pts = results.multiHandLandmarks[0];
      const xs = pts.map((p) => p.x * canvas.width);
      const ys = pts.map((p) => p.y * canvas.height);

      const left = Math.min(...xs);
      const right = Math.max(...xs);
      const top = Math.min(...ys);
      const bottom = Math.max(...ys);

      ctx.strokeStyle = "lime";
      ctx.lineWidth = 4;
      ctx.strokeRect(left - 20, top - 20, right - left + 40, bottom - top + 40);
    });

    if (webcamRef.current?.video) {
      const camera = new cam.Camera(webcamRef.current.video, {
        onFrame: async () => {
          await hands.send({ image: webcamRef.current.video });
        },
        width: 640,
        height: 480,
      });
      camera.start();
    }
  }, []);

  // ---------- AUTO SEND FRAME EVERY 1 SEC ----------
  useEffect(() => {
    const interval = setInterval(() => {
      sendFrame();
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // ---------- SEND FRAME TO BACKEND ----------
  const sendFrame = async () => {
    if (!webcamRef.current) return;
    const imgSrc = webcamRef.current.getScreenshot();
    if (!imgSrc) return;

    const base64 = imgSrc.split(",")[1];

    try {
      const res = await axios.post(
        "http://127.0.0.1:5000/predict",
        { image: base64 },
        { headers: { "Content-Type": "application/json" } }
      );

      const label = res.data.label;

      if (label && label !== prediction && label !== "No Hand") {
        speak(label);
      }

      setPrediction(label);

    } catch (e) {
      setPrediction("Connection Error");
      console.log("Prediction error", e);
    }
  };

  return (
    <div style={{ textAlign: "center", paddingTop: "20px" }}>
      <h1>Sign Language Recognition (Real-Time)</h1>

      {/* Webcam + Canvas */}
      <div
        style={{
          position: "relative",
          width: 640,
          height: 480,
          margin: "20px auto",
        }}
      >
        <Webcam
          ref={webcamRef}
          width={640}
          height={480}
          screenshotFormat="image/jpeg"
          style={{ position: "absolute", top: 0, left: 0, borderRadius: "12px" }}
        />

        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            pointerEvents: "none",
          }}
        ></canvas>
      </div>

      <h2 style={{ fontSize: "30px" }}>
        Prediction: <span style={{ color: "green" }}>{prediction}</span>
      </h2>

      <button
        onClick={() => setVoiceEnabled(!voiceEnabled)}
        style={{
          marginTop: "20px",
          padding: "10px 25px",
          fontSize: "18px",
          background: "#8000ff",
          color: "white",
          border: "none",
          borderRadius: "8px",
          cursor: "pointer",
        }}
      >
        {voiceEnabled ? "Disable Voice" : "Enable Voice"}
      </button>
    </div>
  );
}
