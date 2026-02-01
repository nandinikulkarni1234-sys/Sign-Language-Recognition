import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
from collections import deque, Counter

# ----------------------------------------------------
# Streamlit Page Setup
# ----------------------------------------------------
st.set_page_config(page_title="Sign Language Recognition", layout="wide")

st.markdown("""
# 🤘 INDIAN SIGN LANGUAGE RECOGNITION
""")

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------
st.sidebar.title("⚙️ Settings")
voice_enabled = st.sidebar.checkbox("🔊 Enable Voice Output", value=True)
show_crop = st.sidebar.checkbox("📸 Show Cropped Image", value=True)
show_history = st.sidebar.checkbox("📜 Show Prediction History", value=True)

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------
detector = HandDetector(detectionCon=0.35, minTrackCon=0.35, maxHands=2)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

with open("Model/labels.txt") as f:
    labels = [line.strip().split(" ")[-1] for line in f.readlines()]

# ----------------------------------------------------
# Javascript Voice Output
# ----------------------------------------------------
def speak_js(text):
    js_code = f"""
        <script>
            var msg = new SpeechSynthesisUtterance("{text}");
            msg.rate = 1;
            msg.pitch = 1;
            msg.volume = 1;
            speechSynthesis.cancel();
            speechSynthesis.speak(msg);
        </script>
    """
    st.components.v1.html(js_code, height=0)

# ----------------------------------------------------
# UI Sections
# ----------------------------------------------------
FRAME_COL, INFO_COL = st.columns([3, 2])

frame_holder = FRAME_COL.empty()
crop_holder = FRAME_COL.empty()

pred_box = INFO_COL.empty()
conf_box = INFO_COL.empty()
fps_box = INFO_COL.empty()
window_box = INFO_COL.empty()

history_box = st.empty()
prediction_history = []

# ----------------------------------------------------
# Prediction Smoothing + Stability Logic
# ----------------------------------------------------
WINDOW = 7              # frames in memory
REQUIRED_SAME = 4       # must appear at least this many times
CONF_THRESHOLD = 0.75   # 75% confidence minimum
COOLDOWN = 1.5          # seconds between voice speaking

pred_window = deque(maxlen=WINDOW)
prev_pred = None
last_speak_time = 0.0

# ----------------------------------------------------
# Buttons
# ----------------------------------------------------
start = st.button("🚀 Start Webcam")
stop = st.button("🛑 Stop Webcam")

if "running" not in st.session_state:
    st.session_state.running = False

if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

# ----------------------------------------------------
# Webcam Loop
# ----------------------------------------------------
if st.session_state.running:

    cap = cv2.VideoCapture(0)
    prev_time = time.time()

    while st.session_state.running:

        ret, img = cap.read()
        if not ret:
            st.error("❌ Unable to access webcam.")
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        final_label = "No Hand"
        confidence = 0

        # ================================
        # HAND DETECTION
        # ================================
        if hands:

            if len(hands) == 2:
                x1, y1, w1, h1 = hands[0]["bbox"]
                x2, y2, w2, h2 = hands[1]["bbox"]
                x = min(x1, x2)
                y = min(y1, y2)
                w = max(x1+w1, x2+w2) - x
                h = max(y1+h1, y2+h2) - y
            else:
                x, y, w, h = hands[0]["bbox"]

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            y1, y2 = max(0, y-offset), min(img.shape[0], y + h + offset)
            x1, x2 = max(0, x-offset), min(img.shape[1], x + w + offset)

            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size != 0:
                aspect = h / w

                if aspect > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    gap = (imgSize - wCal) // 2
                    imgWhite[:, gap:gap+wCal] = imgResize

                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    gap = (imgSize - hCal) // 2
                    imgWhite[gap:gap+hCal, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite)
                final_label = labels[index]
                confidence = float(max(prediction))

                if show_crop:
                    crop_holder.image(cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB))

            cv2.rectangle(imgOutput, (x-offset, y-offset),
                          (x+w+offset, y+h+offset), (255, 0, 255), 3)
            cv2.putText(imgOutput, final_label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # ================================
        # FPS
        # ================================
        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now

        # ================================
        # APPLY STABILITY LOGIC
        # ================================
        pred_window.append((final_label if final_label != "No Hand" else None, confidence))

        valid_labels = [lbl for lbl, conf in pred_window if lbl and conf >= CONF_THRESHOLD]

        stable_label = None
        if valid_labels:
            stable_label, count = Counter(valid_labels).most_common(1)[0]
            if count < REQUIRED_SAME:
                stable_label = None

        window_box.caption(f"Window: {pred_window}")

        # ================================
        # FINAL OUTPUT
        # ================================
        frame_holder.image(cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB))
        pred_box.markdown(f"### 🔮 Prediction: **{final_label}**")
        conf_box.markdown(f"### 🎯 Confidence: **{confidence*100:.1f}%**")
        fps_box.markdown(f"### ⚡ FPS: **{int(fps)}**")

        if stable_label:
            if show_history:
                prediction_history.append(stable_label)
                history_box.write(prediction_history[-10:])

            # Voice Output
            if (
                voice_enabled
                and stable_label != prev_pred
                and (time.time() - last_speak_time) > COOLDOWN
            ):
                speak_js(stable_label)
                prev_pred = stable_label
                last_speak_time = time.time()

        time.sleep(0.001)

    cap.release()
