"""
Bước 4: Nhận diện biểu cảm realtime qua webcam
- Phát hiện mặt: OpenCV Haar Cascade
- Phân loại: Model tự train (MobileNetV2)

Chạy: python main.py  hoặc  run_main.bat
"""

import bootstrap

bootstrap.ensure_venv()

import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

from config import (
    CLASS_NAMES,
    EMOTION_VI,
    FONT_PATH,
    IMG_SIZE,
    MODEL_PATH,
    SMOOTH_FRAMES,
    SUBTLE_EMOTIONS,
)


class EmotionSmoother:
    def __init__(self, window=SMOOTH_FRAMES):
        self._history = deque(maxlen=window)

    def update(self, scores):
        self._history.append(scores)
        avg = np.zeros(len(scores))
        for sample in self._history:
            avg += sample
        return avg / len(self._history)


def pick_emotion(scores):
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_idx, top_score = ranked[0]
    top_name = CLASS_NAMES[top_idx]

    if top_name == "neutral":
        for idx, score in ranked[1:]:
            name = CLASS_NAMES[idx]
            if name in SUBTLE_EMOTIONS and score >= 0.18:
                if top_score - score <= 0.12:
                    return name, score

    return top_name, top_score


def put_text_vi(img, text, pos, font_size=24, color=(0, 255, 0)):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(FONT_PATH, font_size)
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def format_top3(scores):
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:3]
    parts = []
    for idx, score in ranked:
        label = EMOTION_VI[CLASS_NAMES[idx]]
        parts.append(f"{label} {score:.0%}")
    return " | ".join(parts)


def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    return faces


def crop_face(frame, box, padding=20):
    x, y, w, h = box
    height, width = frame.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)
    return frame[y1:y2, x1:x2]


def preprocess(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, IMG_SIZE)
    batch = np.expand_dims(face_rgb.astype("float32"), axis=0)
    return batch


def main():
    if not MODEL_PATH.exists():
        print(f"Chưa có model tại {MODEL_PATH}")
        print("Chạy trước: python train.py")
        sys.exit(1)

    print("Đang load model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    smoother = EmotionSmoother()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Nhấn 'q' để thoát")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        faces = detect_faces(frame, face_cascade)

        for box in faces:
            x, y, w, h = box
            face_crop = crop_face(frame, box)
            if face_crop.size == 0:
                continue

            batch = preprocess(face_crop)
            scores = model.predict(batch, verbose=0)[0]
            smoothed = smoother.update(scores)
            emotion, confidence = pick_emotion(smoothed)

            label_vi = EMOTION_VI[emotion]
            main_label = f"{label_vi}: {confidence:.0%}"
            sub_label = format_top3(smoothed)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame = put_text_vi(frame, main_label, (x, max(y - 55, 5)), font_size=26)
            frame = put_text_vi(
                frame, sub_label, (x, max(y - 25, 30)), font_size=18, color=(0, 220, 255)
            )

        cv2.imshow("Nhận diện biểu cảm khuôn mặt", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
