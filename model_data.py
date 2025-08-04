import base64
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input

# ---------------------------
# Utility
# ---------------------------
def brighten_image(pil_img, factor):
    enhancer = ImageEnhance.Brightness(pil_img)
    return enhancer.enhance(factor)

# ---------------------------
# Load YOLO model safely
# ---------------------------
def load_yolo_model():
    try:
        return YOLO("models/best.pt")
    except Exception as e:
        st.error(f"YOLO model load failed: {e}")
        return None

# ---------------------------
# Load fault classification models safely
# ---------------------------
def load_fault_models():
    fault_labels = ["A", "B", "C", "D", "E"]
    models = {}
    for label in fault_labels:
        model_path = f"models/model_{label}.h5"
        try:
            models[label] = load_model(model_path)
        except Exception as e:
            st.warning(f"Could not load model for Fault {label}: {e}")
            models[label] = None
    return models

# Initialize models once
yolo_model = load_yolo_model()
fault_models = load_fault_models()

# ---------------------------
# YOLO detection
# ---------------------------
def make_pred(image_path, conf_threshold=0.25, brightness=1.5):
    if yolo_model is None:
        st.error("YOLO model not loaded.")
        return None, None

    pil_image = Image.open(image_path).convert("RGB")
    bright_image = brighten_image(pil_image, brightness)

    np_image = np.array(bright_image)
    results = yolo_model.predict(source=np_image, conf=conf_threshold, save=False)

    colors = {0: (255, 0, 0), 1: (0, 0, 255)}

    for box in results[0].boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{yolo_model.names[cls]} {conf:.2f}"
        color = colors.get(cls, (0, 255, 0))
        cv2.rectangle(np_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(np_image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return np_image, bright_image

# ---------------------------
# Fault prediction
# ---------------------------
def predict_faults(pil_img):
    results = {}
    resized = pil_img.resize((299, 299))
    arr = preprocess_input(np.expand_dims(img_to_array(resized), axis=0))

    for fault, model in fault_models.items():
        if model is None:
            results[fault] = ("Model not loaded", None)
            continue
        try:
            pred = model.predict(arr, verbose=0)[0][0]
            label = "Present" if pred > 0.5 else "Not Present"
            results[fault] = (label, float(pred))
        except Exception as e:
            results[fault] = (f"Error: {e}", None)

    return results
