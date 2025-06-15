import base64
from ultralytics import YOLO
import streamlit as st
import numpy as np
# import os
import cv2
from PIL import Image, ImageEnhance

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover;
    }}
    .img {{
        opacity:1
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

def brighten_image(pil_img, factor):
    enhancer = ImageEnhance.Brightness(pil_img)
    return enhancer.enhance(factor)

# Prediction function
def make_pred(image_path, conf_threshold=0.25, brightness=1.5):
    pil_image = Image.open(image_path).convert("RGB")
    
    bright_image = brighten_image(pil_image, brightness )  # Apply brightness factor

    np_image = np.array(bright_image)

    results = model.predict(source=np_image, conf=conf_threshold, save=False)

    colors = {
        0:(255,0,0),
        1:(0,0,255)
    }

    for box in results[0].boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = f"{model.names[cls]} {conf:.2f}"
        color = colors[cls]

        cv2.rectangle(np_image, (x1, y1), (x2, y2), color, 6)
        cv2.putText(np_image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, )

    return np_image
try:
    model = YOLO("runs/detect/train6/weights/best.pt")
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

