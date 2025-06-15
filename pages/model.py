import streamlit as st
import os

from PIL import Image
from model_data import make_pred

# File uploader for image input
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])
brightness = st.slider("Adjust Brightness", 0.5, 5.0, 1.2, 0.1)

if uploaded_file:
    image_with_boxes = make_pred(uploaded_file, brightness=brightness)
    col1, col2 = st.columns([2, 1])  # Left wider than right

    with col1:
        st.image(image_with_boxes, caption="Detected Image", use_column_width=True)
    with col2:
        st.markdown("## Diagnosis")
        st.markdown("diagnosis Will appear here")