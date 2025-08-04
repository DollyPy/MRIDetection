import streamlit as st
import os

from PIL import Image
from model_data import make_pred, predict_faults

# File uploader for image input
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])
brightness = st.slider("Adjust Brightness", 0.5, 5.0, 1.2, 0.1)

diagnosis_names = {
    "A": "Decreased Spinal Canal Diameter",
    "B": "Abnormal Bone Density",
    "C": "Decreased Intervertebral Disc Height",
    "D": "Thecal Sac Indentation",
    "E": "Spondylolisthesis"
}

## 
if uploaded_file:
    image_with_boxes, bright_image = make_pred(uploaded_file, brightness=brightness)
    col1, col2 = st.columns([2, 1])  # Left wider than right
    fault_preds = predict_faults(bright_image)

    with col1:
        st.image(image_with_boxes, caption="Detected Image", use_container_width=True)
    with col2:
        st.markdown("## Predicted Faults")
        for label, (status, score) in fault_preds.items():
            if score is not None:
                if status == "Present":
                    st.error(f"{diagnosis_names[label]} is **PRESENT** \nConfidence level ({score:.2f})")
                else:
                    st.success(f"{diagnosis_names[label]} is Not **Present** \nConfidence level ({1-score:.2f})")
            else:
                st.warning(f"{label} - {status}")
