import streamlit as st
import cv2
import numpy as np
from PIL import Image

# File uploader for image input
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read the uploaded file using PIL
    pil_image = Image.open(uploaded_file)  # PIL Image object
    st.image(pil_image, caption="Uploaded Image", use_column_width=True)  # Show original image

    # Convert PIL Image to a NumPy array (compatible with OpenCV)
    image = np.array(pil_image)

    # Convert to BGR if the image has 3 channels (Pillow outputs RGB, OpenCV needs BGR)
    if image.shape[-1] == 3:  # Check if it's a color image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Define coordinates for the vertebrae labels
    n = 270
    vertebrae_coords = {
        "L1": (n, 120),
        "L2": (n, 170),
        "L3": (n, 220),
        "L4": (n, 270),
        "L5": (n, 320),
    }

    # Annotate the image
    for label, coord in vertebrae_coords.items():
        cv2.putText(
            image, text=label, org=coord, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5, color=(255, 0, 0), thickness=2
        )

    # Convert the image back to RGB for displaying in Streamlit
    annotated_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the annotated image
    st.image(annotated_image, caption="Annotated MRI Image", use_column_width=True)
