import base64
import streamlit as st
import numpy as np
import os
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array

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

def make_pred(image_path):
    pass

try:
    model = models.load_model("./models/leaf.h5")
except:
    pass