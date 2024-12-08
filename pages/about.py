import streamlit as st
from model_data import add_bg_from_local

add_bg_from_local("bg-01.jpg")
st.header("ABOUT")
st.write("""
This is a Machine Learning Spine Reader 

""")
