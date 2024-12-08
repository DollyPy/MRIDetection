import streamlit as st
from model_data import add_bg_from_local

add_bg_from_local("bg-02.jpg")
st.title("SpineReaderRx")
#st.subheader("SpineReaderRx")
st.write(
    """
    <b> Dr. Akinmade </b>

    """, unsafe_allow_html=True
)
