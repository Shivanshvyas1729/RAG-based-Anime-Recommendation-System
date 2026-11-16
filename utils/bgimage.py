import base64
from pathlib import Path
import streamlit as st

def get_base64_image(image_file: str) -> str:
    if image_file and Path(image_file).exists():
        with open(image_file, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

def set_background(image_file: str = None, mode: str = "dark", *args, **kwargs):
    """
    Sets a custom background for the Streamlit app.
    Supports 'dark' and 'light' modes with tuned overlay opacity for high text readability.
    """
    b64_encoded = get_base64_image(image_file) if image_file else ""
    
    if mode == "dark":
        if b64_encoded:
            bg_style = f"""
                background-image: linear-gradient(rgba(11, 15, 25, 0.90), rgba(11, 15, 25, 0.95)), url(data:image/png;base64,{b64_encoded});
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            """
        else:
            bg_style = "background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0f172a 100%); background-attachment: fixed;"
    else:  # light mode
        if b64_encoded:
            bg_style = f"""
                background-image: linear-gradient(rgba(248, 250, 252, 0.93), rgba(241, 245, 249, 0.97)), url(data:image/png;base64,{b64_encoded});
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            """
        else:
            bg_style = "background: linear-gradient(135deg, #F8FAFC 0%, #EEF2F6 50%, #F1F5F9 100%); background-attachment: fixed;"

    style = f"""
        <style>
        .stApp {{
            {bg_style}
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)
