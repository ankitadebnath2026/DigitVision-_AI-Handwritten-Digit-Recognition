
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from streamlit_drawable_canvas import st_canvas

# Load trained CNN model
MODEL_PATH = Path(__file__).parent / "mnist_cnn_model.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Application title
st.title("Handwritten Digit Recognition")
st.write("Draw a digit between 0 and 9.")

st.success("CNN Model loaded successfully!")
