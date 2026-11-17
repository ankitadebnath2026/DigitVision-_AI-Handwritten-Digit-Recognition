
import streamlit as st
import tensorflow as tf
import numpy as np
from pathlib import Path
from streamlit_drawable_canvas import st_canvas

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="DigitVision | AI Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F7F8FC;
}

header {visibility: hidden;}
footer {visibility: hidden;}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Navbar */
.navbar {
    background: white;
    padding: 18px 25px;
    border-radius: 14px;
    border: 1px solid #EAECF3;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 35px;
}

.brand {
    font-size: 23px;
    font-weight: 800;
    color: #5429D8;
}

.nav-label {
    color: #667085;
    font-size: 13px;
    font-weight: 600;
}

/* Hero */
.hero {
    background: white;
    border: 1px solid #EAECF3;
    border-radius: 20px;
    padding: 38px;
    margin-bottom: 25px;
}

.tag {
    display: inline-block;
    background: #F0EBFF;
    color: #6537D6;
    padding: 7px 13px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
}

.hero h1 {
    font-size: 39px;
    font-weight: 800;
    color: #17152B;
    margin: 18px 0 10px 0;
}

.hero p {
    font-size: 15px;
    color: #77788A;
    line-height: 1.7;
}

/* Cards */
.card {
    background: white;
    border: 1px solid #EAECF3;
    border-radius: 16px;
    padding: 23px;
    height: 125px;
}

.card-title {
    font-size: 13px;
    color: #85869A;
    font-weight: 600;
}

.card-value {
    font-size: 25px;
    color: #25213B;
    font-weight: 800;
    margin-top: 12px;
}

/* Section */
.section-title {
    color: #211D35;
    font-size: 21px;
    font-weight: 750;
    margin: 30px 0 8px 0;
}

.section-subtitle {
    color: #85869A;
    font-size: 13px;
    margin-bottom: 20px;
}

/* Panels */
.panel {
    background: white;
    border: 1px solid #EAECF3;
    border-radius: 18px;
    padding: 23px;
}

/* Buttons */
.stButton > button {
    background: #6537D6;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 11px 20px;
    font-weight: 700;
    width: 100%;
}

.stButton > button:hover {
    background: #4F25BB;
    color: white;
    border: none;
}

/* Footer */
.footer {
    text-align: center;
    color: #9696A5;
    font-size: 12px;
    padding: 30px 0 10px 0;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
MODEL_PATH = Path(__file__).parent / "mnist_cnn_model.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
    model_status = "Active"
except Exception as e:
    model = None
    model_status = "Error"


# ---------------- NAVBAR ----------------
st.markdown("""
<div class="navbar">
    <div class="brand">✦ DigitVision</div>
    <div class="nav-label">AI POWERED • CNN MODEL</div>
</div>
""", unsafe_allow_html=True)


# ---------------- HERO SECTION ----------------
st.markdown("""
<div class="hero">
    <span class="tag">✦ DEEP LEARNING APPLICATION</span>
    <h1>Handwritten Digit Recognition</h1>
    <p>
        Experience the power of Artificial Intelligence.
        Draw a handwritten number and let our CNN model
        recognize your digit instantly.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- METRICS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-title">DATASET</div>
        <div class="card-value">MNIST</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-title">CLASSES</div>
        <div class="card-value">10 Digits</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card">
        <div class="card-title">MODEL STATUS</div>
        <div class="card-value">{model_status}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- DRAWING SECTION ----------------
st.markdown('<div class="section-title">Try the AI Predictor</div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="section-subtitle">
    Draw any digit from 0 to 9 using your mouse or upload an image.
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown("### ✏️ Draw your digit")

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="digit_canvas"
    )

    st.caption("Use your mouse to draw a digit clearly.")

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown("### 🎯 Prediction Result")

    st.write("Your predicted digit will appear here.")

    if model is None:
        st.error("Model could not be loaded. Check the model file.")
    elif canvas_result.image_data is not None:
        if st.button("Recognize Digit"):
            image = canvas_result.image_data

            # Extract grayscale image
            image = image[:, :, :3]
            image = np.mean(image, axis=2)

            # Convert to 0-255
            image = np.uint8(image)

            # Resize to MNIST dimensions
            image = tf.image.resize(
                image[..., np.newaxis],
                (28, 28)
            ).numpy()

            # Normalize
            image = image / 255.0

            # Model prediction
            prediction = model.predict(
                image.reshape(1, 28, 28, 1),
                verbose=0
            )

            digit = int(np.argmax(prediction[0]))
            confidence = float(np.max(prediction[0])) * 100

            st.success(f"Predicted Digit: {digit}")
            st.metric("Confidence", f"{confidence:.2f}%")

            st.markdown("#### Class probabilities")

            for i, probability in enumerate(prediction[0]):
                st.write(f"Digit {i}: {probability*100:.2f}%")

                st.progress(float(probability))

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    DigitVision © 2026 | Built with Python, TensorFlow & Streamlit
    <br><br>
    Powered by Convolutional Neural Networks
</div>
""", unsafe_allow_html=True)
