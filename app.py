import streamlit as st
import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="DigitVision | AI Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F7F8FC;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

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

/* Metric Cards */
.card {
    background: white;
    border: 1px solid #EAECF3;
    border-radius: 16px;
    padding: 23px;
    min-height: 125px;
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

/* Sections */
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
    /* min-height removed so boxes fit tightly */
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


# ==========================================
# 3. LOAD CNN MODEL
# ==========================================

MODEL_PATH = Path.cwd() / "mnist_cnn_model.keras"
if not MODEL_PATH.exists():
    try:
        MODEL_PATH = Path(__file__).parent / "mnist_cnn_model.keras"
    except NameError:
        pass

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
    model_status = "Active"
except Exception as e:
    model = None
    model_status = "Error"
    st.error(f"Model loading failed. Please ensure 'mnist_cnn_model.keras' is in the same folder. Error: {e}")


# ==========================================
# 4. IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image_array, uploaded=False):
    # Convert image to numpy array
    image_array = np.asarray(image_array)

    # Handle RGBA or RGB images
    if image_array.ndim == 3:
        image_array = image_array[:, :, :3]
        image_array = np.mean(image_array, axis=2)

    # Convert to uint8
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    # Check background color using border pixels safely
    border = np.concatenate([
        image_array[0, :],
        image_array[-1, :],
        image_array[:, 0],
        image_array[:, -1]
    ])

    # MNIST requires white digit on black background
    if np.mean(border) > 127:
        image_array = 255 - image_array

    # Remove small background noise
    image_array = np.where(image_array > 30, image_array, 0).astype(np.uint8)

    # Convert to PIL image
    digit_image = Image.fromarray(image_array)

    # Find digit boundaries
    bbox = digit_image.getbbox()
    if bbox is None:
        return None

    # Crop unnecessary background
    digit_image = digit_image.crop(bbox)

    # Resize safely for ALL versions of Pillow (backwards compatible)
    resampling_filter = getattr(Image, "Resampling", Image).LANCZOS
    digit_image.thumbnail((20, 20), resampling_filter)

    # Create 28x28 black image
    final_image = Image.new("L", (28, 28), 0)

    # Center digit
    x = (28 - digit_image.width) // 2
    y = (28 - digit_image.height) // 2
    final_image.paste(digit_image, (x, y))

    # Convert to numpy
    image = np.array(final_image, dtype=np.float32)

    # Normalize
    image = image / 255.0

    # Reshape for CNN
    image = image.reshape(1, 28, 28, 1)

    return image


# ==========================================
# 5. PREDICTION FUNCTION
# ==========================================

def predict_digit(image):
    prediction = model.predict(image, verbose=0)
    digit = int(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0])) * 100
    return digit, confidence, prediction[0]


# ==========================================
# 6. NAVBAR
# ==========================================

st.markdown("""
<div class="navbar">
    <div class="brand">✦ DigitVision</div>
    <div class="nav-label">AI POWERED • CNN MODEL</div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 7. HERO SECTION
# ==========================================

st.markdown("""
<div class="hero">
    <span class="tag">✦ DEEP LEARNING APPLICATION</span>
    <h1>Handwritten Digit Recognition</h1>
    <p>
        Experience the power of Artificial Intelligence.
        Draw a handwritten number or upload an image
        and let our CNN model recognize your digit instantly.
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 8. DASHBOARD METRICS
# ==========================================

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


# ==========================================
# 9. PREDICTION SECTION
# ==========================================

st.markdown('<div class="section-title">Try the AI Predictor</div>', unsafe_allow_html=True)

st.markdown("""
<div class="section-subtitle">
    Draw a digit between 0 and 9 or upload a handwritten image.
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.15, 0.85], gap="large")


# ==========================================
# 10. LEFT PANEL - DRAWING CANVAS
# ==========================================

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### ✏️ Draw your digit")

    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="digit_canvas",
        return_image_data=True
    )

    st.caption("Use your mouse to draw a digit clearly.")
    st.markdown("### Or upload an image")

    uploaded_file = st.file_uploader(
        "Upload handwritten digit",
        type=["png", "jpg", "jpeg"],
        key="digit_upload"
    )

    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# 11. RIGHT PANEL - PREDICTION RESULT
# ==========================================

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 🎯 Prediction Result")
    st.write("Your predicted digit will appear here.")

    if model is None:
        st.error("CNN model is unavailable. Check model file.")
    else:
        if st.button("Recognize Digit", key="predict_button"):
            try:
                processed_image = None

                # Uploaded image has priority
                if uploaded_file is not None:
                    original_image = Image.open(uploaded_file).convert("RGB")
                    processed_image = preprocess_image(np.array(original_image), uploaded=True)

                # Otherwise use drawing canvas
                elif canvas_result is not None:
                    canvas_image = canvas_result.image_data
                    if canvas_image is not None and len(canvas_image) > 0:
                        processed_image = preprocess_image(canvas_image, uploaded=False)

                # Check empty image
                if processed_image is None:
                    st.warning("Please draw or upload a digit first.")
                else:
                    # Display processed image
                    st.markdown("#### Processed Image")
                    st.image(processed_image.reshape(28, 28), width=140, clamp=True)

                    # Get prediction
                    digit, confidence, probabilities = predict_digit(processed_image)

                    # Display prediction
                    st.success("Prediction completed!")
                    
                    st.markdown(
                        f"""
                        <div style="
                            background:#F0EBFF;
                            border-radius:16px;
                            padding:25px;
                            text-align:center;
                            margin:15px 0;
                        ">
                            <div style="color:#6537D6; font-size:14px; font-weight:600;">
                                PREDICTED DIGIT
                            </div>
                            <div style="font-size:75px; font-weight:800; color:#5429D8;">
                                {digit}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Confidence / Highest Probability only
                    st.markdown("#### Class Probability")
                    st.write(f"Digit **{digit}**: {confidence:.2f}%")
                    
                    # Safely constrain the float between 0.0 and 1.0 for the single progress bar
                    safe_prob = min(max(confidence / 100.0, 0.0), 1.0)
                    st.progress(safe_prob)

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# 12. FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    DigitVision © 2026
    <br><br>
    Built with Python • TensorFlow • CNN • Streamlit
    <br>
    Powered by Deep Learning
</div>
""", unsafe_allow_html=True)
