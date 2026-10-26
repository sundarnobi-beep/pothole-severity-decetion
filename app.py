
import streamlit as st
import cv2
import numpy as np
import pickle
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="Pothole Severity Detection",
    page_icon="🚧",
    layout="wide"
)

st.title("🚧 Pothole Severity Detection System")
st.write("Upload a road image to analyse pothole severity using DIP, ML and Neural Networks.")

project = Path(__file__).resolve().parent.parent
models_path = project / "Models"

@st.cache_resource
def load_models():
    with open(models_path / ""/mount/src/Models/pothole_ml_model.pkl"", "rb") as f:
        ml_model = pickle.load(f)
    with open(models_path / ""/mount/src/Models/pothole_ml_model.pkl"", "rb") as f:
        nn_model = pickle.load(f)
    return ml_model, nn_model

try:
    ml_model, nn_model = load_models()
except Exception as e:
    st.error(f"Could not load models: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    processed = cv2.resize(gray, (64, 64)).flatten().astype(np.float64) / 255.0
    features = processed.reshape(1, -1)

    ml_prediction = ml_model.predict(features)[0]
    nn_prediction = nn_model.predict(features)[0]

    st.subheader("Digital Image Processing (DIP)")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("Original Image")
        st.image(image_array, use_container_width=True)

    with col2:
        st.caption("Grayscale Image")
        st.image(gray, use_container_width=True)

    with col3:
        st.caption("Edge Detection")
        st.image(edges, use_container_width=True)

    st.subheader("Model Predictions")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Machine Learning (SVM)", str(ml_prediction).upper())

    with col2:
        st.metric("Neural Network (MLP)", str(nn_prediction).upper())

    if ml_prediction == nn_prediction:
        st.success(f"Both models predict: {str(ml_prediction).upper()}")
    else:
        st.warning("The models disagree. Their predictions may be unreliable.")

    st.caption(
        "Prototype only: trained on a very small dataset, so predictions "
        "are not reliable for real-world road-safety decisions."
    )
