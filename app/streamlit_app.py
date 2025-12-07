import os
import sys
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# ---------- PATH SETUP ----------
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.append(SRC_DIR)

from data_prep import IMG_HEIGHT, IMG_WIDTH, get_data_generators  # type: ignore

MODEL_PATH = os.path.join(ROOT_DIR, "artifacts", "best_model.h5")
DATA_DIR = os.path.join(ROOT_DIR, "data")


# ---------- CACHED HELPERS ----------
@st.cache_resource
def load_cnn_model():
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Could not load model at {MODEL_PATH}: {e}")
        return None


@st.cache_resource
def get_class_names():
    train_dir = os.path.join(DATA_DIR, "train")
    if os.path.isdir(train_dir):
        # sort to keep consistent order with training generator
        classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        return classes
    # fallback: default classes (adjust if your actual classes differ)
    return ["badminton", "cricket", "karate", "soccer", "swimming", "tennis", "wrestling"]


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# ---------- UI LAYOUT ----------
def main():
    st.set_page_config(
        page_title="Sports Image Classifier",
        layout="wide",
    )

    st.sidebar.title("How to use:")
    st.sidebar.markdown("1. Upload an image of a sport\n2. Click Predict Sport\n3. View predicted class & confidence")

    st.title("Sports Image Classifier")
    model = load_cnn_model()
    class_names = get_class_names()

    uploaded = st.file_uploader("Upload a sport image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload an image to get started.")
        return

    image = Image.open(uploaded)
    # use_container_width instead of deprecated use_column_width
    st.image(image, caption="Input image", use_container_width=True)

    if st.button("Predict Sport"):
        if model is None:
            st.error("Model not available. Check artifacts/best_model.h5")
            return

        arr = preprocess_image(image)
        preds = model.predict(arr)[0]
        top_idx = int(np.argmax(preds))
        top_prob = float(preds[top_idx]) * 100.0
        top_label = class_names[top_idx] if top_idx < len(class_names) else f"class_{top_idx}"

        st.markdown("---")
        st.subheader("Prediction")
        st.metric(label="Predicted sport", value=top_label.capitalize())
        st.write(f"Confidence: {top_prob:.2f}%")

        # optional: show raw probabilities collapsed (comment out if you don't want)
        # with st.expander("Show class probabilities"):
        #     prob_map = {class_names[i]: f"{preds[i]*100:.2f}%" for i in range(len(preds))}
        #     st.json(prob_map)


if __name__ == "__main__":
    main()
