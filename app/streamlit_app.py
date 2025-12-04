import os
import sys
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model

# ---------- PATH SETUP ----------
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.append(SRC_DIR)

from data_prep import IMG_HEIGHT, IMG_WIDTH, get_data_generators  # type: ignore

MODEL_PATH = os.path.join(ROOT_DIR, "src","artifacts", "best_model.h5")
DATA_DIR = os.path.join(ROOT_DIR, "data")


# ---------- CACHED HELPERS ----------
@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)


@st.cache_resource
def get_class_names():
    # Use training generator only to get class indices
    train_gen, _ = get_data_generators(data_dir=DATA_DIR)
    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    return class_names


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
       # page_icon="",
        layout="wide",
    )

    # Custom CSS
    st.markdown(
        """
        <style>
        /* Global background */
        .stApp {
            background: radial-gradient(circle at top left, #1f2933, #020617);
            color: #f9fafb;
            font-family: "Segoe UI", system-ui, sans-serif;
        }
        /* Center card */
        .main-card {
            background: #020617;
            border-radius: 20px;
            padding: 2.5rem 2rem;
            box-shadow: 0 24px 60px rgba(0,0,0,0.65);
            border: 1px solid rgba(148,163,184,0.25);
        }
        .title-text {
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: 0.03em;
        }
        .subtitle-text {
            font-size: 0.98rem;
            color: #cbd5f5;
        }
        .prediction-label {
            font-size: 1.4rem;
            font-weight: 700;
            margin-top: 0.5rem;
        }
        .sport-pill {
            display: inline-block;
            padding: 0.3rem 0.75rem;
            border-radius: 999px;
            background: rgba(59,130,246,0.12);
            border: 1px solid rgba(59,130,246,0.45);
            font-size: 0.85rem;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------- SIDEBAR ----------
    with st.sidebar:
        
        st.markdown("---")
        st.markdown("**How to use:**")
        st.markdown(
            """
            1. Upload an image of a sport  
            2. Click **Predict Sport**  
            3. View predicted class & confidence  
            """
        )
        st.markdown("---")
        

    # ---------- MAIN LAYOUT ----------
    col_left, col_right = st.columns([1.15, 1])

    with col_left:
    
        st.markdown(
             '<div style="text-align:center;">'
            '<div class="title-text">Sports Image Classification using CNN </div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="subtitle-text" style="text-align:center;>'
            "Upload an image containing one of the sports from the training dataset "
            "(Badminton, Cricket, Karate, Soccer, Swimming, Tennis, Wrestling). "
            "The model will predict the sport and show the confidence score."
            "</p>",
            unsafe_allow_html=True,
        )

       # st.markdown("### <p style='text-align:center;'>Upload Image</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a sports image (JPG / PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
        )

        placeholder_image = st.empty()
        result_placeholder = st.empty()

        model = load_cnn_model()
        class_names = get_class_names()

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            placeholder_image.image(
                image, caption="Uploaded Image", use_column_width=True
            )

            if st.button("Predict Sport"):
                with st.spinner("Analyzing image..."):
                    x = preprocess_image(image)
                    preds = model.predict(x)
                    pred_idx = int(np.argmax(preds[0]))
                    confidence = float(np.max(preds[0]))
                    predicted_class = class_names[pred_idx]

                result_placeholder.markdown("---")
                result_placeholder.markdown("<h3 style='text-align:center;'>Prediction Result</h3>", unsafe_allow_html=True)
                result_placeholder.markdown(
                    f'<div class="prediction-label" style="text-align:center;>{predicted_class}</div>',
                    unsafe_allow_html=True,
                )
                st.progress(confidence)
                st.markdown(f"<p style='text-align:center;'>Confidence: <b>{confidence*100:.2f}%</b></p>", unsafe_allow_html=True)

                st.markdown("<h5 style='text-align:center;'>Class probabilities:</h5>", unsafe_allow_html=True)
                prob_cols = st.columns(len(class_names))
                for i, cls_name in enumerate(class_names):
                    with prob_cols[i]:
                        st.write(cls_name)
                        st.write(f"{preds[0][i]*100:.1f}%")
        else:
            st.info(" Upload an image to get started.")

        st.markdown("</div>", unsafe_allow_html=True)  # end main-card

    
if __name__ == "__main__":
    main()
