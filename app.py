import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "plantguard_model.keras"
CLASS_NAMES_PATH = ROOT_DIR / "models" / "class_names.json"

IMG_SIZE = 160

st.set_page_config(
    page_title="PlantGuard AI",
    page_icon="🌿",
    layout="centered",
)

# --------------------------------------------------
# Load model and class names
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_disease(image, model, class_names):
    image = image.convert("RGB")
    resized_image = image.resize((IMG_SIZE, IMG_SIZE))

    image_array = np.asarray(resized_image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # The model already contains its own Rescaling layer.
    probabilities = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])

    return (
        class_names[predicted_index],
        confidence,
        probabilities,
    )


# --------------------------------------------------
# User interface
# --------------------------------------------------

st.title("🌿 PlantGuard AI")
st.subheader("Tomato Leaf Disease Detection")

st.write(
    "Upload a tomato leaf image to get an AI-based prediction "
    "from one of the 10 supported tomato leaf classes."
)

st.info(
    "For best results, upload a clear image with the leaf visible "
    "and minimal background clutter."
)

uploaded_file = st.file_uploader(
    "Choose a tomato leaf image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded leaf image",
            use_container_width=True,
        )

        if st.button("🔍 Predict Disease", type="primary"):
            if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
                st.error(
                    "Model files are missing. Check the models folder."
                )
            else:
                with st.spinner("Analyzing the leaf image..."):
                    model = load_model()
                    class_names = load_class_names()

                    predicted_class, confidence, probabilities = (
                        predict_disease(image, model, class_names)
                    )

                readable_name = predicted_class.replace(
                    "Tomato___", ""
                ).replace("_", " ")

                st.markdown("---")
                st.subheader("Prediction Result")

                st.success(f"Predicted class: **{readable_name}**")
                st.metric(
                    "Model confidence",
                    f"{confidence * 100:.2f}%",
                )

                st.caption(
                    "Confidence is the model's predicted probability, "
                    "not a guarantee that the classification is correct."
                )

                st.subheader("Class probabilities")

                probability_data = {
                    name.replace("Tomato___", "").replace("_", " "):
                    float(probability)
                    for name, probability in zip(
                        class_names, probabilities
                    )
                }

                probability_data = dict(
                    sorted(
                        probability_data.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                )

                st.bar_chart(probability_data)

                st.warning(
                    "This tool provides an experimental AI prediction "
                    "and is not a substitute for expert plant-disease "
                    "assessment. Confirm uncertain results with a "
                    "qualified agricultural expert."
                )

    except Exception as error:
        st.error(f"Unable to process the image: {error}")

st.markdown("---")
st.caption(
    "PlantGuard AI | Computer Vision & CNN Certification Project"
)
