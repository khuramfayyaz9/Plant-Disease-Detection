import streamlit as st
from PIL import Image

from prediction import predict_image
from disease_info import disease_info

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Plant Disease Detection using AI")

st.write(
    "Upload a leaf image and let the AI identify the disease."
)

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:

        if st.button("🔍 Predict Disease"):

            predicted_class, confidence, top3 = predict_image(image)

            nice_name = predicted_class.replace("_", " ")

            st.success(f"🌿 Prediction: {nice_name}")

            st.metric(
                label="Confidence",
                value=f"{confidence*100:.2f}%"
            )

            st.progress(confidence)

            if predicted_class in disease_info:

                st.subheader("📖 Disease Description")
                st.write(
                    disease_info[predicted_class]["description"]
                )

                st.subheader("💊 Treatment")
                st.write(
                    disease_info[predicted_class]["treatment"]
                )

            st.subheader("📊 Top 3 Predictions")

            for disease, score in top3:

                disease = disease.replace("_", " ")

                st.write(
                    f"**{disease}** — {score*100:.2f}%"
                )

st.sidebar.title("🌿 About")

st.sidebar.info(
    """
This application uses a **MobileNetV2 Deep Learning Model**
trained on the PlantVillage dataset.

Supported Plants:

• Tomato
• Potato
• Bell Pepper

Total Classes: 15

Model Accuracy: ~90%
"""
)