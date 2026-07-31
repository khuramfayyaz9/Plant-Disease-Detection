import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras")
CLASS_PATH = os.path.join(BASE_DIR, "models", "class_names.json")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as file:
    class_names = json.load(file)


def predict_image(image):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image).astype(np.float32)

    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = float(prediction[0][predicted_index])

    top3_idx = np.argsort(prediction[0])[-3:][::-1]

    top3 = []

    for idx in top3_idx:
        top3.append(
            (
                class_names[idx],
                float(prediction[0][idx])
            )
        )

    return predicted_class, confidence, top3