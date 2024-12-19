from fastapi import FastAPI
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

model = load_model("model.keras")
class_names = ['backyard', 'bathroom', 'bedroom', 'frontyard', 'kitchen', 'livingRoom']
app = FastAPI()

def predict_image(image_path: str):
    img = load_img(image_path, target_size=(256, 256))
    img_array = img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(round(100 * np.max(predictions[0]), 2))
    return predicted_class, confidence

@app.get("/predict/{image_name}")
def predict(image_name: str):
    image_path = os.path.join(".", image_name)
    if not os.path.exists(image_path):
        return {"error": f"Image file '{image_name}' not found in the root directory."}
    predicted_class, confidence = predict_image(image_path)
    return {
        "filename": image_name,
        "predicted_class": predicted_class,
        "confidence": confidence
    }
