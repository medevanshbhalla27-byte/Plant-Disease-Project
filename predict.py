"""Predict the class of a single leaf image using the saved model."""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from config import IMAGE_SIZE, OUTPUT_DIR

def predict(image_path: Path):
    model_path = OUTPUT_DIR / "plant_model.keras"
    labels_path = OUTPUT_DIR / "class_names.json"

    if not model_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "Trained model or class_names.json not found. Run python step5.py first."
        )
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = tf.keras.models.load_model(model_path)
    with labels_path.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(IMAGE_SIZE)
        image_array = np.asarray(image, dtype=np.float32) / 255.0

    batch = np.expand_dims(image_array, axis=0)
    probabilities = model.predict(batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    print(f"\nPredicted class: {class_names[predicted_index]}")
    print(f"Confidence: {probabilities[predicted_index] * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Path to a leaf image")
    args = parser.parse_args()
    predict(args.image)
