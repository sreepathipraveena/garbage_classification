import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("best_model224.keras")

# Class labels (update if needed)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def predict_image(img):
    img = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = {class_names[i]: float(f"{predictions[0][i]:.4f}") for i in range(len(class_names))}
    return predicted_class, confidence

interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(source="webcam", type="pil", label="📷 Capture or Upload Image"),
    outputs=[gr.Label(num_top_classes=1), gr.Label()],
    title="🗑️ Garbage Classifier",
    description="Capture or upload an image to classify garbage type."
)

interface.launch()
