import logging
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import os
import cv2
import json

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

logging.info("Starting Flask app initialization...")

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Haar Cascade
logging.info("Loading Haar cascade for face detection.")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load label to ID mapping
logging.info("Loading label to ID mapping from JSON.")
with open('label_to_id.json', 'r') as f:
    label_to_id = json.load(f)

# Create ID to label mapping
id_to_label = {v: k for k, v in label_to_id.items()}

# Model configurations
model_info = {
    "Inception_ResNet": {"path": "models/inception_resnet_97", "input_size": (160, 160)},
    "DeepID": {"path": "models/DeepID_71", "input_size": (55, 47)},
    "ArcFace": {"path": "models/ArcFace_84", "input_size": (112, 112)},
    "FaceNet": {"path": "models/faceNet_80", "input_size": (96, 96)},
}

def contains_face(image_path):
    logging.debug(f"Checking for faces in image: {image_path}")
    try:
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Failed to read image: {image_path}")
            return False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        logging.debug(f"Faces detected: {len(faces)}")
        return len(faces) > 0
    except Exception as e:
        logging.exception(f"Error processing image {image_path}: {e}")
        return False

def preprocess_image(image_path, image_size):
    logging.debug(f"Preprocessing image: {image_path} with size: {image_size}")
    try:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size)
        img = img / 255.0
        img = tf.expand_dims(img, axis=0)
        return img
    except Exception as e:
        logging.exception(f"Error preprocessing image {image_path}: {e}")
        raise

@app.route('/')
def index():
    return render_template('index.html', models=model_info.keys())

@app.route('/predict', methods=['POST'])
def predict():
    logging.info("Received prediction request.")
    selected_model = request.form['model']
    img_file = request.files['image']

    logging.debug(f"Selected model: {selected_model}")
    if img_file:
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
        img_file.save(img_path)
        logging.info(f"Image saved to {img_path}")

        if not contains_face(img_path):
            logging.info("No face detected in image.")
            return jsonify({'prediction': "No face found in the image.", 'image_path': '/' + img_path})

        model_data = model_info[selected_model]
        model_path = model_data["path"]
        input_size = model_data["input_size"]

        try:
            model = load_model(model_path)
            logging.info(f"Model loaded from {model_path}")
        except Exception as e:
            logging.exception(f"Failed to load model: {e}")
            return jsonify({'prediction': "Error loading model.", 'image_path': '/' + img_path})

        try:
            img_tensor = preprocess_image(img_path, input_size)
            preds = model.predict(img_tensor)
            predicted_class_id = int(np.argmax(preds, axis=1)[0])
            predicted_name = id_to_label.get(predicted_class_id, "Unknown")

            prediction = f"Predicted: {predicted_name} (ID: {predicted_class_id})"
            logging.info(f"Prediction result: {prediction}")

        except Exception as e:
            logging.exception(f"Prediction failed: {e}")
            prediction = "Error during prediction."

        return jsonify({'prediction': prediction, 'image_path': '/' + img_path})

    return jsonify({'prediction': 'No image uploaded.', 'image_path': ''})

if __name__ == '__main__':
    logging.info("Starting Flask server on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
