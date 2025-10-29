from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from werkzeug.utils import secure_filename
import gdown  # For downloading from Google Drive

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model file setup
MODEL_PATH = 'blood_group_model_final.h5'
DRIVE_FILE_ID = '1hh0ZvcF7AIEvQfcl0XrKfJ73UzvwHuh1'
DRIVE_URL = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'

# Download model from Google Drive if missing
if not os.path.exists(MODEL_PATH):
    print("Model file not found locally. Downloading from Google Drive...")
    try:
        gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading model: {e}")

# Load the trained model
try:
    model = load_model(MODEL_PATH)
    model_loaded = True
    print("Model loaded successfully!")

    # Blood group classes
    blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_blood_group(file_path):
    if not model_loaded:
        return {"error": "Model not loaded"}, 500

    try:
        # Preprocess the image
        processed_image = preprocess_image(file_path)

        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        predicted_blood_group = blood_groups[predicted_class]
        confidence = float(predictions[0][predicted_class]) * 100

        # Get all probabilities
        all_probabilities = {}
        for i, blood_group in enumerate(blood_groups):
            all_probabilities[blood_group] = float(predictions[0][i]) * 100

        return {
            "blood_group": predicted_blood_group,
            "confidence": confidence,
            "all_probabilities": all_probabilities
        }, 200
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get prediction
        result, status_code = predict_blood_group(filepath)

        return jsonify(result), status_code

    return jsonify({"error": "File type not allowed"}), 400

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
