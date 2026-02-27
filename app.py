import os
import json
import secrets
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
IMG_SIZE = (224, 224)
MODEL_PATH = 'mobilenetv2_best.keras'

# ── Disease information dictionary ──────────────────────────────────────────
DISEASE_INFO = {
    'healthy': {
        'status': 'healthy',
        'recommendations': [
            'Continue regular watering and care',
            'Ensure adequate sunlight and nutrients',
            'Monitor for any changes in appearance',
            'Maintain good air circulation'
        ]
    },
    'default_disease': {
        'status': 'diseased',
        'recommendations': [
            'Isolate affected plants to prevent spread',
            'Consult with an agricultural expert',
            'Consider appropriate treatment methods',
            'Monitor other plants for similar symptoms'
        ]
    }
}

model = None
class_indices = None
class_names = None

def load_model():
    global model, class_indices, class_names
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        class_names = {v: k for k, v in class_indices.items()}
        print("Model loaded successfully from", MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(filepath):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img = Image.open(filepath).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def predict_image(filepath):
    img_array = preprocess_image(filepath)
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0])) * 100
    class_label = class_names[predicted_class_idx]

    # Parse class label: format is "Plant___Condition"
    parts = class_label.split('___')
    plant_name = parts[0].replace('_', ' ') if len(parts) > 0 else 'Unknown'
    condition = parts[1].replace('_', ' ') if len(parts) > 1 else class_label.replace('_', ' ')

    is_healthy = 'healthy' in condition.lower()
    if is_healthy:
        info = DISEASE_INFO['healthy']
    else:
        info = DISEASE_INFO['default_disease']

    return {
        'plant': plant_name,
        'condition': condition,
        'is_healthy': is_healthy,
        'confidence': round(confidence, 2),
        'status': info['status'],
        'recommendations': info['recommendations']
    }

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            prediction = predict_image(filepath)
            static_filename = f"upload_{secrets.token_hex(8)}.jpg"
            static_path = os.path.join(app.config['STATIC_FOLDER'], 'images', static_filename)
            Image.open(filepath).save(static_path)
            session['prediction'] = prediction
            session['image_path'] = f"images/{static_filename}"
            os.remove(filepath)
            return jsonify({'success': True})
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/result')
def result():
    prediction = session.get('prediction')
    image_path = session.get('image_path')
    if not prediction:
        return redirect(url_for('upload'))
    return render_template('result.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5050)
