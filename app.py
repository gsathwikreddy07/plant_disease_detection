import os
import json
import secrets
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
from PIL import Image
from rembg import remove
import tensorflow as tf
from google import genai

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
IMG_SIZE = (224, 224)
MODEL_PATH = '/Users/satwikreddy/Desktop/gen ai/PlantCare_AI/models/mobilenetv2_final.keras'

# ── Gemini API Setup ───────────────────────────────────────────────────────────
GEMINI_API_KEY = "Enter your API KEY"   # ← paste your key from aistudio.google.com
client = genai.Client(api_key=GEMINI_API_KEY)

try:
    test = client.models.generate_content(
        model='gemini-2.5-flash',
        contents='say ok', 
    )
    print("✅ Gemini API connected successfully")
except Exception as e:
    print(f"❌ Gemini API connection failed: {e}")
    print("⚠️  App will use fallback recommendations")
# ── Auto-parse all disease info directly from class_indices.json ───────────────
def parse_class_label(class_label):
    """
    Parses dataset folder names into structured disease info.
    Example: 'Tomato___Early_blight'  → plant='Tomato',  condition='Early Blight'
    Example: 'Apple___healthy'        → plant='Apple',   is_healthy=True
    """
    parts = class_label.split('___')

    # Parse plant name — clean underscores and brackets
    plant = parts[0] \
        .replace('_', ' ') \
        .replace('(', '') \
        .replace(')', '') \
        .strip() \
        .title()

    # Parse condition name
    condition = parts[1].replace('_', ' ').strip().title() \
        if len(parts) > 1 else 'Unknown'

    is_healthy = 'healthy' in class_label.lower()

    # Auto-assign severity based on condition keywords
    condition_lower = condition.lower()
    if is_healthy:
        severity = 'None'
    elif any(w in condition_lower for w in ['virus', 'greening', 'mosaic', 'curl']):
        severity = 'Critical'
    elif any(w in condition_lower for w in ['late blight', 'rot', 'esca', 'measles']):
        severity = 'High'
    else:
        severity = 'Moderate'

    return {
        'display_name': condition,
        'plant'       : plant,
        'is_healthy'  : is_healthy,
        'severity'    : severity
    }


# Load class indices and auto-generate DISEASE_INFO
with open('class_indices (1).json', 'r') as f:
    class_indices_raw = json.load(f)

DISEASE_INFO = {
    label: parse_class_label(label)
    for label in class_indices_raw.keys()
}
print(f"✅ Auto-parsed {len(DISEASE_INFO)} disease classes from dataset")

# ── TensorFlow Model ───────────────────────────────────────────────────────────
model      = None
class_names = None

def load_model():
    global model, class_names
    try:
        model       = tf.keras.models.load_model(MODEL_PATH)
        class_names = {v: k for k, v in class_indices_raw.items()}
        print(f"✅ TF Model loaded — {len(class_names)} classes")
    except Exception as e:
        print(f"❌ Error loading model: {e}")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Gemini: Dynamic Recommendations ───────────────────────────────────────────
def get_ai_recommendations(plant, condition, is_healthy, severity, confidence):
    try:
        if is_healthy:
            prompt = f"""
You are an expert agricultural consultant.
A plant health AI detected a HEALTHY {plant} plant with {confidence}% confidence.

Give exactly 5 specific care tips to keep this {plant} healthy and productive.
Rules:
- Specific to {plant} plants only
- Actionable — something the farmer can do today
- One sentence each
- Plain numbered list 1 to 5
- No intro or conclusion text, just the 5 points
"""
        else:
            prompt = f"""
You are an expert plant pathologist.
A plant health AI detected {condition} in a {plant} plant.
Confidence: {confidence}% | Severity: {severity}

Give exactly 5 specific treatment steps for {condition} in {plant} plants.
Rules:
- Specific to {condition} in {plant} only
- In order of priority — most urgent first
- Mention specific fungicides or treatments by name where relevant
- One sentence each
- Plain numbered list 1 to 5
- No intro or conclusion text, just the 5 points
"""

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        raw_text = response.text.strip()
        lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
        recommendations = []
        for line in lines:
            cleaned = line.lstrip('0123456789.-) ').strip()
            if cleaned:
                recommendations.append(cleaned)

        return recommendations[:5]

    except Exception as e:
        print(f"⚠️ Gemini recommendations error: {e}")
        return None  # triggers fallback


# ── TTA Prediction with Background Removal ────────────────────────────────────
def predict_with_tta(filepath, n_augments=10):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    # Step 1: Remove background using rembg
    original   = Image.open(filepath).convert('RGB')
    removed_bg = remove(original)
    white_bg   = Image.new('RGB', removed_bg.size, (255, 255, 255))
    white_bg.paste(removed_bg, mask=removed_bg.split()[3])
    clean_leaf = white_bg.resize((224, 224))
    img_array  = np.array(clean_leaf, dtype=np.float32)

    # Step 2: Create augmented versions for TTA
    augments = []
    for _ in range(n_augments):
        aug = img_array.copy()

        # Random horizontal flip
        if np.random.rand() > 0.5:
            aug = aug[:, ::-1, :]

        # Random brightness adjustment
        factor = np.random.uniform(0.85, 1.15)
        aug    = np.clip(aug * factor, 0, 255)

        # Random 90° rotation
        aug_tensor = tf.image.rot90(
            tf.constant(aug, dtype=tf.float32),
            k=np.random.randint(0, 4)
        ).numpy()

        aug = preprocess_input(aug_tensor)
        augments.append(aug)

    # Step 3: Predict all augmentations and average
    batch          = np.stack(augments, axis=0)
    all_preds      = model.predict(batch, verbose=0)
    avg_pred       = np.mean(all_preds, axis=0)

    predicted_idx  = int(np.argmax(avg_pred))
    confidence     = float(np.max(avg_pred)) * 100

    return predicted_idx, confidence, avg_pred


# ── Main Prediction Function ───────────────────────────────────────────────────
def predict_image(filepath):
    predicted_idx, confidence, avg_pred = predict_with_tta(filepath)

    # Top 3 predictions
    top3_indices = np.argsort(avg_pred)[::-1][:3]
    top3 = [
        {
            'label'     : class_names[int(i)].replace('___', ' → ').replace('_', ' '),
            'confidence': round(float(avg_pred[i]) * 100, 2)
        }
        for i in top3_indices
    ]

    class_label = class_names[predicted_idx].strip()
    info        = DISEASE_INFO.get(class_label, parse_class_label(class_label))

    print(f"DEBUG — '{class_label}' | {confidence:.1f}% | {info['severity']}")

    # Get Gemini AI recommendations
    ai_recs = get_ai_recommendations(
        plant      = info['plant'],
        condition  = info['display_name'],
        is_healthy = info['is_healthy'],
        severity   = info['severity'],
        confidence = round(confidence, 2)
    )

    # Use AI recs if available, otherwise smart fallback
    if ai_recs:
        recommendations = ai_recs
        rec_source      = 'ai'
    else:
        recommendations = [
            f"Isolate the affected {info['plant']} plant immediately to prevent spread",
            "Remove and destroy all visibly infected leaves",
            "Consult a local agricultural expert for specific treatment options",
            "Monitor surrounding plants closely for similar symptoms"
        ] if not info['is_healthy'] else [
            f"Your {info['plant']} plant is healthy — maintain your current care routine",
            "Continue regular watering and feeding on schedule",
            "Monitor regularly for any early signs of change",
            "Ensure good air circulation around the plant"
        ]
        rec_source = 'fallback'

    return {
        'plant'          : info['plant'],
        'condition'      : info['display_name'],
        'is_healthy'     : info['is_healthy'],
        'confidence'     : round(confidence, 2),
        'severity'       : info['severity'],
        'recommendations': recommendations,
        'rec_source'     : rec_source,
        'low_confidence' : confidence < 85.0,
        'top3'           : top3
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

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
            prediction      = predict_image(filepath)
            static_filename = f"upload_{secrets.token_hex(8)}.jpg"
            static_path     = os.path.join('static', 'images', static_filename)
            Image.open(filepath).save(static_path)

            session['prediction'] = prediction
            session['image_path'] = f"images/{static_filename}"
            os.remove(filepath)

            return jsonify({'success': True})

        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            print(f"Prediction error: {e}")
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/result')
def result():
    prediction = session.get('prediction')
    image_path = session.get('image_path')
    if not prediction:
        return redirect(url_for('upload'))
    return render_template('result.html',
                           prediction=prediction,
                           image_path=image_path)


@app.route('/chat', methods=['POST'])
def chat():
    data         = request.get_json()
    user_message = data.get('message', '').strip()
    chat_history = data.get('history', [])
    prediction   = data.get('prediction', {})

    print(f"DEBUG CHAT — message: '{user_message}' | plant: {prediction.get('plant')}")

    if not user_message:
        return jsonify({'error': 'Empty message'}), 400

    try:
        plant      = prediction.get('plant', 'Unknown')
        condition  = prediction.get('condition', 'Unknown')
        is_healthy = prediction.get('is_healthy', False)
        severity   = prediction.get('severity', 'Unknown')
        confidence = prediction.get('confidence', 0)

        # Build full context prompt
        system_context = f"""
You are PlantCare AI, a friendly and expert agricultural assistant chatbot.

Current plant analysis result the farmer is viewing:
- Plant Type    : {plant}
- Condition     : {condition}
- Health Status : {'Healthy ✅' if is_healthy else 'Disease Detected ⚠️'}
- Severity      : {severity}
- AI Confidence : {confidence}%

Your role is to answer follow-up questions about this specific plant and condition.
Guidelines:
- Be friendly, specific, and practical
- Keep responses to 2–4 sentences — concise and clear
- Always relate answers back to {plant} and {condition} specifically
- If asked something unrelated to plant care, politely redirect back to the plant topic
- Never repeat the full analysis details unless specifically asked
"""

        # Build conversation string with last 6 messages for context
        conversation = system_context + "\n\nConversation:\n"
        for msg in chat_history[-6:]:
            role          = "Farmer" if msg['role'] == 'user' else "PlantCare AI"
            conversation += f"{role}: {msg['content']}\n"
        conversation += f"Farmer: {user_message}\nPlantCare AI:"

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=conversation,
        )
        reply    = response.text.strip()

        return jsonify({'reply': reply})

    except Exception as e:
        print(f"❌ Chat error: {e}")
        return jsonify({'reply': f"Sorry, I'm having trouble right now. Error: {str(e)}"}), 500


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5051)
