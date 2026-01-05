import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy  # For entropy-based unknown detection

# Disable TensorFlow OneDNN optimization
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set thresholds
CONFIDENCE_THRESHOLD = 0.6  # Probability threshold
num_classes = 6  
ENTROPY_THRESHOLD = np.log2(num_classes) * 0.2  # 50% of max entropy

# Model directory and image size
MODEL_DIR = "models"
IMAGE_SIZE = (227, 227)

def extract_glcm_features(image):
    """Extracts GLCM features from an image."""
    img_gray = rgb2gray(image)
    img_gray = (img_gray * 255).astype(np.uint8)
    
    # Compute GLCM
    glcm = graycomatrix(img_gray, distances=[50], angles=[np.pi/2], levels=256)
    
    # Extract statistical properties
    features = np.array([
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0]
    ]).reshape(1, -1)
    
    # Normalize GLCM features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features

def create_app():
    """Creates Flask app instance."""
    app = Flask(__name__)
    CORS(app)
    return app

app = create_app()

@app.route('/', methods=['GET', 'POST'])
def choose_model():
    """Renders the model selection page."""
    return render_template('home.html')

@app.route('/classify', methods=['POST'])
def classify_images():
    """Handles multiple image classification requests."""
    images = request.files.getlist('images')  # Get multiple files
    model_name = request.form['model']
    model_path = os.path.join(MODEL_DIR, model_name)

    results = []  # Store predictions for all images

    try:
        # Load the selected model
        model = tf.keras.models.load_model(model_path)

        for image in images:
            image_path = os.path.join("uploads", image.filename)
            os.makedirs("uploads", exist_ok=True)
            image.save(image_path)

            # Preprocess image
            img = load_img(image_path, target_size=IMAGE_SIZE)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # If the model requires GLCM features, extract them
            if "glcm" in model_name.lower():
                glcm_features = extract_glcm_features(img_array[0])  
                predictions = model.predict([img_array, glcm_features])  
            else:
                predictions = model.predict(img_array)  

            probs = predictions[0]
            max_prob = float(np.max(probs))  # Highest probability
            predicted_class = int(np.argmax(probs))   

            # Compute entropy of the probability distribution
            ent = float(entropy(probs)) 

            # Classify as "Unknown" if entropy is high or max_prob is too low
            if ent > ENTROPY_THRESHOLD or max_prob < CONFIDENCE_THRESHOLD:
                predicted_class = "Unknown 🫥"
                max_prob = None

            results.append({
                "image_name": image.filename,
                "predicted_label": predicted_class,
            })
            print(f"ENTROPY_THRESHOLD: {ENTROPY_THRESHOLD}")
            print(f"Entropy: {ent}")
            print(f"ent > ENTROPY_THRESHOLD: {ent > ENTROPY_THRESHOLD}")
            print(f"ent > ENTROPY_THRESHOLD: {max_prob is not None and max_prob < CONFIDENCE_THRESHOLD}")

        return jsonify(results)

    except Exception as e:
        print(f"Error during classification: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
