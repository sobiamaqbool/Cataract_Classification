import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import joblib
from PIL import Image
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained CNN-SVM pipeline
model_path = "/Users/siblingsmac/Desktop/nn_svm_pipeline.pkl"
nn_svm_pipeline = joblib.load(model_path)

# Extract CNN feature extractor from pipeline
feature_extractor = nn_svm_pipeline.named_steps['feature_extractor']
if isinstance(feature_extractor, tf.keras.Model):
    cnn_model = feature_extractor
else:
    raise ValueError("Feature extractor is not a valid Keras model!")

# Extract SVM classifier
svm_model = nn_svm_pipeline.named_steps['svm_classifier']

# Allowed image formats
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function: Maximum Entropy Thresholding
def maximum_entropy_threshold(img_gray):
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    cdf = hist.cumsum()
    entropy = -np.cumsum(hist * np.log2(hist + 1e-8))

    max_entropy, threshold = float('-inf'), 0
    for i in range(256):
        ent_b = entropy[i] / cdf[i] if cdf[i] > 0 else 0
        ent_f = (entropy[-1] - entropy[i]) / (1 - cdf[i]) if (1 - cdf[i]) > 0 else 0
        ent_sum = ent_b + ent_f
        if ent_sum > max_entropy:
            max_entropy = ent_sum
            threshold = i
    return threshold

# Function: Nonlinear Contrast Stretching
def nonlinear_contrast_stretching(img):
    min_val, max_val = np.percentile(img, [2, 98])
    img_stretched = np.clip((img - min_val) * (255 / (max_val - min_val + 1e-8)), 0, 255)
    return img_stretched.astype(np.uint8)

# Function: Preprocess Image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Maximum Entropy Thresholding
    threshold = maximum_entropy_threshold(gray_img)
    _, entropy_img = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)

    # Apply Edge Detection (Canny)
    edges = cv2.Canny(gray_img, 50, 150)

    # Apply Nonlinear Contrast Stretching
    contrast_img = nonlinear_contrast_stretching(gray_img)

    # Resize image for CNN input
    processed_img = cv2.resize(contrast_img, (224, 224))
    processed_img = np.expand_dims(processed_img, axis=-1)  # Add channel dimension
    processed_img = np.repeat(processed_img, 3, axis=-1)  # Convert to 3-channel
    processed_img = processed_img.astype(np.float32) / 255.0  # Normalize

    return np.expand_dims(processed_img, axis=0)  # Add batch dimension

# Homepage route
@app.route('/')
def index():
    return render_template('index.html')

# Upload and classify image
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess image
        preprocessed_img = preprocess_image(file_path)

        # Extract features using CNN
        extracted_features = cnn_model.predict(preprocessed_img)

        # Predict using SVM
        prediction = svm_model.predict(extracted_features)

        # Class labels
        class_labels = ['normal', 'cataract', 'diabetic_retinopathy', 'glaucoma']
        predicted_class = class_labels[int(prediction[0])]

        return render_template('result.html', filename=filename, predicted_class=predicted_class)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)



