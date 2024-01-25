from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import os
import cv2
import sys
sys.path.append('C:/Users/Anhar/Documents/AI/AI_revisi')

from backend_classification.test_classification import ImageClassifierTester

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'C:/Users/Anhar/Documents/AI/AI_revisi/server/uploads_images'  # Sesuaikan dengan lokasi penyimpanan Anda
app.config['PREPROCESS_FOLDER'] = 'C:/Users/Anhar/Documents/AI/AI_revisi/server/preprocess_images'  # Folder untuk hasil pengolahan citra

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    filename = secure_filename(file.filename)
    file_extension = os.path.splitext(filename)[1].lower()
    
    if file_extension != '.jpg' and file_extension != '.jpeg' and file_extension != '.png':
        return jsonify({'error': 'Citra harus dalam format JPG.'})
    else:   
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Proses citra menggunakan OpenCV atau library lain
        image = cv2.imread(file_path)
        # Tambahkan kode pemrosesan citra di sini
        # processed_image = process_image(image)  # Asumsi process_image adalah fungsi Anda untuk memproses citra
        prediction, features, image = processed_image(file_path)
        # Simpan citra yang telah diproses
        processed_filename = 'processed_' + filename
        processed_file_path = os.path.join(app.config['PREPROCESS_FOLDER'], processed_filename)
        cv2.imwrite(processed_file_path, image)

        return jsonify(prediction)  # Mengirimkan hasil prediksi ke klien dalam format JSON

def processed_image(file_path):
    MODEL_DIR = 'C:/Users/Anhar/Documents/AI/AI_revisi/backend_classification/model'
    FEATURE_DIR = 'C:/Users/Anhar/Documents/AI/AI_revisi/backend_classification/fitur'
    FEATURE_TYPE = 'histogram'  # choose from 'histogram', 'glcm', or 'histogram_glcm'
    CLASSIFIER_TYPE = "mlp"  # "mlp", "naive_bayes"

    TEST_IMAGE_PATH = file_path

    # Create an instance of ImageClassifierTester
    tester = ImageClassifierTester(MODEL_DIR, FEATURE_DIR, FEATURE_TYPE)
    tester.load_data()
    tester.load_classifier(CLASSIFIER_TYPE)

    # Test the classifier on the test image
    prediction, features, image = tester.test_classifier(TEST_IMAGE_PATH)
    print("Prediction:", prediction)
    return prediction, features, image

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
