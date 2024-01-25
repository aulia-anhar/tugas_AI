import os
import cv2
import numpy as np
import pickle

import sys
sys.path.append('C:/Users/Anhar/Documents/AI/AI_revisi/backend_classification')
from FeatureExtractor_GLCM import GLCMFeatureExtractor

class ImageClassifierTester:
    def __init__(self, model_dir, feature_dir, feature_type):
        self.model_dir = model_dir
        self.feature_dir = feature_dir
        self.feature_type = feature_type
        self.data = None
        self.labels = None
        self.classifier = None
        self.feature_extractors = {
            "histogram": self.extract_histogram,
            "glcm": self.extract_glcm
        }

    def extract_histogram(self, image):
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hist = hist.flatten()
        # Flatten and reshape histogram to 1-dimensional array
        hist = hist.reshape(1, -1)
        return hist

    def extract_glcm(self, image):
        feature_extractor = GLCMFeatureExtractor()
        glcm_features = feature_extractor.compute_glcm_features(image)
        return glcm_features

    def load_data(self):
        self.data = np.load(os.path.join(self.feature_dir, 'data.npy'))
        self.labels = np.load(os.path.join(self.feature_dir, 'labels.npy'))

    def load_classifier(self, classifier_type):
        model_file = os.path.join(self.model_dir, f'{classifier_type}_model.pkl')
        with open(model_file, 'rb') as f:
            self.classifier = pickle.load(f)

    def read_image(self, test_image_path):
        image = cv2.imread(test_image_path)
        return image

    def process_image(self, image):
        image = image
        return image

    def test_classifier(self, test_image_path):
        image = self.read_image(test_image_path)	
        image = self.process_image(image)
        features = self.feature_extractors[self.feature_type](image)
        features = features.reshape(1, -1)

        prediction = self.classifier.predict(features)
        return prediction[0], features, image


if __name__ == "__main__":
    MODEL_DIR = 'C:/Users/Anhar/Documents/AI/AI_revisi/backend_classification/model'
    FEATURE_DIR = 'C:/Users/Anhar/Documents/AI/AI_revisi/backend_classification/fitur'
    FEATURE_TYPE = 'histogram'  # choose from 'histogram', 'glcm', or 'histogram_glcm'
    CLASSIFIER_TYPE = "mlp"  # "mlp", "naive_bayes"

    TEST_IMAGE_PATH = 'C:/Users/Anhar/Documents/AI/AI_revisi/backend_classification/dataset/Car_lite/Cangkir/2396346_6bd3750b-3896-491b-a5f0-b567f14a49f6_554_554.jpg'
    

    # Create an instance of ImageClassifierTester
    tester = ImageClassifierTester(MODEL_DIR, FEATURE_DIR, FEATURE_TYPE)
    tester.load_data()
    tester.load_classifier(CLASSIFIER_TYPE)

    # Test the classifier on the test image
    prediction = tester.test_classifier(TEST_IMAGE_PATH)
    print("Prediction:", prediction)
