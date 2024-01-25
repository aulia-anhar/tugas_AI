import cv2
import numpy as np
from scipy import io

class GLCMFeatureExtractor:
    def __init__(self):  # image instead of F
        self.image = None
        self.tinggi = None
        self.lebar = None
        self.dim = None 
        self.G0 = None
        self.G45 = None 
        self.G90 = None
        self.G135 = None

    def compute_glcm_rap(self):
        GLCM0, total_piksel0 = np.zeros((256, 256)), 0
        GLCM45, total_piksel45 = np.zeros((256, 256)), 0
        GLCM90, total_piksel90 = np.zeros((256, 256)), 0
        GLCM135, total_piksel135 = np.zeros((256, 256)), 0

        for y in range(1, self.tinggi - 1):
            for x in range(1, self.lebar - 1):
                a = self.image[y, x]
                b = self.image[y, x + 1]
                GLCM0[a, b] += 1
                total_piksel0 += 1

                a = self.image[y, x]
                b = self.image[y - 1, x + 1]
                GLCM45[a, b] += 1
                total_piksel45 += 1

                a = self.image[y, x]
                b = self.image[y - 1, x]
                GLCM90[a, b] += 1
                total_piksel90 += 1

                a = self.image[y, x]
                b = self.image[y - 1, x - 1]
                GLCM135[a, b] += 1
                total_piksel135 += 1

        GLCM0 /= total_piksel0
        GLCM45 /= total_piksel45
        GLCM90 /= total_piksel90
        GLCM135 /= total_piksel135

        G0 = self.compute_features(GLCM0)
        G45 = self.compute_features(GLCM45)
        G90 = self.compute_features(GLCM90)
        G135 = self.compute_features(GLCM135)

        return G0, G45, G90, G135

    def compute_features(self, GLCM):
        asm = np.sum(GLCM ** 2)
        kontras = np.sum((np.arange(256)[:, None] - np.arange(256)) ** 2 * GLCM)
        idm = np.sum(GLCM / (1 + (np.arange(256)[:, None] - np.arange(256)) ** 2))
        entropi = -np.sum(GLCM * np.where(GLCM != 0, np.log(GLCM), 0))
        
        px = np.sum(GLCM, axis=1)
        py = np.sum(GLCM, axis=0)
        px_py = np.outer(px, py)
        sigma_x = np.sqrt(np.sum((np.arange(256) - px) ** 2 * px))
        sigma_y = np.sqrt(np.sum((np.arange(256) - py) ** 2 * py))
        korelasi = np.sum((np.arange(256)[:, None] - px) * (np.arange(256) - py) * GLCM) / (sigma_x * sigma_y)

        features = {'asm': asm, 'kontras': kontras, 'idm': idm, 'entropi': entropi, 'korelasi': korelasi}
        return features


    #@classmethod
    def compute_glcm_features_for_channel(self, channel_image):        
        self.tinggi, self.lebar = channel_image.shape
        G0, G45, G90, G135 = self.compute_glcm_rap()
        features = np.concatenate((list(G0.values()), list(G45.values()), list(G90.values()), list(G135.values())))
        return features

    #@classmethod
    def compute_glcm_features(self, image):        
        self.image = image
        if len(image.shape) == 3:
            blue, green, red = cv2.split(image)
            blue_features = self.compute_glcm_features_for_channel(blue)
            green_features = self.compute_glcm_features_for_channel(green)
            red_features = self.compute_glcm_features_for_channel(red)
            main_features = np.concatenate((blue_features, green_features, red_features))
            main_features = main_features.reshape(1, -1)
        else:
            glcm = self(image)
            G0, G45, G90, G135 = glcm.compute_glcm_rap()
            main_features = np.concatenate((list(G0.values()), list(G45.values()), list(G90.values()), list(G135.values())))
            main_features = main_features.reshape(1, -1)
        return main_features  # return main_features instead of main_features.reshape(1, -1)
    
    def print_features(self, features):
        print("Gray Level Co-occurence Matrix (GLCM)")
        print("Main features:")
        print(features)
        print("Shape Main features:")
        print(features.shape)
        #print("Label features:")
        #print(self.labels_mean)
        
if __name__ == "__main__":
    image_path = "C:/Users/Anhar/Documents/AI/AI_revisi/backend_classification/dataset/Car_lite/Cangkir/2396346_6bd3750b-3896-491b-a5f0-b567f14a49f6_554_554.jpg"
    image = cv2.imread(image_path)

    feature_extractor = GLCMFeatureExtractor()
    features = feature_extractor.compute_glcm_features(image)
    print("Main features:")
    print(features)
    print("Shape Main features:")
    print(features.shape)