import joblib
import numpy as np
import cv2 
import warnings
import os 
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from skimage.feature import hog, local_binary_pattern
from PIL import Image
warnings.filterwarnings("ignore")


MODEL_DIR = "models"

class ScrewClassifier():
    def __init__(self, model_fp="mlp_classifier_PblHogResnet18_861Features_model_v02.pkl", scaler_fp="scaler.pkl"):
        self.resnet18 = models.resnet18(pretrained=True)
        # Remove the fully connected layer to get feature vectors
        self.resnet18 = torch.nn.Sequential(*list(self.resnet18.children())[:-1])
        self.resnet18.eval()  # Set the model to evaluation mode
        
        # Step 2: Define Image Preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image to 224x224
            transforms.ToTensor(),          # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])

        self.scaler = joblib.load(os.path.join(MODEL_DIR, scaler_fp))
        # load trained model
        self.trained_model = joblib.load(os.path.join(MODEL_DIR, model_fp))
    
    def classify_screw(self, im_rgb, return_prob=False):
        im_g = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY)
        hog_features = self.extract_hog_features(im_g)
        lbp_features = self.extract_lbp_features(im_g)
        resnet_features = self.extract_resnet18_features(im_rgb)
        merged_features = np.concatenate([hog_features, lbp_features, resnet_features]).reshape(1,-1)
        merged_features_scaled = self.scaler.transform(merged_features)
        
        if not return_prob:
            return self.trained_model.predict(merged_features_scaled)[0]
        else:
            return self.trained_model.predict_proba(merged_features_scaled)[0]
        
    def extract_hog_features(self, im_g):
        resized_image = cv2.resize(im_g, (64, 64))
        hog_features, hog_image = hog(
            resized_image,
            orientations=9,  # Number of gradient orientations
            pixels_per_cell=(16, 16),  # Size of cell
            cells_per_block=(2, 2),  # Number of cells per block
            block_norm='L2-Hys',  # Block normalization
            visualize=True,  # Output HOG image
            feature_vector=True,  # Return features as a vector
            )
        return hog_features

    def extract_lbp_features(self, im_g):
        radius = 3  # Radius of the circular neighborhood
        n_points = 8 * radius  # Number of points in the circular neighborhood
        lbp = local_binary_pattern(im_g, n_points, radius, method='uniform')
        n_bins = 25
        lbp_histogram, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        
        # Normalize the histogram
        lbp_histogram = lbp_histogram.astype("float")
        lbp_histogram /= (lbp_histogram.sum() + 1e-6)
        return lbp_histogram 

    def extract_resnet18_features(self, im_rgb):
        image = Image.fromarray(im_rgb)
        input_tensor = self.preprocess(image).unsqueeze(0)  # Preprocess and add batch dimension
        with torch.no_grad():
            features = self.resnet18(input_tensor).squeeze().numpy()  # Extract and convert to NumPy array
        return features 
    


