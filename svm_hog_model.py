"""
SVM + HOG Model Implementation
"""

import numpy as np
import time
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from skimage.feature import hog
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import SVM_HOG_CONFIG

class SVMHOGModel:
    def __init__(self, config=None):
        self.config = config or SVM_HOG_CONFIG
        self.hog_params = self.config['hog_params']
        self.svm_params = self.config['svm_params']
        
        # Initialize pipeline
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(**self.svm_params))
        ])
        
        self.is_trained = False
        self.training_time = 0
        
    def extract_hog_features(self, images):
        """Extract HOG features from images"""
        print("Extracting HOG features...")
        
        features = []
        for img in tqdm(images, desc="HOG extraction"):
            try:
                # Convert to grayscale if needed for HOG
                if len(img.shape) == 3:
                    # Convert RGB to grayscale
                    gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
                else:
                    gray = img
                
                # Ensure image is valid for HOG
                if gray.shape[0] < 32 or gray.shape[1] < 32:
                    print(f"Warning: Image too small for HOG: {gray.shape}")
                    features.append(np.zeros(1296))  # Default HOG feature size
                    continue
                
                # Extract HOG features
                hog_features = hog(
                    gray,
                    orientations=self.hog_params['orientations'],
                    pixels_per_cell=self.hog_params['pixels_per_cell'],
                    cells_per_block=self.hog_params['cells_per_block'],
                    block_norm=self.hog_params['block_norm'],
                    visualize=self.hog_params['visualize']
                )
                
                features.append(hog_features)
                
            except Exception as e:
                print(f"Error extracting HOG features: {e}")
                # Use zero features as fallback
                features.append(np.zeros(1296))  # Default HOG feature size
        
        return np.array(features)
    
    def train(self, X_train, y_train):
        """Train the SVM + HOG model"""
        print("Training SVM + HOG model...")
        
        # Extract HOG features
        X_train_hog = self.extract_hog_features(X_train)
        
        # Train the pipeline
        start_time = time.time()
        self.pipeline.fit(X_train_hog, y_train)
        self.training_time = time.time() - start_time
        
        self.is_trained = True
        print(f"Training completed in {self.training_time:.2f} seconds")
        
        return self
    
    def predict(self, X_test):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract HOG features
        X_test_hog = self.extract_hog_features(X_test)
        
        # Make predictions
        predictions = self.pipeline.predict(X_test_hog)
        
        return predictions
    
    def predict_proba(self, X_test):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract HOG features
        X_test_hog = self.extract_hog_features(X_test)
        
        # Get probabilities
        probabilities = self.pipeline.predict_proba(X_test_hog)
        
        return probabilities
    
    def evaluate(self, X_test, y_test, class_names):
        """Evaluate model performance"""
        print("Evaluating SVM + HOG model...")
        
        # Make predictions
        start_time = time.time()
        predictions = self.predict(X_test)
        inference_time = (time.time() - start_time) / len(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        
        # Generate classification report
        report = classification_report(
            y_test, predictions, 
            target_names=class_names, 
            output_dict=True
        )
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'training_time': self.training_time,
            'inference_time_per_sample': inference_time,
            'predictions': predictions,
            'model_name': 'SVM + HOG'
        }
        
        print(f"SVM + HOG Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Training Time: {self.training_time:.2f} seconds")
        print(f"  Inference Time: {inference_time:.6f} seconds per sample")
        
        return results
    
    def measure_inference_time(self, X_sample, num_runs=5):
        """Measure detailed inference time"""
        if not self.is_trained:
            raise ValueError("Model must be trained before measuring inference time")
        
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = self.predict(X_sample)
            end_time = time.time()
            times.append((end_time - start_time) / len(X_sample))
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times)
        }
    
    def get_model_info(self):
        """Get model configuration information"""
        return {
            'model_type': 'SVM + HOG',
            'hog_params': self.hog_params,
            'svm_params': self.svm_params,
            'is_trained': self.is_trained,
            'training_time': self.training_time
        }