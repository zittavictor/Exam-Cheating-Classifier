"""
Data loading and preprocessing utilities
"""

import os
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import DATASET_CONFIG, CNN_CONFIG

class ImageDataLoader:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or DATASET_CONFIG['data_dir']
        self.image_size = DATASET_CONFIG['image_size']
        self.test_size = DATASET_CONFIG['test_size']
        self.random_state = DATASET_CONFIG['random_state']
        self.supported_formats = DATASET_CONFIG['supported_formats']
        
        self.label_encoder = LabelEncoder()
        self.class_names = []
        
    def load_dataset(self):
        """Load images from subdirectories organized by class"""
        print("Loading dataset...")
        
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Dataset directory '{self.data_dir}' not found!")
        
        images = []
        labels = []
        
        # Get class directories
        class_dirs = [d for d in os.listdir(self.data_dir) 
                     if os.path.isdir(os.path.join(self.data_dir, d))]
        
        if not class_dirs:
            raise ValueError(f"No class directories found in '{self.data_dir}'")
        
        self.class_names = sorted(class_dirs)
        print(f"Found {len(self.class_names)} classes: {self.class_names}")
        
        # Load images from each class
        for class_name in tqdm(self.class_names, desc="Loading classes"):
            class_path = os.path.join(self.data_dir, class_name)
            
            # Get all supported image files
            image_files = [f for f in os.listdir(class_path) 
                          if any(f.lower().endswith(ext) for ext in self.supported_formats)]
            
            if not image_files:
                print(f"Warning: No supported images found in '{class_path}'")
                continue
            
            print(f"  - {class_name}: {len(image_files)} images")
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                try:
                    # Load and preprocess image
                    image = self._load_and_preprocess_image(img_path)
                    if image is not None:
                        images.append(image)
                        labels.append(class_name)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
        
        if not images:
            raise ValueError("No images were successfully loaded!")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Dataset loaded: {len(X)} images, {len(self.class_names)} classes")
        print(f"Image shape: {X.shape}")
        
        return X, y_encoded, self.class_names
    
    def _load_and_preprocess_image(self, img_path):
        """Load and preprocess a single image"""
        try:
            # Load image using PIL (better PNG support)
            img = Image.open(img_path)
            
            # Convert to RGB if needed (handle RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize image
            img = img.resize(self.image_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Normalize pixel values to [0, 1]
            img_array = img_array.astype(np.float32) / 255.0
            
            return img_array
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None
    
    def prepare_data_for_svm(self, X, y):
        """Prepare data for SVM (flatten images)"""
        # Flatten images for SVM
        X_flat = X.reshape(X.shape[0], -1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_flat, y, test_size=self.test_size, 
            random_state=self.random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def prepare_data_for_cnn(self, X, y):
        """Prepare data for CNN (keep image structure)"""
        # Convert labels to categorical
        y_categorical = to_categorical(y, num_classes=len(self.class_names))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=self.test_size, 
            random_state=self.random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def create_data_generator(self, X_train, y_train, batch_size=32):
        """Create data generator with augmentation for CNN"""
        datagen = ImageDataGenerator(**CNN_CONFIG['augmentation'])
        
        # Fit the generator to training data
        datagen.fit(X_train)
        
        # Create generator
        generator = datagen.flow(X_train, y_train, batch_size=batch_size)
        
        return generator
    
    def get_class_distribution(self, y):
        """Get class distribution for analysis"""
        unique, counts = np.unique(y, return_counts=True)
        distribution = dict(zip(unique, counts))
        
        print("\nClass Distribution:")
        for i, class_name in enumerate(self.class_names):
            count = distribution.get(i, 0)
            print(f"  {class_name}: {count} images")
        
        return distribution