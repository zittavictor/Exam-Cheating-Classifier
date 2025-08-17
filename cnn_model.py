"""
CNN Model Implementation
"""

import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

from config import CNN_CONFIG

class CNNModel:
    def __init__(self, num_classes, config=None):
        self.config = config or CNN_CONFIG
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.is_trained = False
        self.training_time = 0
        
        # Build model
        self._build_model()
    
    def _build_model(self):
        """Build CNN architecture"""
        print("Building CNN model...")
        
        self.model = Sequential()
        
        # Input layer
        input_shape = self.config['architecture']['input_shape']
        
        # Convolutional layers
        for i, conv_config in enumerate(self.config['architecture']['conv_layers']):
            if i == 0:
                # First layer needs input shape
                self.model.add(Conv2D(
                    filters=conv_config['filters'],
                    kernel_size=conv_config['kernel_size'],
                    activation=conv_config['activation'],
                    input_shape=input_shape
                ))
            else:
                self.model.add(Conv2D(
                    filters=conv_config['filters'],
                    kernel_size=conv_config['kernel_size'],
                    activation=conv_config['activation']
                ))
            
            # Add pooling after each conv layer
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Flatten before dense layers
        self.model.add(Flatten())
        
        # Dense layers
        for dense_config in self.config['architecture']['dense_layers']:
            self.model.add(Dense(
                units=dense_config['units'],
                activation=dense_config['activation']
            ))
            
            # Add dropout if specified
            if 'dropout' in dense_config:
                self.model.add(Dropout(dense_config['dropout']))
        
        # Output layer
        self.model.add(Dense(
            units=self.num_classes,
            activation=self.config['architecture']['output_activation']
        ))
        
        # Compile model
        optimizer = Adam(learning_rate=self.config['compilation']['learning_rate'])
        
        self.model.compile(
            optimizer=optimizer,
            loss=self.config['compilation']['loss'],
            metrics=self.config['compilation']['metrics']
        )
        
        print("CNN model built successfully!")
        self.model.summary()
    
    def train(self, X_train, y_train, X_val=None, y_val=None, data_generator=None):
        """Train the CNN model"""
        print("Training CNN model...")
        
        # Setup callbacks
        callbacks = []
        
        # Early stopping
        early_stopping_config = self.config['training']['early_stopping']
        early_stopping = EarlyStopping(
            monitor=early_stopping_config['monitor'],
            patience=early_stopping_config['patience'],
            restore_best_weights=early_stopping_config['restore_best_weights']
        )
        callbacks.append(early_stopping)
        
        # Training parameters
        epochs = self.config['training']['epochs']
        batch_size = self.config['training']['batch_size']
        validation_split = self.config['training']['validation_split']
        
        start_time = time.time()
        
        if data_generator is not None:
            # Train with data augmentation
            print("Training with data augmentation...")
            
            # Calculate steps
            steps_per_epoch = len(X_train) // batch_size
            
            # Validation data
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
            else:
                validation_data = None
            
            self.history = self.model.fit(
                data_generator,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Train without data augmentation
            print("Training without data augmentation...")
            
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
        
        self.training_time = time.time() - start_time
        self.is_trained = True
        
        print(f"Training completed in {self.training_time:.2f} seconds")
        
        return self
    
    def predict(self, X_test):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get predictions
        predictions = self.model.predict(X_test)
        
        # Convert probabilities to class predictions
        predicted_classes = np.argmax(predictions, axis=1)
        
        return predicted_classes
    
    def predict_proba(self, X_test):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        probabilities = self.model.predict(X_test)
        
        return probabilities
    
    def evaluate(self, X_test, y_test, class_names):
        """Evaluate model performance"""
        print("Evaluating CNN model...")
        
        # Make predictions
        start_time = time.time()
        predictions = self.predict(X_test)
        inference_time = (time.time() - start_time) / len(X_test)
        
        # Convert categorical y_test to class indices if needed
        if len(y_test.shape) > 1:
            y_test_classes = np.argmax(y_test, axis=1)
        else:
            y_test_classes = y_test
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_classes, predictions)
        
        # Generate classification report
        report = classification_report(
            y_test_classes, predictions, 
            target_names=class_names, 
            output_dict=True
        )
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'training_time': self.training_time,
            'inference_time_per_sample': inference_time,
            'predictions': predictions,
            'model_name': 'CNN',
            'history': self.history.history if self.history else None
        }
        
        print(f"CNN Results:")
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
            'model_type': 'CNN',
            'architecture': self.config['architecture'],
            'compilation': self.config['compilation'],
            'training': self.config['training'],
            'augmentation': self.config['augmentation'],
            'is_trained': self.is_trained,
            'training_time': self.training_time,
            'num_parameters': self.model.count_params() if self.model else 0
        }