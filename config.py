"""
Configuration file for hyperparameters and settings
"""

# Dataset Configuration
DATASET_CONFIG = {
    'data_dir': 'dataset',  # Update this path to your dataset location
    'image_size': (128, 128),  # Target image size for processing
    'test_size': 0.2,  # Train/test split ratio
    'random_state': 42,
    'supported_formats': ['.png'],  # PNG support as requested
}

# SVM + HOG Configuration
SVM_HOG_CONFIG = {
    'hog_params': {
        'orientations': 9,  # Number of orientation bins
        'pixels_per_cell': (8, 8),  # Size of a cell
        'cells_per_block': (2, 2),  # Number of cells in each block
        'block_norm': 'L2-Hys',  # Normalization method
        'visualize': False,
    },
    'svm_params': {
        'C': 1.0,  # Regularization parameter
        'kernel': 'rbf',  # Kernel type
        'gamma': 'scale',  # Kernel coefficient
        'random_state': 42,
        'probability': True,  # Enable probability estimates
    },
    'scaler': 'StandardScaler'  # Feature scaling method
}

# CNN Configuration
CNN_CONFIG = {
    'architecture': {
        'input_shape': (128, 128, 3),  # Input image shape
        'conv_layers': [
            {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'},
            {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
            {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu'},
        ],
        'dense_layers': [
            {'units': 128, 'activation': 'relu', 'dropout': 0.5},
            {'units': 64, 'activation': 'relu', 'dropout': 0.3},
        ],
        'output_activation': 'softmax'
    },
    'compilation': {
        'optimizer': 'adam',
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy'],
        'learning_rate': 0.001
    },
    'training': {
        'epochs': 50,
        'batch_size': 32,
        'validation_split': 0.2,
        'early_stopping': {
            'monitor': 'val_accuracy',
            'patience': 10,
            'restore_best_weights': True
        }
    },
    'augmentation': {
        'rotation_range': 20,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'shear_range': 0.2,
        'zoom_range': 0.2,
        'horizontal_flip': True,
        'fill_mode': 'nearest'
    }
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1-score'],
    'plot_style': 'seaborn-v0_8',
    'figure_size': (12, 8),
    'save_plots': True,
    'plot_dir': 'results',
}

# Timing Configuration
TIMING_CONFIG = {
    'inference_samples': 100,  # Number of samples for inference timing
    'timing_runs': 5,  # Number of timing runs for averaging
}