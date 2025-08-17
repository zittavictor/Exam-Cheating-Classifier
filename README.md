# Image Classification Comparison: SVM+HOG vs CNN

This project implements and compares two different approaches for image classification:

1. **SVM + HOG (Histogram of Oriented Gradients)**
2. **CNN (Convolutional Neural Network)**

## 🎯 Features

- **Comprehensive Comparison**: Accuracy, training time, and inference time analysis
- **Configurable Hyperparameters**: Easy to modify parameters for both models
- **Data Augmentation**: Enhanced CNN training with image transformations
- **PNG Support**: Optimized for PNG image formats
- **Detailed Visualizations**: Performance comparison charts and confusion matrices
- **Flexible Dataset Loading**: Supports any dataset organized by class directories

## 📋 Requirements

```bash
pip install -r requirements.txt
```

## 🗂️ Dataset Structure

Organize your dataset with subdirectories for each class:

```
dataset/
├── class1/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── class2/
│   ├── image3.png
│   ├── image4.png
│   └── ...
└── class3/
    ├── image5.png
    └── ...
```

## 🚀 Usage

### Basic Usage

```bash
python main.py --data_dir /path/to/your/dataset
```

### Advanced Options

```bash
# Quick test with reduced epochs
python main.py --data_dir dataset --quick_test

# Disable data augmentation
python main.py --data_dir dataset --no_augmentation

# Custom dataset path
python main.py --data_dir /custom/path/to/dataset
```

## ⚙️ Configuration

All hyperparameters can be modified in `config.py`:

### SVM + HOG Configuration
- HOG parameters (orientations, pixels per cell, etc.)
- SVM parameters (C, kernel, gamma)
- Feature scaling options

### CNN Configuration
- Network architecture (conv layers, dense layers)
- Training parameters (epochs, batch size, learning rate)
- Data augmentation settings
- Early stopping configuration

## 📊 Results

The program generates:

1. **Accuracy Comparison**: Overall and per-class accuracy
2. **Training Time Analysis**: Time comparison between models
3. **Inference Speed**: Per-sample prediction time
4. **Confusion Matrices**: Detailed error analysis
5. **Classification Reports**: Precision, recall, and F1-scores

All results are saved in the `results/` directory with both visualizations and JSON summaries.

## 🔧 Model Details

### SVM + HOG
- **Feature Extraction**: HOG features from grayscale images
- **Classification**: RBF kernel SVM with probability estimates
- **Preprocessing**: Standard scaling of features
- **Advantages**: Fast training, good with small datasets

### CNN
- **Architecture**: 3 convolutional layers with max pooling
- **Dense Layers**: 2 fully connected layers with dropout
- **Optimization**: Adam optimizer with configurable learning rate
- **Augmentation**: Rotation, shifting, shearing, zooming, flipping
- **Advantages**: High accuracy, automatic feature learning

## 📈 Expected Performance

- **SVM + HOG**: Faster training, good baseline accuracy
- **CNN**: Higher accuracy potential, better with larger datasets
- **Trade-offs**: Accuracy vs. computational efficiency

## 🛠️ Customization

### Adding New Models
1. Create a new model class following the same interface
2. Implement `train()`, `predict()`, and `evaluate()` methods
3. Add configuration to `config.py`
4. Update `main.py` to include the new model

### Modifying Architectures
- **SVM**: Adjust kernel type, C parameter, gamma in `config.py`
- **CNN**: Modify layer configurations, add/remove layers
- **Data Augmentation**: Customize augmentation parameters

## 📝 Output Files

- `model_comparison.json`: Complete results and model information
- `accuracy_comparison.png`: Accuracy visualization
- `training_time_comparison.png`: Training time comparison
- `inference_time_comparison.png`: Inference speed comparison
- `confusion_matrices.png`: Confusion matrices for both models
- `detailed_metrics.png`: Precision, recall, and F1-score comparison

## 🔍 Troubleshooting

### Common Issues

1. **Dataset Not Found**: Ensure the dataset directory exists and follows the expected structure
2. **Memory Issues**: Reduce batch size or image size in `config.py`
3. **Long Training Times**: Use `--quick_test` flag for initial testing
4. **GPU Issues**: TensorFlow will automatically use available GPUs

### Performance Tips

- Use smaller image sizes for faster training
- Adjust batch size based on available memory
- Enable early stopping to prevent overfitting
- Use data augmentation for better CNN performance

## 🤝 Contributing

Feel free to contribute by:
- Adding new model architectures
- Improving visualization
- Optimizing performance
- Adding new evaluation metrics

## 📄 License

This project is open source and available under the MIT License.