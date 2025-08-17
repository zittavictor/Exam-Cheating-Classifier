# 🚀 SVM+HOG vs CNN Image Classification Comparison

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/your-exam-repo/blob/main/SVM_vs_CNN_Image_Classification_Comparison.ipynb)

A comprehensive comparison between traditional computer vision (SVM + HOG) and deep learning (CNN) approaches for image classification.

## 🎯 Overview

This project implements and compares two different image classification methodologies:
- **SVM + HOG**: Traditional computer vision approach using Histogram of Oriented Gradients features with Support Vector Machine
- **CNN**: Deep learning approach using Convolutional Neural Networks with data augmentation

## 📊 Features

- 🔍 **Automatic GPU Detection**: Utilizes GPU acceleration when available
- 📈 **Comprehensive Analysis**: Accuracy, training time, and inference speed comparison
- 🎨 **Rich Visualizations**: Performance charts, confusion matrices, and detailed metrics
- 📥 **Separate Downloads**: Models, results, dataset, and configurations
- ⚙️ **Configurable Parameters**: Easy hyperparameter tuning for both approaches
- 🧪 **Reproducible Results**: Consistent random seeds and detailed documentation

## 🗂️ Dataset

The project includes a synthetic dataset with three classes:
- **Normal**: Blue-ish images with noise patterns
- **Cheating**: Red-ish images with diagonal line patterns  
- **Looking Around**: Green-ish images with circular patterns

**Dataset Statistics:**
- 📊 **Classes**: 3
- 🖼️ **Images**: 150 total (50 per class)
- 📏 **Size**: 128x128 pixels
- 🎨 **Format**: PNG images
- 🔀 **Split**: 80% training, 20% testing

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
1. Click the "Open in Colab" badge above
2. Run all cells in sequence
3. Download results using the provided download cells

### Option 2: Local Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/your-exam-repo.git
cd your-exam-repo

# Install dependencies
pip install -r requirements.txt

# Run the comparison
python main.py --data_dir dataset
```

## 📋 Requirements

### For Google Colab
All dependencies are automatically installed in the notebook.

### For Local Setup
```
numpy==1.24.3
opencv-python==4.8.1.78
scikit-learn==1.3.0
tensorflow==2.13.0
matplotlib==3.7.2
seaborn==0.12.2
Pillow==10.0.0
scikit-image==0.21.0
tqdm==4.66.1
```

## 🏗️ Architecture

### SVM + HOG Model
- **Feature Extraction**: Histogram of Oriented Gradients (HOG)
- **Classification**: Support Vector Machine with RBF kernel
- **Preprocessing**: Standard scaling of features
- **Advantages**: Fast training, interpretable features, good with small datasets

### CNN Model
- **Architecture**: 3 convolutional layers + 2 dense layers
- **Optimization**: Adam optimizer with early stopping
- **Augmentation**: Rotation, shifting, shearing, zooming, flipping
- **Advantages**: High accuracy potential, automatic feature learning

## 📈 Results

### Expected Performance Characteristics
- **SVM + HOG**: 
  - ✅ Faster training time
  - ✅ Faster inference speed
  - ✅ Lower memory usage
  - ⚠️ May have lower accuracy on complex patterns

- **CNN**: 
  - ✅ Higher accuracy potential
  - ✅ Better scalability
  - ✅ Automatic feature learning
  - ⚠️ Longer training time
  - ⚠️ Higher computational requirements

## 📁 Project Structure

```
your-exam-repo/
├── SVM_vs_CNN_Image_Classification_Comparison.ipynb  # Main Colab notebook
├── dataset/                                          # Image dataset
│   ├── normal/
│   │   ├── normal_001.png
│   │   └── ...
│   ├── cheating/
│   │   ├── cheating_001.png
│   │   └── ...
│   └── looking_around/
│       ├── looking_around_001.png
│       └── ...
├── results/                                          # Generated results
│   ├── model_comparison.json
│   ├── accuracy_comparison.png
│   ├── training_time_comparison.png
│   ├── inference_time_comparison.png
│   ├── confusion_matrices.png
│   └── detailed_metrics.png
├── README.md                                         # This file
└── requirements.txt                                  # Dependencies
```

## ⚙️ Configuration

Both models can be easily configured by modifying the configuration dictionaries in the notebook:

### SVM + HOG Configuration
```python
SVM_HOG_CONFIG = {
    'hog_params': {
        'orientations': 9,
        'pixels_per_cell': (8, 8),
        'cells_per_block': (2, 2),
    },
    'svm_params': {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
    }
}
```

### CNN Configuration
```python
CNN_CONFIG = {
    'training': {
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
    },
    'architecture': {
        'conv_layers': [
            {'filters': 32, 'kernel_size': (3, 3)},
            {'filters': 64, 'kernel_size': (3, 3)},
            {'filters': 128, 'kernel_size': (3, 3)},
        ]
    }
}
```

## 📊 Evaluation Metrics

The comparison includes:
- **Accuracy**: Overall and per-class performance
- **Training Time**: Model training duration
- **Inference Speed**: Prediction time per sample
- **Confusion Matrices**: Error analysis
- **Precision, Recall, F1-Score**: Detailed performance metrics

## 📥 Downloads

The notebook provides separate download packages for:
1. **Results Package**: JSON reports and performance metrics
2. **Models Package**: Trained SVM and CNN models
3. **Dataset Package**: Complete image dataset
4. **Configuration Package**: All hyperparameter settings

## 🔧 Customization

### Using Your Own Dataset
1. Organize your images in subdirectories by class
2. Update the `dataset_dir` path in the notebook
3. Ensure images are in supported formats (PNG, JPG)
4. Run the notebook to train and compare models

### Modifying Model Architectures
- **SVM**: Adjust kernel type, C parameter, gamma in config
- **CNN**: Modify layer configurations, add/remove layers
- **Augmentation**: Customize data augmentation parameters

## 🎮 GPU Acceleration

The notebook automatically detects and configures GPU usage:
- **Automatic Detection**: Checks for available GPUs
- **Memory Growth**: Enables dynamic GPU memory allocation
- **Mixed Precision**: Uses float16 for better performance when available
- **Fallback**: Gracefully falls back to CPU if no GPU is available

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory (GPU)**
   - Reduce batch size in CNN config
   - Reduce image size in dataset config

2. **Long Training Times**
   - Use GPU acceleration
   - Reduce number of epochs for testing
   - Use smaller dataset for initial experiments

3. **Import Errors**
   - Ensure all dependencies are installed
   - Use Google Colab for automatic dependency management

## 📚 Educational Value

This project demonstrates:
- **Traditional vs Modern Approaches**: SVM+HOG vs CNN comparison
- **Feature Engineering**: Manual (HOG) vs automatic (CNN) feature extraction
- **Performance Trade-offs**: Accuracy vs speed vs computational requirements
- **Best Practices**: Proper evaluation, visualization, and reproducibility

## 🤝 Contributing

Feel free to contribute by:
- Adding new model architectures
- Improving visualizations
- Adding new evaluation metrics
- Optimizing performance
- Enhancing documentation

## 📄 License

This project is open source and available under the MIT License.

## 🎯 Academic Use

Perfect for:
- Machine Learning coursework
- Computer Vision projects
- Algorithm comparison studies
- Research demonstrations
- Educational presentations

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Review the notebook comments and documentation
3. Open an issue in the repository
4. Refer to the TensorFlow and scikit-learn documentation

---

**🎉 Happy Learning and Experimenting!**

*This project showcases the evolution from traditional computer vision to deep learning approaches, providing hands-on experience with both methodologies.*