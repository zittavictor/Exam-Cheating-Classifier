"""
Demo script to showcase both SVM+HOG and CNN models
This script demonstrates the key features and provides examples
"""

import numpy as np
from data_loader import ImageDataLoader
from svm_hog_model import SVMHOGModel
from cnn_model import CNNModel
from evaluator import ModelEvaluator
import os

def demo_basic_functionality():
    """Demo basic functionality of both models"""
    print("🎯 DEMO: Image Classification Comparison System")
    print("="*60)
    
    # Check if sample dataset exists
    if not os.path.exists('dataset'):
        print("❌ Sample dataset not found!")
        print("Run: python create_sample_dataset.py")
        return
    
    # Load data
    print("📂 Loading sample dataset...")
    data_loader = ImageDataLoader(data_dir='dataset')
    X, y, class_names = data_loader.load_dataset()
    
    print(f"✅ Loaded {len(X)} images from {len(class_names)} classes")
    print(f"📊 Classes: {class_names}")
    
    # Prepare data for both models
    X_train_svm, X_test_svm, y_train_svm, y_test_svm = data_loader.prepare_data_for_svm(X, y)
    X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = data_loader.prepare_data_for_cnn(X, y)
    
    print(f"\n🔧 Training Data Split:")
    print(f"   Training samples: {len(X_train_svm)}")
    print(f"   Test samples: {len(X_test_svm)}")
    
    # Demo SVM + HOG
    print(f"\n🔧 SVM + HOG Demo:")
    print("-" * 30)
    svm_model = SVMHOGModel()
    
    # Show configuration
    model_info = svm_model.get_model_info()
    print(f"Model Type: {model_info['model_type']}")
    print(f"HOG Orientations: {model_info['hog_params']['orientations']}")
    print(f"SVM Kernel: {model_info['svm_params']['kernel']}")
    print(f"SVM C Parameter: {model_info['svm_params']['C']}")
    
    # Train and evaluate
    svm_model.train(X_train_svm, y_train_svm)
    svm_results = svm_model.evaluate(X_test_svm, y_test_svm, class_names)
    
    # Demo CNN
    print(f"\n🧠 CNN Demo:")
    print("-" * 30)
    cnn_model = CNNModel(num_classes=len(class_names))
    
    # Show configuration
    model_info = cnn_model.get_model_info()
    print(f"Model Type: {model_info['model_type']}")
    print(f"Total Parameters: {model_info['num_parameters']:,}")
    print(f"Conv Layers: {len(model_info['architecture']['conv_layers'])}")
    print(f"Dense Layers: {len(model_info['architecture']['dense_layers'])}")
    
    # Train with reduced epochs for demo
    print("Training CNN (reduced epochs for demo)...")
    cnn_model.config['training']['epochs'] = 3  # Reduce for demo
    
    # Create data generator for augmentation
    train_generator = data_loader.create_data_generator(X_train_cnn, y_train_cnn)
    cnn_model.train(X_train_cnn, y_train_cnn, data_generator=train_generator)
    cnn_results = cnn_model.evaluate(X_test_cnn, y_test_cnn, class_names)
    
    # Compare results
    print(f"\n📊 COMPARISON SUMMARY:")
    print("="*60)
    print(f"{'Metric':<20} {'SVM+HOG':<15} {'CNN':<15} {'Winner':<10}")
    print("-"*60)
    print(f"{'Accuracy':<20} {svm_results['accuracy']:<15.4f} {cnn_results['accuracy']:<15.4f} {'CNN' if cnn_results['accuracy'] > svm_results['accuracy'] else 'SVM':<10}")
    print(f"{'Training Time (s)':<20} {svm_results['training_time']:<15.2f} {cnn_results['training_time']:<15.2f} {'SVM' if svm_results['training_time'] < cnn_results['training_time'] else 'CNN':<10}")
    print(f"{'Inference (ms)':<20} {svm_results['inference_time_per_sample']*1000:<15.3f} {cnn_results['inference_time_per_sample']*1000:<15.3f} {'SVM' if svm_results['inference_time_per_sample'] < cnn_results['inference_time_per_sample'] else 'CNN':<10}")
    
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"   • CNN typically achieves higher accuracy")
    print(f"   • SVM+HOG trains much faster")
    print(f"   • SVM+HOG has faster inference")
    print(f"   • Choice depends on your priorities: accuracy vs speed")
    
    return svm_results, cnn_results, class_names

def demo_configuration_options():
    """Demo different configuration options"""
    print(f"\n⚙️  CONFIGURATION DEMO:")
    print("="*60)
    
    from config import SVM_HOG_CONFIG, CNN_CONFIG
    
    print("📋 SVM + HOG Configurable Parameters:")
    print(f"   HOG Orientations: {SVM_HOG_CONFIG['hog_params']['orientations']}")
    print(f"   Pixels per Cell: {SVM_HOG_CONFIG['hog_params']['pixels_per_cell']}")
    print(f"   SVM Kernel: {SVM_HOG_CONFIG['svm_params']['kernel']}")
    print(f"   SVM C Parameter: {SVM_HOG_CONFIG['svm_params']['C']}")
    
    print(f"\n📋 CNN Configurable Parameters:")
    print(f"   Conv Layers: {len(CNN_CONFIG['architecture']['conv_layers'])}")
    print(f"   Dense Layers: {len(CNN_CONFIG['architecture']['dense_layers'])}")
    print(f"   Epochs: {CNN_CONFIG['training']['epochs']}")
    print(f"   Batch Size: {CNN_CONFIG['training']['batch_size']}")
    print(f"   Learning Rate: {CNN_CONFIG['compilation']['learning_rate']}")
    
    print(f"\n📋 Data Augmentation Options:")
    for key, value in CNN_CONFIG['augmentation'].items():
        print(f"   {key}: {value}")

def demo_prediction_examples():
    """Demo making predictions on new data"""
    print(f"\n🔮 PREDICTION DEMO:")
    print("="*60)
    
    # This would demonstrate how to make predictions on new images
    print("📝 To make predictions on new images:")
    print("   1. Load and preprocess your images using ImageDataLoader")
    print("   2. Use model.predict(images) for class predictions")
    print("   3. Use model.predict_proba(images) for probability scores")
    print("   4. Both models return consistent prediction formats")

if __name__ == "__main__":
    try:
        # Run main demo
        svm_results, cnn_results, class_names = demo_basic_functionality()
        
        # Show configuration options
        demo_configuration_options()
        
        # Show prediction demo
        demo_prediction_examples()
        
        print(f"\n✅ DEMO COMPLETED!")
        print("="*60)
        print("🚀 To run full comparison: python main.py --data_dir dataset")
        print("📊 Results are saved in 'results/' directory")
        print("📖 See README.md for detailed documentation")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 Make sure to run: python create_sample_dataset.py first")