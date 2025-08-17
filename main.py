"""
Main script to run and compare SVM+HOG vs CNN models
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_loader import ImageDataLoader
from svm_hog_model import SVMHOGModel
from cnn_model import CNNModel
from evaluator import ModelEvaluator
from config import DATASET_CONFIG, SVM_HOG_CONFIG, CNN_CONFIG, TIMING_CONFIG

def main():
    """Main function to run the comparison"""
    parser = argparse.ArgumentParser(description='Compare SVM+HOG vs CNN for image classification')
    parser.add_argument('--data_dir', type=str, default='dataset', 
                       help='Path to dataset directory')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Custom configuration file')
    parser.add_argument('--no_augmentation', action='store_true',
                       help='Disable data augmentation for CNN')
    parser.add_argument('--quick_test', action='store_true',
                       help='Run with reduced epochs for quick testing')
    
    args = parser.parse_args()
    
    print("🚀 Starting Image Classification Comparison: SVM+HOG vs CNN")
    print("="*80)
    
    # Check if dataset directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Dataset directory '{args.data_dir}' not found!")
        print("\n📋 Expected dataset structure:")
        print("dataset/")
        print("├── class1/")
        print("│   ├── image1.png")
        print("│   └── image2.png")
        print("├── class2/")
        print("│   ├── image3.png")
        print("│   └── image4.png")
        print("└── ...")
        return
    
    try:
        # 1. Load and prepare data
        print("\n📂 Step 1: Loading and preparing data...")
        data_loader = ImageDataLoader(data_dir=args.data_dir)
        
        # Load raw data
        X, y, class_names = data_loader.load_dataset()
        
        # Show class distribution
        data_loader.get_class_distribution(y)
        
        # Prepare data for both models
        X_train_svm, X_test_svm, y_train_svm, y_test_svm = data_loader.prepare_data_for_svm(X, y)
        X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = data_loader.prepare_data_for_cnn(X, y)
        
        print(f"✅ Data prepared successfully!")
        print(f"   Training samples: {len(X_train_svm)}")
        print(f"   Test samples: {len(X_test_svm)}")
        print(f"   Classes: {len(class_names)}")
        
        # 2. Train SVM + HOG Model
        print("\n🔧 Step 2: Training SVM + HOG Model...")
        svm_model = SVMHOGModel(config=SVM_HOG_CONFIG)
        svm_model.train(X_train_svm, y_train_svm)
        
        # 3. Train CNN Model
        print("\n🧠 Step 3: Training CNN Model...")
        
        # Modify config for quick testing if requested
        cnn_config = CNN_CONFIG.copy()
        if args.quick_test:
            cnn_config['training']['epochs'] = 5
            cnn_config['training']['early_stopping']['patience'] = 2
        
        cnn_model = CNNModel(num_classes=len(class_names), config=cnn_config)
        
        # Setup data augmentation if not disabled
        if not args.no_augmentation:
            print("   Using data augmentation...")
            train_generator = data_loader.create_data_generator(
                X_train_cnn, y_train_cnn, 
                batch_size=cnn_config['training']['batch_size']
            )
            cnn_model.train(X_train_cnn, y_train_cnn, data_generator=train_generator)
        else:
            print("   Training without data augmentation...")
            cnn_model.train(X_train_cnn, y_train_cnn)
        
        # 4. Evaluate both models
        print("\n📊 Step 4: Evaluating models...")
        
        # Evaluate SVM
        svm_results = svm_model.evaluate(X_test_svm, y_test_svm, class_names)
        svm_results['y_true'] = y_test_svm  # Store true labels for confusion matrix
        
        # Evaluate CNN
        cnn_results = cnn_model.evaluate(X_test_cnn, y_test_cnn, class_names)
        # Convert categorical labels back to indices for confusion matrix
        if len(y_test_cnn.shape) > 1:
            cnn_results['y_true'] = np.argmax(y_test_cnn, axis=1)
        else:
            cnn_results['y_true'] = y_test_cnn
        
        # 5. Detailed timing analysis
        print("\n⏱️  Step 5: Detailed timing analysis...")
        
        # Sample for inference timing
        sample_size = min(TIMING_CONFIG['inference_samples'], len(X_test_svm))
        X_sample_svm = X_test_svm[:sample_size]
        X_sample_cnn = X_test_cnn[:sample_size]
        
        # Measure inference times
        svm_timing = svm_model.measure_inference_time(X_sample_svm, TIMING_CONFIG['timing_runs'])
        cnn_timing = cnn_model.measure_inference_time(X_sample_cnn, TIMING_CONFIG['timing_runs'])
        
        print(f"SVM Inference Timing (avg of {TIMING_CONFIG['timing_runs']} runs):")
        print(f"  Mean: {svm_timing['mean_time']*1000:.3f} ms/sample")
        print(f"  Std:  {svm_timing['std_time']*1000:.3f} ms/sample")
        
        print(f"CNN Inference Timing (avg of {TIMING_CONFIG['timing_runs']} runs):")
        print(f"  Mean: {cnn_timing['mean_time']*1000:.3f} ms/sample")
        print(f"  Std:  {cnn_timing['std_time']*1000:.3f} ms/sample")
        
        # Update results with detailed timing
        svm_results.update(svm_timing)
        cnn_results.update(cnn_timing)
        
        # 6. Compare models and generate visualizations
        print("\n📈 Step 6: Generating comparison and visualizations...")
        evaluator = ModelEvaluator()
        comparison = evaluator.compare_models(svm_results, cnn_results, class_names)
        
        # 7. Save model information
        print("\n💾 Step 7: Saving model information...")
        
        # Create results directory if it doesn't exist
        results_dir = 'results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Save model configurations and results
        import json
        
        model_info = {
            'svm_model': svm_model.get_model_info(),
            'cnn_model': cnn_model.get_model_info(),
            'dataset_info': {
                'data_dir': args.data_dir,
                'num_classes': len(class_names),
                'class_names': class_names,
                'total_samples': len(X),
                'train_samples': len(X_train_svm),
                'test_samples': len(X_test_svm)
            },
            'results_summary': {
                'svm_accuracy': svm_results['accuracy'],
                'cnn_accuracy': cnn_results['accuracy'],
                'svm_training_time': svm_results['training_time'],
                'cnn_training_time': cnn_results['training_time'],
                'svm_inference_time': svm_results['inference_time_per_sample'],
                'cnn_inference_time': cnn_results['inference_time_per_sample']
            }
        }
        
        with open(os.path.join(results_dir, 'model_comparison.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ Model information saved to {results_dir}/model_comparison.json")
        
        # 8. Final summary
        print("\n🏁 EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("📁 Results saved in 'results' directory:")
        print("  - model_comparison.json (detailed results)")
        print("  - accuracy_comparison.png")
        print("  - training_time_comparison.png")
        print("  - inference_time_comparison.png")
        print("  - confusion_matrices.png")
        print("  - detailed_metrics.png")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)