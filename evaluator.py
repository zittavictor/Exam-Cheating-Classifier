"""
Evaluation and Visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time
import os
import warnings
warnings.filterwarnings('ignore')

from config import EVALUATION_CONFIG, TIMING_CONFIG

class ModelEvaluator:
    def __init__(self, config=None):
        self.config = config or EVALUATION_CONFIG
        self.timing_config = TIMING_CONFIG
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create results directory
        self.results_dir = self.config['plot_dir']
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def compare_models(self, svm_results, cnn_results, class_names):
        """Compare two models and generate comprehensive evaluation"""
        print("Comparing model performance...")
        
        comparison = {
            'models': {
                'SVM + HOG': svm_results,
                'CNN': cnn_results
            },
            'class_names': class_names,
            'comparison_metrics': self._calculate_comparison_metrics(svm_results, cnn_results)
        }
        
        # Generate visualizations
        self._plot_accuracy_comparison(comparison)
        self._plot_training_time_comparison(comparison)
        self._plot_inference_time_comparison(comparison)
        self._plot_confusion_matrices(comparison)
        self._plot_detailed_metrics(comparison)
        
        # Print summary
        self._print_comparison_summary(comparison)
        
        return comparison
    
    def _calculate_comparison_metrics(self, svm_results, cnn_results):
        """Calculate comparative metrics"""
        metrics = {
            'accuracy_difference': cnn_results['accuracy'] - svm_results['accuracy'],
            'training_time_ratio': cnn_results['training_time'] / svm_results['training_time'],
            'inference_time_ratio': cnn_results['inference_time_per_sample'] / svm_results['inference_time_per_sample'],
            'svm_faster_training': svm_results['training_time'] < cnn_results['training_time'],
            'svm_faster_inference': svm_results['inference_time_per_sample'] < cnn_results['inference_time_per_sample'],
            'cnn_more_accurate': cnn_results['accuracy'] > svm_results['accuracy']
        }
        
        return metrics
    
    def _plot_accuracy_comparison(self, comparison):
        """Plot accuracy comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Overall accuracy comparison
        models = list(comparison['models'].keys())
        accuracies = [comparison['models'][model]['accuracy'] for model in models]
        
        colors = ['#3498db', '#e74c3c']
        bars = ax1.bar(models, accuracies, color=colors, alpha=0.7)
        ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Per-class accuracy comparison
        class_names = comparison['class_names']
        svm_report = comparison['models']['SVM + HOG']['classification_report']
        cnn_report = comparison['models']['CNN']['classification_report']
        
        x = np.arange(len(class_names))
        width = 0.35
        
        svm_class_acc = [svm_report[class_name]['precision'] for class_name in class_names]
        cnn_class_acc = [cnn_report[class_name]['precision'] for class_name in class_names]
        
        ax2.bar(x - width/2, svm_class_acc, width, label='SVM + HOG', color='#3498db', alpha=0.7)
        ax2.bar(x + width/2, cnn_class_acc, width, label='CNN', color='#e74c3c', alpha=0.7)
        
        ax2.set_title('Per-Class Precision Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Precision')
        ax2.set_xlabel('Classes')
        ax2.set_xticks(x)
        ax2.set_xticklabels(class_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if self.config['save_plots']:
            plt.savefig(os.path.join(self.results_dir, 'accuracy_comparison.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_training_time_comparison(self, comparison):
        """Plot training time comparison"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        models = list(comparison['models'].keys())
        training_times = [comparison['models'][model]['training_time'] for model in models]
        
        colors = ['#3498db', '#e74c3c']
        bars = ax.bar(models, training_times, color=colors, alpha=0.7)
        
        ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Training Time (seconds)')
        
        # Add value labels on bars
        for bar, time_val in zip(bars, training_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(training_times) * 0.01,
                    f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if self.config['save_plots']:
            plt.savefig(os.path.join(self.results_dir, 'training_time_comparison.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_inference_time_comparison(self, comparison):
        """Plot inference time comparison"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        models = list(comparison['models'].keys())
        inference_times = [comparison['models'][model]['inference_time_per_sample'] * 1000 
                         for model in models]  # Convert to milliseconds
        
        colors = ['#3498db', '#e74c3c']
        bars = ax.bar(models, inference_times, color=colors, alpha=0.7)
        
        ax.set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Inference Time per Sample (ms)')
        
        # Add value labels on bars
        for bar, time_val in zip(bars, inference_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(inference_times) * 0.01,
                    f'{time_val:.3f}ms', ha='center', va='bottom', fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if self.config['save_plots']:
            plt.savefig(os.path.join(self.results_dir, 'inference_time_comparison.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_confusion_matrices(self, comparison):
        """Plot confusion matrices for both models"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        class_names = comparison['class_names']
        
        for idx, (model_name, results) in enumerate(comparison['models'].items()):
            # Get true labels (assuming they're the same for both models)
            if model_name == 'CNN':
                # For CNN, we need to handle categorical labels
                y_true = results.get('y_true', None)
                if y_true is None:
                    continue
            else:
                y_true = results.get('y_true', None)
                if y_true is None:
                    continue
            
            y_pred = results['predictions']
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[idx])
            axes[idx].set_title(f'{model_name} Confusion Matrix', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        if self.config['save_plots']:
            plt.savefig(os.path.join(self.results_dir, 'confusion_matrices.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_detailed_metrics(self, comparison):
        """Plot detailed metrics comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        class_names = comparison['class_names']
        svm_report = comparison['models']['SVM + HOG']['classification_report']
        cnn_report = comparison['models']['CNN']['classification_report']
        
        metrics = ['precision', 'recall', 'f1-score']
        
        for idx, metric in enumerate(metrics):
            row = idx // 2
            col = idx % 2
            
            x = np.arange(len(class_names))
            width = 0.35
            
            svm_values = [svm_report[class_name][metric] for class_name in class_names]
            cnn_values = [cnn_report[class_name][metric] for class_name in class_names]
            
            axes[row, col].bar(x - width/2, svm_values, width, label='SVM + HOG', 
                             color='#3498db', alpha=0.7)
            axes[row, col].bar(x + width/2, cnn_values, width, label='CNN', 
                             color='#e74c3c', alpha=0.7)
            
            axes[row, col].set_title(f'{metric.capitalize()} Comparison', fontsize=12, fontweight='bold')
            axes[row, col].set_ylabel(metric.capitalize())
            axes[row, col].set_xlabel('Classes')
            axes[row, col].set_xticks(x)
            axes[row, col].set_xticklabels(class_names, rotation=45, ha='right')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        # Overall metrics comparison
        overall_metrics = ['macro avg', 'weighted avg']
        
        # Get values for macro and weighted averages
        x = np.arange(len(overall_metrics))
        width = 0.35
        
        svm_vals = [svm_report[om]['precision'] for om in overall_metrics if om in svm_report]
        cnn_vals = [cnn_report[om]['precision'] for om in overall_metrics if om in cnn_report]
        
        if svm_vals and cnn_vals:
            axes[1, 1].bar(x - width/2, svm_vals, width, label='SVM + HOG', 
                          color='#3498db', alpha=0.7)
            axes[1, 1].bar(x + width/2, cnn_vals, width, label='CNN', 
                          color='#e74c3c', alpha=0.7)
        
        axes[1, 1].set_title('Overall Metrics Comparison', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_xlabel('Metric Type')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(overall_metrics, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if self.config['save_plots']:
            plt.savefig(os.path.join(self.results_dir, 'detailed_metrics.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def _print_comparison_summary(self, comparison):
        """Print comprehensive comparison summary"""
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON SUMMARY")
        print("="*80)
        
        svm_results = comparison['models']['SVM + HOG']
        cnn_results = comparison['models']['CNN']
        metrics = comparison['comparison_metrics']
        
        print(f"\n📊 ACCURACY COMPARISON:")
        print(f"  SVM + HOG: {svm_results['accuracy']:.4f}")
        print(f"  CNN:       {cnn_results['accuracy']:.4f}")
        print(f"  Difference: {metrics['accuracy_difference']:.4f} {'(CNN better)' if metrics['accuracy_difference'] > 0 else '(SVM better)'}")
        
        print(f"\n⏱️  TRAINING TIME COMPARISON:")
        print(f"  SVM + HOG: {svm_results['training_time']:.2f} seconds")
        print(f"  CNN:       {cnn_results['training_time']:.2f} seconds")
        print(f"  Ratio:     {metrics['training_time_ratio']:.2f}x {'(CNN slower)' if metrics['training_time_ratio'] > 1 else '(CNN faster)'}")
        
        print(f"\n🚀 INFERENCE TIME COMPARISON:")
        print(f"  SVM + HOG: {svm_results['inference_time_per_sample']*1000:.3f} ms/sample")
        print(f"  CNN:       {cnn_results['inference_time_per_sample']*1000:.3f} ms/sample")
        print(f"  Ratio:     {metrics['inference_time_ratio']:.2f}x {'(CNN slower)' if metrics['inference_time_ratio'] > 1 else '(CNN faster)'}")
        
        print(f"\n🏆 RECOMMENDATIONS:")
        if metrics['cnn_more_accurate']:
            print("  • CNN achieves higher accuracy")
        else:
            print("  • SVM + HOG achieves higher accuracy")
        
        if metrics['svm_faster_training']:
            print("  • SVM + HOG trains faster")
        else:
            print("  • CNN trains faster")
        
        if metrics['svm_faster_inference']:
            print("  • SVM + HOG has faster inference")
        else:
            print("  • CNN has faster inference")
        
        # Overall recommendation
        print(f"\n💡 OVERALL RECOMMENDATION:")
        if metrics['cnn_more_accurate'] and not metrics['svm_faster_inference']:
            print("  • Use CNN for better accuracy and acceptable inference time")
        elif metrics['svm_faster_inference'] and not metrics['cnn_more_accurate']:
            print("  • Use SVM + HOG for faster inference and comparable accuracy")
        else:
            print("  • Choice depends on your priority: accuracy vs. speed")
        
        print("="*80)