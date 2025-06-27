# utils.py
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

class ValidationLossEarlyStopper:
    def __init__(self, patience=35, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop_triggered = False

    def __call__(self, val_loss):
        if self.early_stop_triggered:
            return True
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping: Validation loss did not improve for {self.patience} consecutive epochs.")
                self.early_stop_triggered = True
                return True
        return False
    
    def reset(self):
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop_triggered = False


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def display_data_samples(data_tensor, description, preprocessor, original_feature_names, n_samples=10):
    """
    Reverses the preprocessing for a sample of data and displays it in a
    human-readable DataFrame format.
    """
    print(f"\n--- Displaying {n_samples} Human-Readable Samples for: {description} ---")
    if data_tensor is None or len(data_tensor) == 0:
        print("  No data to display.")
        print("-" * (len(description) + 40))
        return
    
    num_samples_to_show = min(n_samples, len(data_tensor))
    sample_tensor = data_tensor[:num_samples_to_show]
    sample_np = sample_tensor.cpu().numpy()

    try:
        # Use the preprocessor to reverse the scaling and one-hot encoding
        inversed_data = preprocessor.inverse_transform(sample_np)
        
        # Create a pandas DataFrame with the original feature names for nice printing
        df = pd.DataFrame(inversed_data, columns=original_feature_names)

        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 120):
            print(df.round(2)) # Round numerical values for cleaner display

    except Exception as e:
        print(f"  Could not perform inverse transform. Error: {e}")
        print("  Displaying raw processed data instead (first 15 features):")
        num_features_to_show = min(15, sample_np.shape[1])
        df = pd.DataFrame(sample_np[:, :num_features_to_show])
        df.columns = [f'F{i+1}' for i in range(num_features_to_show)]
        print(df.round(4))

    print("-" * (len(description) + 40))

def save_results_table(report_dict, timings_data, output_path="results_summary.png", classifier_mode="binary", class_names=None):
    """
    Generates and saves a table image of metrics and timings.
    Displays Recall before Precision.
    """
    # <<< MODIFIED: Reordered Recall and Precision >>>
    metric_labels, metric_values = [], []
    if not report_dict:
        metric_labels.append("Metrics"); metric_values.append("N/A")
    else:
        metric_labels.append('Overall Accuracy')
        metric_values.append(f"{report_dict.get('accuracy', 0)*100:.2f}%")
        
        # Determine which classes to report on
        keys_to_report = class_names if classifier_mode == "multiclass" and class_names is not None else ['Normal (Class 0)', 'Abnormal (Class 1)']
        
        for class_key in keys_to_report:
            if class_key in report_dict:
                metric_labels.extend([f'{class_key} - Recall', f'{class_key} - Precision', f'{class_key} - F1-Score', f'{class_key} - Support'])
                metric_values.extend([
                    f"{report_dict[class_key].get('recall', 0)*100:.2f}%",
                    f"{report_dict[class_key].get('precision', 0)*100:.2f}%",
                    f"{report_dict[class_key].get('f1-score', 0)*100:.2f}%",
                    f"{report_dict[class_key].get('support', 0)}"
                ])

        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in report_dict:
                metric_labels.extend([f'{avg_type.title()} - Recall', f'{avg_type.title()} - Precision', f'{avg_type.title()} - F1-Score', f'{avg_type.title()} - Support'])
                metric_values.extend([
                    f"{report_dict[avg_type].get('recall', 0)*100:.2f}%",
                    f"{report_dict[avg_type].get('precision', 0)*100:.2f}%",
                    f"{report_dict[avg_type].get('f1-score', 0)*100:.2f}%",
                    f"{report_dict[avg_type].get('support', 0)}"
                ])

def save_results_table(report_dict, timings_data, output_path="results_summary.png", classifier_mode="binary", class_names=None):
    """
    Generates and saves a table image of the classification metrics and step timings.
    Adjusts metrics displayed based on classifier_mode.
    """
    metric_labels = []
    metric_values = []

    if not report_dict: # Handle empty report_dict
        print("Warning: No metrics data to generate table.")
        metric_labels.append("Metrics")
        metric_values.append("N/A")
    elif classifier_mode == "binary":
        metric_labels = ['Overall Accuracy', 
                         'Normal - Recall', 'Normal - Precision', 'Normal - F1-Score', 'Normal - Support',
                         'Abnormal - Recall', 'Abnormal - Precision', 'Abnormal - F1-Score', 'Abnormal - Support',
                         'Macro Avg - Recall', 'Macro Avg - Precision', 'Macro Avg - F1-Score', 'Macro Avg - Support',
                         'Weighted Avg - Recall', 'Weighted Avg - Precision', 'Weighted Avg - F1-Score', 'Weighted Avg - Support']
        metric_values = [
            f"{report_dict.get('accuracy', 0)*100:.2f}%",
            f"{report_dict.get('Normal (Class 0)', {}).get('recall', 0)*100:.2f}%",
            f"{report_dict.get('Normal (Class 0)', {}).get('precision', 0)*100:.2f}%", # Assuming target_names were used
            f"{report_dict.get('Normal (Class 0)', {}).get('f1-score', 0)*100:.2f}%",
            f"{report_dict.get('Normal (Class 0)', {}).get('support', 0)}",
            f"{report_dict.get('Abnormal (Class 1)', {}).get('recall', 0)*100:.2f}%",
            f"{report_dict.get('Abnormal (Class 1)', {}).get('precision', 0)*100:.2f}%",
            f"{report_dict.get('Abnormal (Class 1)', {}).get('f1-score', 0)*100:.2f}%",
            f"{report_dict.get('Abnormal (Class 1)', {}).get('support', 0)}",
            f"{report_dict.get('macro avg', {}).get('recall', 0)*100:.2f}%",
            f"{report_dict.get('macro avg', {}).get('precision', 0)*100:.2f}%",
            f"{report_dict.get('macro avg', {}).get('f1-score', 0)*100:.2f}%",
            f"{report_dict.get('macro avg', {}).get('support', 0)}",
            f"{report_dict.get('weighted avg', {}).get('recall', 0)*100:.2f}%",
            f"{report_dict.get('weighted avg', {}).get('precision', 0)*100:.2f}%",
            f"{report_dict.get('weighted avg', {}).get('f1-score', 0)*100:.2f}%",
            f"{report_dict.get('weighted avg', {}).get('support', 0)}"
        ]
    elif classifier_mode == "multiclass":
        metric_labels.append('Overall Accuracy')
        metric_values.append(f"{report_dict.get('accuracy', 0)*100:.2f}%")
        
        if class_names: # Add per-class if class_names are provided
            for i, name in enumerate(class_names):
                # report_dict uses string keys for classes from target_names
                class_key = name 
                metric_labels.extend([
                    f'{name} - Recall', f'{name} - Precision', f'{name} - F1-Score', f'{name} - Support'
                ])
                metric_values.extend([
                    f"{report_dict.get(class_key, {}).get('recall', 0)*100:.2f}%",
                    f"{report_dict.get(class_key, {}).get('precision', 0)*100:.2f}%",
                    f"{report_dict.get(class_key, {}).get('f1-score', 0)*100:.2f}%",
                    f"{report_dict.get(class_key, {}).get('support', 0)}"
                ])

        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in report_dict:
                metric_labels.extend([
                    f'{avg_type.title()} - Recall', f'{avg_type.title()} - Precision', 
                    f'{avg_type.title()} - F1-Score', f'{avg_type.title()} - Support'
                ])
                metric_values.extend([
                    f"{report_dict[avg_type].get('recall', 0)*100:.2f}%",
                    f"{report_dict[avg_type].get('precision', 0)*100:.2f}%",
                    f"{report_dict[avg_type].get('f1-score', 0)*100:.2f}%",
                    f"{report_dict[avg_type].get('support', 0)}"
                ])
    
    metrics_table_data = [[label, value] for label, value in zip(metric_labels, metric_values)]
    timings_table_data = [[step, duration_str] for step, duration_str in timings_data.items()]

    num_tables = 1 if not timings_data else 2
    fig_height = 6 * len(metric_labels) / 15 + (4 if timings_data else 0) # Dynamic height
    fig_height = max(8, fig_height) # Minimum height

    fig, axs = plt.subplots(num_tables, 1, figsize=(10, fig_height))
    fig.patch.set_facecolor('white')

    current_ax_idx = 0
    if metrics_table_data:
        ax_metrics = axs[0] if num_tables > 1 else axs
        ax_metrics.axis('tight'); ax_metrics.axis('off')
        ax_metrics.set_title("Final Classification Metrics", fontsize=14, loc='center', pad=20)
        metrics_mpl_table = ax_metrics.table(cellText=metrics_table_data,
                                         colLabels=["Metric", "Value"],
                                         loc='center', cellLoc='left', colWidths=[0.6, 0.3])
        metrics_mpl_table.auto_set_font_size(False); metrics_mpl_table.set_fontsize(9)
        metrics_mpl_table.scale(1.1, 1.1)
        current_ax_idx +=1

    if timings_data:
        ax_timings = axs[current_ax_idx] if num_tables > 1 else axs # Handle if only one table
        ax_timings.axis('tight'); ax_timings.axis('off')
        ax_timings.set_title("Step Timings", fontsize=14, loc='center', pad=20)
        timings_mpl_table = ax_timings.table(cellText=timings_table_data,
                                         colLabels=["Step", "Duration (H:M:S)"],
                                         loc='center', cellLoc='left', colWidths=[0.6, 0.3])
        timings_mpl_table.auto_set_font_size(False); timings_mpl_table.set_fontsize(9)
        timings_mpl_table.scale(1.1, 1.1)

    plt.tight_layout(pad=3.0)
    try:
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        print(f"Results table saved to {output_path}")
    except Exception as e:
        print(f"Error saving results table: {e}")
    plt.close(fig)