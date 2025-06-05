# utils.py
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class TrainingLossEarlyStopper:
    def __init__(self, patience=35, rel_delta_threshold=1e-6):
        self.patience = patience
        self.rel_delta_threshold = rel_delta_threshold
        self.consecutive_count = 0
        self.previous_loss = None
        self.early_stop_triggered = False

    def __call__(self, current_loss):
        if self.early_stop_triggered: # If already stopped, stay stopped
            return True
            
        if self.previous_loss is None:
            self.previous_loss = current_loss
            return False

        if self.previous_loss == 0:
            relative_diff = 0.0 if current_loss == 0 else float('inf')
        else:
            relative_diff = abs(self.previous_loss - current_loss) / abs(self.previous_loss)

        if relative_diff < self.rel_delta_threshold:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 0

        self.previous_loss = current_loss

        if self.consecutive_count >= self.patience:
            print(f"Early stopping: Relative training loss difference < {self.rel_delta_threshold} "
                  f"for {self.patience} consecutive epochs.")
            self.early_stop_triggered = True
            return True
        return False
    
    def reset(self):
        self.consecutive_count = 0
        self.previous_loss = None
        self.early_stop_triggered = False


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


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
                         'Normal - Precision', 'Normal - Recall', 'Normal - F1-Score', 'Normal - Support',
                         'Abnormal - Precision', 'Abnormal - Recall', 'Abnormal - F1-Score', 'Abnormal - Support',
                         'Macro Avg - Precision', 'Macro Avg - Recall', 'Macro Avg - F1-Score', 'Macro Avg - Support',
                         'Weighted Avg - Precision', 'Weighted Avg - Recall', 'Weighted Avg - F1-Score', 'Weighted Avg - Support']
        metric_values = [
            f"{report_dict.get('accuracy', 0)*100:.2f}%",
            f"{report_dict.get('Normal (Class 0)', {}).get('precision', 0)*100:.2f}%", # Assuming target_names were used
            f"{report_dict.get('Normal (Class 0)', {}).get('recall', 0)*100:.2f}%",
            f"{report_dict.get('Normal (Class 0)', {}).get('f1-score', 0)*100:.2f}%",
            f"{report_dict.get('Normal (Class 0)', {}).get('support', 0)}",
            f"{report_dict.get('Abnormal (Class 1)', {}).get('precision', 0)*100:.2f}%",
            f"{report_dict.get('Abnormal (Class 1)', {}).get('recall', 0)*100:.2f}%",
            f"{report_dict.get('Abnormal (Class 1)', {}).get('f1-score', 0)*100:.2f}%",
            f"{report_dict.get('Abnormal (Class 1)', {}).get('support', 0)}",
            f"{report_dict.get('macro avg', {}).get('precision', 0)*100:.2f}%",
            f"{report_dict.get('macro avg', {}).get('recall', 0)*100:.2f}%",
            f"{report_dict.get('macro avg', {}).get('f1-score', 0)*100:.2f}%",
            f"{report_dict.get('macro avg', {}).get('support', 0)}",
            f"{report_dict.get('weighted avg', {}).get('precision', 0)*100:.2f}%",
            f"{report_dict.get('weighted avg', {}).get('recall', 0)*100:.2f}%",
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
                    f'{name} - Precision', f'{name} - Recall', f'{name} - F1-Score', f'{name} - Support'
                ])
                metric_values.extend([
                    f"{report_dict.get(class_key, {}).get('precision', 0)*100:.2f}%",
                    f"{report_dict.get(class_key, {}).get('recall', 0)*100:.2f}%",
                    f"{report_dict.get(class_key, {}).get('f1-score', 0)*100:.2f}%",
                    f"{report_dict.get(class_key, {}).get('support', 0)}"
                ])

        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in report_dict:
                metric_labels.extend([
                    f'{avg_type.title()} - Precision', f'{avg_type.title()} - Recall', 
                    f'{avg_type.title()} - F1-Score', f'{avg_type.title()} - Support'
                ])
                metric_values.extend([
                    f"{report_dict[avg_type].get('precision', 0)*100:.2f}%",
                    f"{report_dict[avg_type].get('recall', 0)*100:.2f}%",
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