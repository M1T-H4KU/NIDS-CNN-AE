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


def save_results_table(metrics_data, timings_data, output_path="results_summary.png"):
    metric_labels = ['Overall Accuracy', 
                     'Normal - Precision', 'Normal - Recall', 'Normal - F1-Score', 'Normal - Support',
                     'Abnormal - Precision', 'Abnormal - Recall', 'Abnormal - F1-Score', 'Abnormal - Support',
                     'Macro Avg - Precision', 'Macro Avg - Recall', 'Macro Avg - F1-Score', 'Macro Avg - Support',
                     'Weighted Avg - Precision', 'Weighted Avg - Recall', 'Weighted Avg - F1-Score', 'Weighted Avg - Support']
    
    metric_values = [
        f"{metrics_data.get('accuracy', 0)*100:.2f}%",
        f"{metrics_data.get('Normal (Class 0)', {}).get('precision', 0)*100:.2f}%",
        f"{metrics_data.get('Normal (Class 0)', {}).get('recall', 0)*100:.2f}%",
        f"{metrics_data.get('Normal (Class 0)', {}).get('f1-score', 0)*100:.2f}%",
        f"{metrics_data.get('Normal (Class 0)', {}).get('support', 0)}",
        f"{metrics_data.get('Abnormal (Class 1)', {}).get('precision', 0)*100:.2f}%",
        f"{metrics_data.get('Abnormal (Class 1)', {}).get('recall', 0)*100:.2f}%",
        f"{metrics_data.get('Abnormal (Class 1)', {}).get('f1-score', 0)*100:.2f}%",
        f"{metrics_data.get('Abnormal (Class 1)', {}).get('support', 0)}",
        f"{metrics_data.get('macro avg', {}).get('precision', 0)*100:.2f}%",
        f"{metrics_data.get('macro avg', {}).get('recall', 0)*100:.2f}%",
        f"{metrics_data.get('macro avg', {}).get('f1-score', 0)*100:.2f}%",
        f"{metrics_data.get('macro avg', {}).get('support', 0)}",
        f"{metrics_data.get('weighted avg', {}).get('precision', 0)*100:.2f}%",
        f"{metrics_data.get('weighted avg', {}).get('recall', 0)*100:.2f}%",
        f"{metrics_data.get('weighted avg', {}).get('f1-score', 0)*100:.2f}%",
        f"{metrics_data.get('weighted avg', {}).get('support', 0)}"
    ]
    
    metrics_table_data = [[label, value] for label, value in zip(metric_labels, metric_values)]
    timings_table_data = [[step, duration_str] for step, duration_str in timings_data.items()]

    fig, axs = plt.subplots(2, 1, figsize=(10, 10)) # Increased height a bit for more metric rows
    fig.patch.set_facecolor('white') # Set background to white for saved image

    axs[0].axis('tight')
    axs[0].axis('off')
    axs[0].set_title("Final Classification Metrics", fontsize=14, loc='center', pad=20)
    metrics_mpl_table = axs[0].table(cellText=metrics_table_data,
                                     colLabels=["Metric", "Value"],
                                     loc='center', cellLoc='left', colWidths=[0.6, 0.3])
    metrics_mpl_table.auto_set_font_size(False)
    metrics_mpl_table.set_fontsize(9) # Adjusted fontsize
    metrics_mpl_table.scale(1.1, 1.1)

    axs[1].axis('tight')
    axs[1].axis('off')
    axs[1].set_title("Step Timings", fontsize=14, loc='center', pad=20)
    timings_mpl_table = axs[1].table(cellText=timings_table_data,
                                     colLabels=["Step", "Duration (H:M:S)"],
                                     loc='center', cellLoc='left', colWidths=[0.6, 0.3])
    timings_mpl_table.auto_set_font_size(False)
    timings_mpl_table.set_fontsize(9)
    timings_mpl_table.scale(1.1, 1.1)

    plt.tight_layout(pad=3.0)
    try:
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        print(f"Results table saved to {output_path}")
    except Exception as e:
        print(f"Error saving results table: {e}")
    plt.close(fig)