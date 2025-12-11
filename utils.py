# utils.py
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import deque

class ValidationLossEarlyStopper:
    """Stops training when validation loss stops improving."""
    def __init__(self, patience=35, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop_triggered = False

    def __call__(self, val_loss):
        if self.early_stop_triggered: return True
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping on validation loss: No improvement for {self.patience} epochs.")
                self.early_stop_triggered = True
                return True
        return False
    
    def reset(self):
        self.counter, self.best_loss, self.early_stop_triggered = 0, float('inf'), False

class TrainingLossEarlyStopper:
    """
    Stops training when the relative difference of training loss is small
    for a consecutive number of epochs, as described in the paper.
    """
    def __init__(self, patience=35, rel_delta_threshold=1e-6):
        self.patience = patience
        self.rel_delta_threshold = rel_delta_threshold
        self.consecutive_count = 0
        self.previous_loss = None
        self.early_stop_triggered = False

    def __call__(self, current_loss):
        if self.early_stop_triggered: return True
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
            print(f"Early stopping on training loss: Relative difference < {self.rel_delta_threshold} "
                  f"for {self.patience} consecutive epochs.")
            self.early_stop_triggered = True
            return True
        return False
    
    def reset(self):
        self.consecutive_count, self.previous_loss, self.early_stop_triggered = 0, None, False

def format_time(seconds):
    """Helper function to format seconds into H:M:S string"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def save_training_graphs(history, model_name_str, output_path="training_graph.png"):
    """
    Generates and saves line graphs for training/validation loss and accuracy.
    """
    has_accuracy = 'train_acc' in history and history['train_acc']
    num_subplots = 2 if has_accuracy else 1
    fig, axs = plt.subplots(num_subplots, 1, figsize=(8, 5 * num_subplots), sharex=True)
    if num_subplots == 1: axs = [axs]
    fig.suptitle(f'{model_name_str} Training History', fontsize=16)

    # Plot Loss
    ax_loss = axs[0]
    ax_loss.plot(history['loss'], label='Training Loss', color='blue')
    if 'val_loss' in history and history['val_loss']:
        ax_loss.plot(history['val_loss'], label='Validation Loss', color='orange')
    ax_loss.set_title("Loss over Epochs")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, linestyle='--', alpha=0.6)

    # Plot Accuracy
    if has_accuracy:
        ax_acc = axs[1]
        ax_acc.plot(history['train_acc'], label='Training Accuracy', color='blue')
        if 'val_acc' in history and history['val_acc']:
            ax_acc.plot(history['val_acc'], label='Validation Accuracy', color='orange')
        ax_acc.set_title("Accuracy over Epochs")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.set_xlabel("Epoch")
        ax_acc.legend()
        ax_acc.grid(True, linestyle='--', alpha=0.6)
    else:
        ax_loss.set_xlabel("Epoch")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    try:
        plt.savefig(output_path, dpi=150)
        print(f"Training graph saved to {output_path}")
    except Exception as e:
        print(f"Error saving training graph: {e}")
    plt.close(fig)

def save_confusion_matrix(y_true, y_pred, class_names, output_path="confusion_matrix.png"):
    """
    Generates and saves a heatmap image of the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_title(f'Confusion Matrix', fontsize=16)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=150)
        print(f"Confusion matrix image saved to {output_path}")
    except Exception as e:
        print(f"Error saving confusion matrix: {e}")
    plt.close(fig)