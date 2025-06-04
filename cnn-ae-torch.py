import numpy as np
import pandas as pd
import os
import time # For time recording
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split # Not strictly needed by current flow but good for general use

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report # classification_report is key
import matplotlib.pyplot as plt # For generating table image

def format_time(seconds):
    """Helper function to format seconds into H:M:S string"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# --- 0. Configuration & Device Setup ---
# These should be adjusted based on your actual NSL-KDD preprocessing
NUM_FEATURES_PREPROCESSED = 122 # Placeholder, will be updated by load_data
LATENT_DIM = 50
AE_HIDDEN_DIM_1 = 80
AE_EPOCHS = 300 # As per paper for AE
AE_BATCH_SIZE = 64
# AE_RECONSTRUCTION_TARGET = 0.97 # For reference

CNN_FILTERS = 32
CNN_KERNEL_SIZE = 5
CNN_POOL_SIZE = 3 # Kernel size for MaxPool1d
CNN_POOL_STRIDE = 3 # Stride for MaxPool1d
CNN_FC_NEURONS = 16
CNN_EPOCHS = 50 # Placeholder for CNN
CNN_BATCH_SIZE = 64

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

EXTRACTED_FEATURES_FILE = "ae_extracted_features.pt" # .pt for PyTorch tensors


# --- 1. NSL-KDD Data Loading and Preprocessing ---
# (This part is largely the same, just ensure output is suitable for PyTorch tensors)
def load_and_preprocess_nsl_kdd(
    base_path="datasets/nsl-kdd/",
    train_filename="KDDTrain+.txt",
    test_filename="KDDTest+.txt"
):
    global NUM_FEATURES_PREPROCESSED # To update the global variable
    print(f"Loading and preprocessing NSL-KDD data from {base_path}...")
    train_path = os.path.join(base_path, train_filename)
    test_path = os.path.join(base_path, test_filename)

    column_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
        'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
        'attack_type', 'difficulty_level'
    ]
    try:
        df_train = pd.read_csv(train_path, header=None, names=column_names)
        df_test = pd.read_csv(test_path, header=None, names=column_names)
    except FileNotFoundError:
        print(f"Error: Ensure '{train_filename}' and '{test_filename}' exist in '{base_path}'")
        raise

    X_train_raw = df_train.iloc[:, :-2]
    y_train_labels_raw = df_train['attack_type']
    X_test_raw = df_test.iloc[:, :-2]
    y_test_labels_raw = df_test['attack_type']

    y_train = y_train_labels_raw.apply(lambda x: 0 if x == 'normal' else 1).values
    y_test = y_test_labels_raw.apply(lambda x: 0 if x == 'normal' else 1).values

    categorical_features_indices = [1, 2, 3] # protocol_type, service, flag
    numerical_features_indices = [i for i in range(X_train_raw.shape[1]) if i not in categorical_features_indices]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('scaler', MinMaxScaler())]), numerical_features_indices),
            ('cat', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features_indices)
        ],
        remainder='passthrough'
    )

    X_train_processed = preprocessor.fit_transform(X_train_raw).astype(np.float32)
    X_test_processed = preprocessor.transform(X_test_raw).astype(np.float32)

    NUM_FEATURES_PREPROCESSED = X_train_processed.shape[1]
    print(f"Data preprocessed. Number of features: {NUM_FEATURES_PREPROCESSED}")
    print(f"X_train_processed shape: {X_train_processed.shape}, y_train shape: {y_train.shape}")
    print(f"X_test_processed shape: {X_test_processed.shape}, y_test shape: {y_test.shape}")

    # Convert to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train_processed)
    y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1) # For BCEWithLogitsLoss
    X_test_tensor = torch.from_numpy(X_test_processed)
    y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(1)   # For BCEWithLogitsLoss

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

# --- 2. PyTorch Dataset Class ---
class NSLKDDDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        if self.labels is not None:
            y = self.labels[idx]
            return x, y
        return x

# --- 3. Autoencoder (AE) Model ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim_1):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.encoder_bn1 = nn.BatchNorm1d(hidden_dim_1)
        self.encoder_relu1 = nn.ReLU()
        self.encoder_fc2 = nn.Linear(hidden_dim_1, latent_dim)
        self.encoder_bn2 = nn.BatchNorm1d(latent_dim)
        self.encoder_relu2 = nn.ReLU() # Latent space output also gets BN and ReLU as per paper

        # Decoder
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim_1)
        self.decoder_bn1 = nn.BatchNorm1d(hidden_dim_1)
        self.decoder_relu1 = nn.ReLU()
        self.decoder_fc2 = nn.Linear(hidden_dim_1, input_dim)
        self.decoder_sigmoid = nn.Sigmoid() # Assuming inputs are scaled [0,1]

    def encode(self, x):
        x = self.encoder_fc1(x)
        x = self.encoder_bn1(x)
        x = self.encoder_relu1(x)
        x = self.encoder_fc2(x)
        x = self.encoder_bn2(x)
        x = self.encoder_relu2(x)
        return x

    def decode(self, x):
        x = self.decoder_fc1(x)
        x = self.decoder_bn1(x)
        x = self.decoder_relu1(x)
        x = self.decoder_fc2(x)
        x = self.decoder_sigmoid(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

# --- 4. CNN Classifier Model ---
class CNNClassifier(nn.Module):
    def __init__(self, input_dim_cnn, num_classes=1): # input_dim_cnn is LATENT_DIM
        super(CNNClassifier, self).__init__()
        # Conv1D expects (batch_size, in_channels, sequence_length)
        # Our AE output is (batch_size, LATENT_DIM), so in_channels=1, sequence_length=LATENT_DIM
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=CNN_FILTERS,
                               kernel_size=CNN_KERNEL_SIZE, padding=(CNN_KERNEL_SIZE - 1) // 2) # 'same' padding
        self.bn1 = nn.BatchNorm1d(CNN_FILTERS)
        self.relu1 = nn.ReLU()
        # Paper: "max-pooling layer with windows of size 3 to the first convolutional layer"
        self.pool1 = nn.MaxPool1d(kernel_size=CNN_POOL_SIZE, stride=CNN_POOL_STRIDE) # Stride not specified, common to use kernel_size

        self.conv2 = nn.Conv1d(in_channels=CNN_FILTERS, out_channels=CNN_FILTERS,
                               kernel_size=CNN_KERNEL_SIZE, padding=(CNN_KERNEL_SIZE - 1) // 2) # 'same' padding
        self.bn2 = nn.BatchNorm1d(CNN_FILTERS)
        self.relu2 = nn.ReLU()

        self.flatten = nn.Flatten()

        # Need to calculate the input size for the first FC layer
        # This can be done by a dummy forward pass or manual calculation
        # Let's assume LATENT_DIM = 50, CNN_POOL_SIZE = 3, CNN_POOL_STRIDE = 3
        # After conv1 (padding='same'): (batch, 32, 50)
        # After pool1 (kernel=3, stride=3): L_out = floor((L_in - kernel_size)/stride + 1)
        # L_out = floor((50 - 3)/3 + 1) = floor(47/3 + 1) = floor(15.66 + 1) = 16
        # So after pool1: (batch, 32, 16)
        # After conv2 (padding='same'): (batch, 32, 16)
        # After flatten: batch_size * (32 * 16) = batch_size * 512
        # This calculation needs to be robust.
        # A common way is to do a dummy forward pass of the conv part.
        # For now, I'll compute it based on the example.
        # (L_in - K_pool + 2*P_pool)/S_pool + 1. If P_pool=0
        # L_pool1_out = (LATENT_DIM - CNN_POOL_SIZE) // CNN_POOL_STRIDE + 1
        # flattened_size = CNN_FILTERS * L_pool1_out # This is if conv2 doesn't change length
        
        # Let's define a helper to calculate conv output size
        def _get_conv_output_size(input_size, kernel_size, stride, padding=0, dilation=1):
            return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        # Size after Conv1 (padding='same' keeps it LATENT_DIM if stride=1)
        conv1_out_len = LATENT_DIM
        # Size after Pool1
        pool1_out_len = _get_conv_output_size(conv1_out_len, CNN_POOL_SIZE, CNN_POOL_STRIDE)
        # Size after Conv2 (padding='same' keeps it pool1_out_len if stride=1)
        conv2_out_len = pool1_out_len
        
        self.flattened_features = CNN_FILTERS * conv2_out_len
        
        self.fc1 = nn.Linear(self.flattened_features, CNN_FC_NEURONS)
        self.relu3 = nn.ReLU()
        self.fc_output = nn.Linear(CNN_FC_NEURONS, num_classes)
        # For BCEWithLogitsLoss, no sigmoid here.

    def forward(self, x):
        # x shape: (batch_size, latent_dim)
        x = x.unsqueeze(1) # Reshape to (batch_size, 1, latent_dim) for Conv1D

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc_output(x)
        return x

# --- 5. Training and Evaluation Functions ---
def train_autoencoder(model, dataloader, epochs, learning_rate=1e-3):
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("\n--- Training Autoencoder ---")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_features in dataloader:
            batch_features = batch_features.to(DEVICE) # Dataloader returns a single item if labels=None
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_features) # Reconstruct input
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs -1:
            print(f"Epoch [{epoch+1}/{epochs}], AE Loss: {avg_epoch_loss:.6f}")
    print("Autoencoder training finished.")

def extract_ae_features(encoder_part, dataloader):
    encoder_part.to(DEVICE)
    encoder_part.eval()
    all_features = []
    with torch.no_grad():
        for batch_data in dataloader:
            batch_data = batch_data.to(DEVICE)
            encoded_features = encoder_part.encode(batch_data)
            all_features.append(encoded_features.cpu())
    return torch.cat(all_features, dim=0)

def train_classifier(model, train_loader, val_loader, epochs, learning_rate=1e-3, num_classes=1):
    model.to(DEVICE)
    if num_classes == 1: # Binary
        criterion = nn.BCEWithLogitsLoss()
    else: # Multi-class
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    target_names = ['Normal (Class 0)', 'Abnormal (Class 1)'] # For classification_report

    print("\n--- Training CNN Classifier ---")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(DEVICE), batch_labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if num_classes == 1:
                predicted = (torch.sigmoid(outputs) > 0.5).float()
            else:
                _, predicted = torch.max(outputs.data, 1)
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_labels).sum().item()
            
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy_epoch = 100 * correct_train / total_train

        # Validation
        model.eval()
        val_loss = 0
        all_val_preds_list = []
        all_val_labels_list = []
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(DEVICE), batch_labels.to(DEVICE)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                if num_classes == 1:
                    predicted_val = (torch.sigmoid(outputs) > 0.5).float()
                else:
                    _, predicted_val = torch.max(outputs.data, 1)
                all_val_preds_list.extend(predicted_val.cpu().numpy())
                all_val_labels_list.extend(batch_labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        all_val_labels_np = np.array(all_val_labels_list).squeeze()
        all_val_preds_np = np.array(all_val_preds_list).squeeze()

        # Use classification_report for detailed metrics
        # output_dict=True makes it easy to parse
        # zero_division=0 will return 0.0 for precision/recall/F1 if division by zero occurs
        report_dict = classification_report(all_val_labels_np, all_val_preds_np, 
                                            target_names=target_names, output_dict=True, zero_division=0)
        
        val_accuracy = report_dict['accuracy'] * 100 # Overall accuracy from the report

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs -1 :
            print(f"Epoch [{epoch+1}/{epochs}]:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy_epoch:.2f}%")
            print(f"  Val Loss  : {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            for class_name in target_names:
                if class_name in report_dict:
                    print(f"  {class_name}: "
                          f"P: {report_dict[class_name]['precision']*100:.2f}%, "
                          f"R: {report_dict[class_name]['recall']*100:.2f}%, "
                          f"F1: {report_dict[class_name]['f1-score']*100:.2f}%")
    print("CNN training finished.")
    return model

def save_results_table(metrics_data, timings_data, output_path="results_summary.png"):
    """
    Generates and saves a table image of the classification metrics and step timings.
    """
    # Prepare data for the metrics table
    metric_labels = ['Overall Accuracy', 
                     'Normal - Precision', 'Normal - Recall', 'Normal - F1-Score', 'Normal - Support',
                     'Abnormal - Precision', 'Abnormal - Recall', 'Abnormal - F1-Score', 'Abnormal - Support',
                     'Macro Avg - Precision', 'Macro Avg - Recall', 'Macro Avg - F1-Score',
                     'Weighted Avg - Precision', 'Weighted Avg - Recall', 'Weighted Avg - F1-Score']
    
    metric_values = [
        f"{metrics_data.get('accuracy', 0)*100:.2f}%",
        f"{metrics_data.get('Normal (Class 0)', {}).get('recall', 0)*100:.2f}%",
        f"{metrics_data.get('Normal (Class 0)', {}).get('precision', 0)*100:.2f}%",
        f"{metrics_data.get('Normal (Class 0)', {}).get('f1-score', 0)*100:.2f}%",
        f"{metrics_data.get('Normal (Class 0)', {}).get('support', 0)}",
        f"{metrics_data.get('Abnormal (Class 1)', {}).get('recall', 0)*100:.2f}%",
        f"{metrics_data.get('Abnormal (Class 1)', {}).get('precision', 0)*100:.2f}%",
        f"{metrics_data.get('Abnormal (Class 1)', {}).get('f1-score', 0)*100:.2f}%",
        f"{metrics_data.get('Abnormal (Class 1)', {}).get('support', 0)}",
        f"{metrics_data.get('macro avg', {}).get('precision', 0)*100:.2f}%",
        f"{metrics_data.get('macro avg', {}).get('recall', 0)*100:.2f}%",
        f"{metrics_data.get('macro avg', {}).get('f1-score', 0)*100:.2f}%",
        f"{metrics_data.get('weighted avg', {}).get('precision', 0)*100:.2f}%",
        f"{metrics_data.get('weighted avg', {}).get('recall', 0)*100:.2f}%",
        f"{metrics_data.get('weighted avg', {}).get('f1-score', 0)*100:.2f}%"
    ]
    
    metrics_table_data = [[label, value] for label, value in zip(metric_labels, metric_values)]

    # Prepare data for the timings table
    timings_table_data = [[step, duration_str] for step, duration_str in timings_data.items()]

    # Create figure and axes
    # Adjust subplot counts and figure size based on how many tables you want
    fig, axs = plt.subplots(2, 1, figsize=(10, 8)) # 2 rows, 1 column

    # --- Metrics Table ---
    axs[0].axis('tight')
    axs[0].axis('off')
    axs[0].set_title("Final Classification Metrics", fontsize=14, loc='center', pad=20)
    metrics_mpl_table = axs[0].table(cellText=metrics_table_data,
                                     colLabels=["Metric", "Value"],
                                     loc='center',
                                     cellLoc='left',
                                     colWidths=[0.6, 0.3]) # Adjust colWidths as needed
    metrics_mpl_table.auto_set_font_size(False)
    metrics_mpl_table.set_fontsize(10)
    metrics_mpl_table.scale(1.1, 1.1) # Adjust scale for better fit

    plt.tight_layout(pad=3.0) # Add padding between subplots and title
    try:
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        print(f"Results table saved to {output_path}")
    except Exception as e:
        print(f"Error saving results table: {e}")
    plt.close(fig)
    

# --- 6. Main Workflow ---
if __name__ == '__main__':
    step_timings = {} # Dictionary to store timings

    # --- Step 1: Load and preprocess data (This always runs) ---
    print("\n--- Starting: Data Loading & Preprocessing ---")
    start_time_data_load = time.time() # Renamed start_time for clarity
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = load_and_preprocess_nsl_kdd()
    # NUM_FEATURES_PREPROCESSED is set globally by load_and_preprocess_nsl_kdd
    duration_data_load = time.time() - start_time_data_load
    step_timings['Data Loading & Preprocessing'] = format_time(duration_data_load)
    print(f"--- Finished: Data Loading & Preprocessing in {format_time(duration_data_load)} ---")

    # --- Step 2: Autoencoder Training and/or Feature Loading/Extraction ---
    X_train_features_ae = None
    X_test_features_ae = None
    actual_latent_dim_for_cnn = LATENT_DIM # Default to global config

    if os.path.exists(EXTRACTED_FEATURES_FILE):
        print(f"\n--- Loading pre-extracted AE features from {EXTRACTED_FEATURES_FILE} ---")
        start_time_load_feat = time.time()
        try:
            loaded_data = torch.load(EXTRACTED_FEATURES_FILE, map_location=DEVICE) # Load to current device
            X_train_features_ae = loaded_data['train_features'].to(DEVICE)
            X_test_features_ae = loaded_data['test_features'].to(DEVICE)
            # Retrieve dimensions used when features were saved for consistency
            # num_original_features_saved = loaded_data.get('num_original_features', NUM_FEATURES_PREPROCESSED)
            actual_latent_dim_for_cnn = loaded_data.get('latent_dim', X_train_features_ae.shape[1])
            
            # Optional: Sanity check if loaded dimensions match current config or data
            # if num_original_features_saved != NUM_FEATURES_PREPROCESSED:
            #     print(f"Warning: Loaded features were based on {num_original_features_saved} original features, current is {NUM_FEATURES_PREPROCESSED}")
            if actual_latent_dim_for_cnn != LATENT_DIM:
                 print(f"Warning: Loaded features have latent_dim {actual_latent_dim_for_cnn}, global LATENT_DIM is {LATENT_DIM}. Using {actual_latent_dim_for_cnn} for CNN.")

            duration_load_feat = time.time() - start_time_load_feat
            step_timings['AE Feature Loading'] = format_time(duration_load_feat)
            print(f"--- Finished: AE Feature Loading in {format_time(duration_load_feat)} ---")
            print(f"Loaded AE features. Train shape: {X_train_features_ae.shape}, Test shape: {X_test_features_ae.shape}")
            
            # If features are loaded, we skip AE training and direct extraction.
            step_timings['Autoencoder Training'] = "Skipped (loaded features)"
            step_timings['AE Feature Extraction'] = "Skipped (loaded features)"
        except Exception as e:
            print(f"Error loading features from {EXTRACTED_FEATURES_FILE}: {e}. Will proceed to train AE and extract features.")
            X_train_features_ae = None # Ensure it's None so the else block runs

    if X_train_features_ae is None or X_test_features_ae is None: # If loading failed or file didn't exist
        # --- Step 2.1: Build and Train Autoencoder (if features not loaded) ---
        print("\n--- Starting: Autoencoder Training ---")
        start_time_ae_train = time.time()
        autoencoder_model = Autoencoder(
            input_dim=NUM_FEATURES_PREPROCESSED, 
            latent_dim=LATENT_DIM, # Use global LATENT_DIM for AE definition
            hidden_dim_1=AE_HIDDEN_DIM_1
        ).to(DEVICE) # Move model to device
        print("\n--- Autoencoder Architecture ---")
        print(autoencoder_model)
        
        ae_train_dataset = NSLKDDDataset(X_train_tensor)
        ae_train_loader = DataLoader(ae_train_dataset, batch_size=AE_BATCH_SIZE, shuffle=True)
        train_autoencoder(autoencoder_model, ae_train_loader, epochs=AE_EPOCHS)
        duration_ae_train = time.time() - start_time_ae_train
        step_timings['Autoencoder Training'] = format_time(duration_ae_train)
        print(f"--- Finished: Autoencoder Training in {format_time(duration_ae_train)} ---")

        # --- Step 2.2: Extract Features using the trained Encoder part of the AE ---
        print("\n--- Starting: AE Feature Extraction ---")
        start_time_extract_feat = time.time()
        ae_train_dataset_for_extraction = NSLKDDDataset(X_train_tensor)
        ae_train_loader_for_extraction = DataLoader(ae_train_dataset_for_extraction, batch_size=AE_BATCH_SIZE, shuffle=False)
        ae_test_dataset_for_extraction = NSLKDDDataset(X_test_tensor)
        ae_test_loader_for_extraction = DataLoader(ae_test_dataset_for_extraction, batch_size=AE_BATCH_SIZE, shuffle=False)

        X_train_features_ae = extract_ae_features(autoencoder_model, ae_train_loader_for_extraction)
        X_test_features_ae = extract_ae_features(autoencoder_model, ae_test_loader_for_extraction)
        actual_latent_dim_for_cnn = X_train_features_ae.shape[1] # Set from actual extracted features
        
        duration_extract_feat = time.time() - start_time_extract_feat
        step_timings['AE Feature Extraction'] = format_time(duration_extract_feat)
        print(f"--- Finished: AE Feature Extraction in {format_time(duration_extract_feat)} ---")

        # Save the extracted features
        print(f"\n--- Saving AE features to {EXTRACTED_FEATURES_FILE} ---")
        try:
            torch.save({
                'train_features': X_train_features_ae.cpu(), # Save as CPU tensors
                'test_features': X_test_features_ae.cpu(),  # Save as CPU tensors
                'num_original_features': NUM_FEATURES_PREPROCESSED,
                'latent_dim': actual_latent_dim_for_cnn 
            }, EXTRACTED_FEATURES_FILE)
            print(f"--- Features saved. ---")
        except Exception as e:
            print(f"Error saving features to {EXTRACTED_FEATURES_FILE}: {e}")
    
    # Ensure features are on the correct device for CNN training
    if X_train_features_ae is not None:
        X_train_features_ae = X_train_features_ae.to(DEVICE)
    if X_test_features_ae is not None:
        X_test_features_ae = X_test_features_ae.to(DEVICE)

    if X_train_features_ae is None or X_test_features_ae is None:
        print("Critical Error: AE features are not available. Exiting.")
        exit()
        
    print(f"AE features ready for CNN. Train shape: {X_train_features_ae.shape}, Test shape: {X_test_features_ae.shape}")
    print(f"Using latent dimension for CNN: {actual_latent_dim_for_cnn}")


    # --- Step 3: Train CNN Classifier with AE features ---
    cnn_train_dataset = NSLKDDDataset(X_train_features_ae, y_train_tensor) # y_train_tensor from data loading
    cnn_train_loader = DataLoader(cnn_train_dataset, batch_size=CNN_BATCH_SIZE, shuffle=True)
    cnn_test_loader = DataLoader(NSLKDDDataset(X_test_features_ae, y_test_tensor), batch_size=CNN_BATCH_SIZE, shuffle=False)

    num_classes_for_cnn = 1 # Binary classification
    cnn_classifier_model = CNNClassifier(
        input_dim_cnn=actual_latent_dim_for_cnn, # Use the actual latent dim
        num_classes=num_classes_for_cnn
    ).to(DEVICE) # Move model to device
    
    print("\n--- CNN Classifier Architecture ---")
    print(cnn_classifier_model) # Will reflect actual_latent_dim_for_cnn in its flattened layer calculation

    print("\n--- Starting: CNN Classifier Training & Validation ---")
    start_time_cnn_train = time.time()
    trained_cnn_model = train_classifier(cnn_classifier_model, cnn_train_loader, cnn_test_loader, 
                                         epochs=CNN_EPOCHS, num_classes=num_classes_for_cnn)
    duration_cnn_train = time.time() - start_time_cnn_train
    step_timings['CNN Training & Validation'] = format_time(duration_cnn_train)
    print(f"--- Finished: CNN Training & Validation in {format_time(duration_cnn_train)} ---")

    # --- Step 4: Final Evaluation on Test Data ---
    print("\n--- Starting: Final CNN Evaluation on Test Data ---")
    start_time_final_eval = time.time()
    trained_cnn_model.eval() 
    
    all_final_preds_list = []
    all_final_labels_list = []
    
    with torch.no_grad():
        for features, labels in cnn_test_loader: # Use cnn_test_loader
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = trained_cnn_model(features)
            if num_classes_for_cnn == 1:
                predicted_final = (torch.sigmoid(outputs) > 0.5).float()
            else:
                _, predicted_final = torch.max(outputs.data, 1)
            all_final_preds_list.extend(predicted_final.cpu().numpy())
            all_final_labels_list.extend(labels.cpu().numpy())

    all_final_labels_np = np.array(all_final_labels_list).squeeze()
    all_final_preds_np = np.array(all_final_preds_list).squeeze()

    target_names_report = ['Normal (Class 0)', 'Abnormal (Class 1)']
    final_report_dict = classification_report(all_final_labels_np, all_final_preds_np, 
                                              target_names=target_names_report, output_dict=True, zero_division=0)
    duration_final_eval = time.time() - start_time_final_eval
    step_timings['Final CNN Evaluation'] = format_time(duration_final_eval)
    print(f"--- Finished: Final CNN Evaluation in {format_time(duration_final_eval)} ---")

    print("\n--- Final Test Performance Report (Percentage Format) ---")
    # Overall Accuracy
    if 'accuracy' in final_report_dict:
        print(f"Overall Accuracy: {final_report_dict['accuracy']*100:.2f}%")
    
    print("-" * 50) # Separator

    # Per-class metrics
    for class_name in target_names_report:
        if class_name in final_report_dict:
            metrics = final_report_dict[class_name]
            print(f"Class: {class_name}")
            print(f"  Recall:    {metrics['recall']*100:.2f}%")
            print(f"  Precision: {metrics['precision']*100:.2f}%")
            print(f"  F1-Score:  {metrics['f1-score']*100:.2f}%")
            print(f"  Support:   {metrics['support']}") # Support is a count, not percentage
            print("-" * 30) # Separator for classes
            
    # Averaged metrics (macro and weighted)
    for avg_type in ['macro avg', 'weighted avg']:
        if avg_type in final_report_dict:
            metrics = final_report_dict[avg_type]
            print(f"{avg_type.replace('avg', 'Average').title()}:") # Make it look nicer e.g. "Macro Average"
            print(f"  Recall:    {metrics['recall']*100:.2f}%")
            print(f"  Precision: {metrics['precision']*100:.2f}%")
            print(f"  F1-Score:  {metrics['f1-score']*100:.2f}%")
            # Support for averages might also be present in report_dict, but usually not displayed as prominently
            if 'support' in metrics:
                 print(f"  Support:   {metrics['support']}")
            print("-" * 30)
    
    # The line below that printed the default string report can now be removed or commented out:
    # print(classification_report(all_final_labels_np, all_final_preds_np, target_names=target_names_report, zero_division=0))
    
    save_results_table(final_report_dict, step_timings, output_path="nslkdd_pytorch_results.png")

    print("\nProcess finished.")
    print("\n--- Summary of Step Timings ---")
    for step, time_taken in step_timings.items():
        print(f"Time for {step}: {time_taken}")