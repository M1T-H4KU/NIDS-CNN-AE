import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split # if needed for a validation split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

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

    target_names_binary = ['normal (0)', 'abnormal (1)'] # For classification_report

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
        all_val_preds = []
        all_val_labels = []
        
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
                
                all_val_preds.extend(predicted_val.cpu().numpy())
                all_val_labels.extend(batch_labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        
        all_val_labels_np = np.array(all_val_labels).squeeze()
        all_val_preds_np = np.array(all_val_preds).squeeze()

        # Overall accuracy (still useful)
        val_overall_accuracy = accuracy_score(all_val_labels_np, all_val_preds_np) * 100
        
        # Classification report for per-class metrics
        # For binary classification (positive class is 1 by default in precision/recall/f1, report shows both)
        # zero_division=0 will return 0 if there are no positive predictions/labels
        report_val = classification_report(all_val_labels_np, all_val_preds_np, 
                                           target_names=target_names_binary, zero_division=0, digits=4)
        
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs -1 :
            print(f"Epoch [{epoch+1}/{epochs}]:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy_epoch:.2f}%")
            print(f"  Val Loss  : {avg_val_loss:.4f}, Val Overall Acc: {val_overall_accuracy:.2f}%")
            print(f"  Validation Classification Report:\n{report_val}")
            # The 'recall' for 'normal (0)' is your "Accuracy on normal case"
            # The 'recall' for 'abnormal (1)' is your "Accuracy on abnormal case"

    print("CNN training finished.")
    return model

# --- 6. Main Workflow ---
if __name__ == '__main__':
    # Step 1: Load and preprocess data
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = load_and_preprocess_nsl_kdd()

    # Create datasets and dataloaders for AE (only needs features for reconstruction)
    ae_train_dataset = NSLKDDDataset(X_train_tensor) # Labels are not used for AE input/output comparison
    ae_train_loader = DataLoader(ae_train_dataset, batch_size=AE_BATCH_SIZE, shuffle=True)
    # For feature extraction, also prepare test data loader
    ae_test_dataset_for_extraction = NSLKDDDataset(X_test_tensor)
    ae_test_loader_for_extraction = DataLoader(ae_test_dataset_for_extraction, batch_size=AE_BATCH_SIZE, shuffle=False)


    # Step 2.1: Build and Train Autoencoder
    autoencoder_model = Autoencoder(
        input_dim=NUM_FEATURES_PREPROCESSED,
        latent_dim=LATENT_DIM,
        hidden_dim_1=AE_HIDDEN_DIM_1
    )
    print("\n--- Autoencoder Architecture ---")
    print(autoencoder_model) # PyTorch way to see layers
    
    train_autoencoder(autoencoder_model, ae_train_loader, epochs=AE_EPOCHS)

    # Step 2.2: Extract Features using the trained Encoder part of the AE
    print("\nExtracting features using the trained Autoencoder's Encoder part...")
    # For X_train_features_ae
    ae_train_dataset_for_extraction = NSLKDDDataset(X_train_tensor) # Recreate if needed or reuse
    ae_train_loader_for_extraction = DataLoader(ae_train_dataset_for_extraction, batch_size=AE_BATCH_SIZE, shuffle=False)

    X_train_features_ae = extract_ae_features(autoencoder_model, ae_train_loader_for_extraction)
    X_test_features_ae = extract_ae_features(autoencoder_model, ae_test_loader_for_extraction)
    
    print(f"Shape of original training data: {X_train_tensor.shape}")
    print(f"Shape of AE-extracted training features: {X_train_features_ae.shape}")
    print(f"Shape of original test data: {X_test_tensor.shape}")
    print(f"Shape of AE-extracted test features: {X_test_features_ae.shape}")

    # Step 3: Train CNN Classifier with AE features
    # Create datasets and dataloaders for CNN
    cnn_train_dataset = NSLKDDDataset(X_train_features_ae, y_train_tensor)
    cnn_train_loader = DataLoader(cnn_train_dataset, batch_size=CNN_BATCH_SIZE, shuffle=True)
    
    cnn_test_dataset = NSLKDDDataset(X_test_features_ae, y_test_tensor)
    cnn_test_loader = DataLoader(cnn_test_dataset, batch_size=CNN_BATCH_SIZE, shuffle=False)

    # Build CNN model
    num_classes_for_cnn = 1 # Binary classification
    cnn_classifier_model = CNNClassifier(input_dim_cnn=LATENT_DIM, num_classes=num_classes_for_cnn)
    print("\n--- CNN Classifier Architecture ---")
    print(cnn_classifier_model)

    # Train CNN
    trained_cnn_model = train_classifier(cnn_classifier_model, cnn_train_loader, cnn_test_loader, 
                                         epochs=CNN_EPOCHS, num_classes=num_classes_for_cnn)

    # Final Evaluation on Test Data (using the test_loader)
    print("\n--- Final CNN Evaluation on Test Data ---")
    trained_cnn_model.eval() # Ensure model is in evaluation mode
    
    final_test_loss = 0
    all_final_preds = []
    all_final_labels = []
    
    criterion_cnn = nn.BCEWithLogitsLoss() if num_classes_for_cnn == 1 else nn.CrossEntropyLoss()

    with torch.no_grad():
        for features, labels in cnn_test_loader: # Use cnn_test_loader
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = trained_cnn_model(features)
            loss = criterion_cnn(outputs, labels)
            final_test_loss += loss.item()
            
            if num_classes_for_cnn == 1:
                predicted_final = (torch.sigmoid(outputs) > 0.5).float()
            else:
                _, predicted_final = torch.max(outputs.data, 1) # For multi-class
            
            all_final_preds.extend(predicted_final.cpu().numpy())
            all_final_labels.extend(labels.cpu().numpy())

    avg_final_test_loss = final_test_loss / len(cnn_test_loader)
    
    all_final_labels_np = np.array(all_final_labels).squeeze()
    all_final_preds_np = np.array(all_final_preds).squeeze()

    final_accuracy = accuracy_score(all_final_labels_np, all_final_preds_np) * 100
    final_precision = precision_score(all_final_labels_np, all_final_preds_np, zero_division=0) * 100
    final_recall = recall_score(all_final_labels_np, all_final_preds_np, zero_division=0) * 100
    final_f1 = f1_score(all_final_labels_np, all_final_preds_np, zero_division=0) * 100
    
    print(f"Test Loss: {avg_final_test_loss:.4f}")
    print(f"Test Accuracy: {final_accuracy:.2f}%")
    print(f"Test Precision: {final_precision:.2f}%")
    print(f"Test Recall: {final_recall:.2f}%")
    print(f"Test F1-Score: {final_f1:.2f}%")

    # Optional: Detailed Confusion Matrix for final test results
    final_conf_matrix = confusion_matrix(all_final_labels_np, all_final_preds_np)
    print(f"Test Confusion Matrix:\n{final_conf_matrix}")

    print("\nProcess finished.")