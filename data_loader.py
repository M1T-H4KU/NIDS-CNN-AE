# data_loader.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import configs as cfg

# Define the multi-class mapping
NSL_KDD_CLASS_MAPPING = {
    'normal': 0,
    # DoS
    'back': 1, 'land': 1, 'neptune': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,
    'mailbomb': 1, 'processtable': 1, 'udpstorm': 1, 'apache2': 1, 'worm': 1,
    # Probe
    'satan': 2, 'ipsweep': 2, 'nmap': 2, 'portsweep': 2,
    'mscan': 2, 'saint': 2,
    # R2L
    'guess_passwd': 3, 'ftp_write': 3, 'imap': 3, 'phf': 3, 'multihop': 3,
    'warezmaster': 3, 'warezclient': 3, 'spy': 3, 'xlock': 3, 'xsnoop': 3,
    'snmpguess': 3, 'snmpgetattack': 3, 'httptunnel': 3, 'sendmail': 3, 'named': 3,
    # U2R
    'buffer_overflow': 4, 'loadmodule': 4, 'perl': 4, 'rootkit': 4,
    'sqlattack': 4, 'xterm': 4, 'ps': 4
}
# Default label for unknown attack types if any (can be mapped to a general attack category or ignored)
DEFAULT_ATTACK_LABEL = 1 # Default to DoS if specific attack not in map (or handle as error)

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

def map_attack_to_multiclass(attack_type_str):
    # Use the mapping from configs.py
    return cfg.NSL_KDD_CLASS_MAPPING_STR_TO_INT.get(
        attack_type_str, 
        cfg.DEFAULT_ATTACK_LABEL_INT if attack_type_str != 'normal' else 0
    )
def load_and_preprocess_nsl_kdd(base_path, train_filename, test_filename):
    print(f"Loading and preprocessing NSL-KDD data from {base_path} for multi-class GAN...")
    train_path = os.path.join(base_path, train_filename)
    test_path = os.path.join(base_path, test_filename)

    column_names = [ # ... (column names as before) ...
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
    y_train_labels_raw_str = df_train['attack_type'] # Keep as string initially
    X_test_raw = df_test.iloc[:, :-2]
    y_test_labels_raw_str = df_test['attack_type']   # Keep as string initially

    # Apply multi-class mapping for GAN training
    y_train_multiclass = y_train_labels_raw_str.apply(map_attack_to_multiclass).values
    y_test_multiclass = y_test_labels_raw_str.apply(map_attack_to_multiclass).values
    
    print(f"Unique multi-class labels in training set: {np.unique(y_train_multiclass)}")
    print(f"Unique multi-class labels in test set: {np.unique(y_test_multiclass)}")


    categorical_features_indices = [1, 2, 3]
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

    num_features_processed = X_train_processed.shape[1]
    print(f"Data preprocessed. Number of features: {num_features_processed}")

    # Tensors with original multi-class labels (for BEGAN)
    X_train_original_tensor = torch.from_numpy(X_train_processed)
    y_train_original_multiclass_tensor = torch.from_numpy(y_train_multiclass).long().unsqueeze(1) # Use long for class indices, keep as (N,1)
    
    X_test_original_tensor = torch.from_numpy(X_test_processed)
    y_test_original_multiclass_tensor = torch.from_numpy(y_test_multiclass).long().unsqueeze(1) # Use long for class indices

    print(f"Shapes for BEGAN/multi-class: X_train={X_train_original_tensor.shape}, y_train_multi={y_train_original_multiclass_tensor.shape}")
    
    print(f"Unique multi-class labels in training set after mapping: {np.unique(y_train_multiclass)}")
    return X_train_original_tensor, y_train_original_multiclass_tensor, \
           X_test_original_tensor, y_test_original_multiclass_tensor, \
           num_features_processed