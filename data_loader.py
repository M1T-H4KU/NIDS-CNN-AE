# data_loader.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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
        return x # For AE training or GAN real data loader

def load_and_preprocess_nsl_kdd(base_path, train_filename, test_filename):
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

    num_features_processed = X_train_processed.shape[1] # This will be returned
    print(f"Data preprocessed. Number of features: {num_features_processed}")
    print(f"X_train_processed shape: {X_train_processed.shape}, y_train shape: {y_train.shape}")
    print(f"X_test_processed shape: {X_test_processed.shape}, y_test shape: {y_test.shape}")

    X_train_tensor = torch.from_numpy(X_train_processed)
    y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1)
    X_test_tensor = torch.from_numpy(X_test_processed)
    y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(1)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, num_features_processed