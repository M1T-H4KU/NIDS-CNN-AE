# data_loader.py
import os
import io
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.io import arff # For handling ARFF files
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import configs as cfg # Import configurations

class NSLKDDDataset(Dataset):
    """
    Custom PyTorch Dataset for NSL-KDD data.
    """
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
        return x # For unsupervised tasks like AE or GAN training

def map_attack_to_multiclass(attack_type_str):
    """Maps a raw attack string to its integer multi-class label using the config mapping."""
    return cfg.NSL_KDD_CLASS_MAPPING_STR_TO_INT.get(
        attack_type_str, 
        cfg.DEFAULT_ATTACK_LABEL_INT if attack_type_str != 'normal' else 0
    )

def apply_outlier_removal(df, label_series, numerical_col_names, class_map_int_to_str):
    """
    Applies MAD-based outlier removal per class on numerical features.
    Returns a DataFrame with outlier rows removed and the corresponding filtered label_series.
    """
    df_with_labels = df.copy()
    temp_label_col = '~class_label_temp~' # Use a temporary, unique column name
    df_with_labels[temp_label_col] = label_series

    rows_to_drop_indices = set()

    for class_int_label, class_str_name in class_map_int_to_str.items():
        class_data_df = df_with_labels[df_with_labels[temp_label_col] == class_int_label]
        if class_data_df.empty:
            continue
        
        # print(f"  Applying outlier analysis for {class_str_name} (Label: {class_int_label}), samples: {len(class_data_df)}")

        for col_name in numerical_col_names:
            feature_data = class_data_df[col_name].astype(float)
            
            if feature_data.empty or feature_data.nunique() < 2: # Skip if no data or all values are the same
                continue

            median_val = feature_data.median()
            abs_diff_from_median = np.abs(feature_data - median_val)
            mad = abs_diff_from_median.median()

            if mad == 0:
                continue

            sigma_hat = 1.4826 * mad
            threshold = 10 * sigma_hat
            
            outliers_in_feature = feature_data[abs_diff_from_median > threshold]
            
            if not outliers_in_feature.empty:
                rows_to_drop_indices.update(outliers_in_feature.index)

    if rows_to_drop_indices:
        print(f"  Total unique rows to drop due to outliers: {len(rows_to_drop_indices)}")
        df_cleaned = df.drop(index=list(rows_to_drop_indices))
        label_series_cleaned = label_series.drop(index=list(rows_to_drop_indices))
        print(f"  Shape after outlier removal: {df_cleaned.shape}")
    else:
        print("  No outliers found or removed based on MAD criteria.")
        df_cleaned = df
        label_series_cleaned = label_series
        
    return df_cleaned, label_series_cleaned


def load_and_preprocess_nsl_kdd(base_path, train_filename, test_filename, perform_outlier_removal=False):
    """
    Loads and preprocesses NSL-KDD data from either TXT or ARFF files.
    The process includes optional outlier removal, one-hot encoding, and min-max scaling.
    Returns PyTorch tensors for features and multi-class labels, and the number of processed features.
    """
    print(f"Loading NSL-KDD data from {base_path}...")
    print(f"Outlier removal: {'Enabled' if perform_outlier_removal else 'Disabled'}")
    train_path = os.path.join(base_path, train_filename)
    test_path = os.path.join(base_path, test_filename)

    df_train_raw_full = None
    df_test_raw_full = None

    # --- Load data based on file extension ---
    if train_filename.lower().endswith('.arff'):
        print(f"Parsing ARFF files: {train_filename}, {test_filename}")
        try:
            with open(train_path, 'r', encoding='utf-8') as f:
                train_content_str = f.read()
            with open(test_path, 'r', encoding='utf-8') as f:
                test_content_str = f.read()

            fixed_train_content = train_content_str.replace(",' 'icmp''", ",icmp")
            fixed_test_content = test_content_str.replace(",' 'icmp''", ",icmp")

            train_data, train_meta = arff.loadarff(io.StringIO(fixed_train_content))
            test_data, test_meta = arff.loadarff(io.StringIO(fixed_test_content))

        except FileNotFoundError:
            print(f"Error: Ensure '{train_filename}' and '{test_filename}' exist in '{base_path}'")
            raise
        
        df_train_raw_full = pd.DataFrame(train_data)
        df_test_raw_full = pd.DataFrame(test_data)

        for col in df_train_raw_full.select_dtypes([object]).columns:
            df_train_raw_full[col] = df_train_raw_full[col].str.decode('utf-8', errors='ignore')
        for col in df_test_raw_full.select_dtypes([object]).columns:
            df_test_raw_full[col] = df_test_raw_full[col].str.decode('utf-8', errors='ignore')
            
        if 'class' in df_train_raw_full.columns:
            df_train_raw_full.rename(columns={'class': 'attack_type'}, inplace=True)
        if 'class' in df_test_raw_full.columns:
            df_test_raw_full.rename(columns={'class': 'attack_type'}, inplace=True)

    elif train_filename.lower().endswith('.txt'):
        print(f"Parsing TXT files: {train_filename}, {test_filename}")
        column_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root',
            'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
            'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
            'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level'
        ]
        try:
            df_train_raw_full = pd.read_csv(train_path, header=None, names=column_names)
            df_test_raw_full = pd.read_csv(test_path, header=None, names=column_names)
        except FileNotFoundError:
            print(f"Error: Ensure '{train_filename}' and '{test_filename}' exist in '{base_path}'")
            raise
    else:
        raise ValueError(f"Unsupported file format for: {train_filename}")

    # --- Preprocessing Pipeline ---
    
    # 1. Separate features (X) and labels
    cols_to_drop = ['attack_type']
    if 'difficulty_level' in df_train_raw_full.columns:
        cols_to_drop.append('difficulty_level')
        
    X_train_raw = df_train_raw_full.drop(columns=cols_to_drop)
    y_train_str = df_train_raw_full['attack_type']
    X_test_raw = df_test_raw_full.drop(columns=cols_to_drop)
    y_test_str = df_test_raw_full['attack_type']

    # 2. Choose label mapping based on file type (binary vs multi-class)
    is_binary_mode_data = train_filename.lower().endswith('.arff')
    if is_binary_mode_data:
        print("Applying BINARY label mapping (Normal: 0, Abnormal: 1)")
        y_train_multiclass = y_train_str.apply(lambda x: 0 if x == 'normal' else 1).values
        y_test_multiclass = y_test_str.apply(lambda x: 0 if x == 'normal' else 1).values
    else: # Multi-class mode from TXT file
        print("Applying MULTI-CLASS label mapping (Normal: 0, DoS: 1, ...)")
        y_train_multiclass = y_train_str.apply(map_attack_to_multiclass).values
        y_test_multiclass = y_test_str.apply(map_attack_to_multiclass).values
        
    y_train_multiclass_series = pd.Series(y_train_multiclass, index=X_train_raw.index)
    y_test_multiclass_series = pd.Series(y_test_multiclass, index=X_test_raw.index)
    
    print(f"Original train shape: {X_train_raw.shape}, Original test shape: {X_test_raw.shape}")
    
    # 3. Outlier Analysis and Removal (Conditional)
    if perform_outlier_removal:
        categorical_feature_names = ['protocol_type', 'service', 'flag']
        numerical_col_names = [col for col in X_train_raw.columns if col not in categorical_feature_names]
        
        print("\nPerforming Outlier Analysis on Training Data...")
        X_train_cleaned, y_train_multiclass_series_cleaned = apply_outlier_removal(
            X_train_raw, y_train_multiclass_series, numerical_col_names, cfg.NSL_KDD_CLASS_NAMES_INT_TO_STR
        )
        print("\nPerforming Outlier Analysis on Test Data...")
        X_test_cleaned, y_test_multiclass_series_cleaned = apply_outlier_removal(
            X_test_raw, y_test_multiclass_series, numerical_col_names, cfg.NSL_KDD_CLASS_NAMES_INT_TO_STR
        )
    else:
        print("\nSkipping Outlier Analysis.")
        X_train_cleaned, y_train_multiclass_series_cleaned = X_train_raw, y_train_multiclass_series
        X_test_cleaned, y_test_multiclass_series_cleaned = X_test_raw, y_test_multiclass_series

    y_train_multiclass_final = y_train_multiclass_series_cleaned.values
    y_test_multiclass_final = y_test_multiclass_series_cleaned.values

    # 4. One-Hot Encoding and Min-Max Scaling on cleaned data
    categorical_feature_names = ['protocol_type', 'service', 'flag']
    num_indices_for_ct = [X_train_cleaned.columns.get_loc(col) for col in X_train_cleaned.columns if col not in categorical_feature_names]
    cat_indices_for_ct = [X_train_cleaned.columns.get_loc(col) for col in categorical_feature_names if col in X_train_cleaned.columns]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('scaler', MinMaxScaler())]), num_indices_for_ct),
            ('cat', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_indices_for_ct)
        ],
        remainder='passthrough'
    )
    
    X_train_processed = preprocessor.fit_transform(X_train_cleaned).astype(np.float32)
    X_test_processed = preprocessor.transform(X_test_cleaned).astype(np.float32)

    num_features_processed = X_train_processed.shape[1]
    print(f"\nData preprocessed. Final number of features: {num_features_processed}")
    print(f"Processed X_train shape: {X_train_processed.shape}, y_train (cleaned) shape: {y_train_multiclass_final.shape}")
    print(f"Processed X_test shape: {X_test_processed.shape}, y_test (cleaned) shape: {y_test_multiclass_final.shape}")

    # 5. Convert to PyTorch Tensors
    X_train_original_tensor = torch.from_numpy(X_train_processed)
    y_train_original_multiclass_tensor = torch.from_numpy(y_train_multiclass_final).long().unsqueeze(1)
    
    X_test_original_tensor = torch.from_numpy(X_test_processed)
    y_test_original_multiclass_tensor = torch.from_numpy(y_test_multiclass_final).long().unsqueeze(1)
    
    return X_train_original_tensor, y_train_original_multiclass_tensor, \
           X_test_original_tensor, y_test_original_multiclass_tensor, \
           num_features_processed