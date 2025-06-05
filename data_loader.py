# data_loader.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import configs as cfg # Assuming configs.py has NSL_KDD_CLASS_MAPPING_STR_TO_INT etc.

class NSLKDDDataset(Dataset):
    # ... (class definition remains the same) ...
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
    df_with_labels['~class_label~'] = label_series # Use a unique temp column name

    rows_to_drop_indices = set()

    for class_int_label, class_str_name in class_map_int_to_str.items():
        class_data_df = df_with_labels[df_with_labels['~class_label~'] == class_int_label]
        if class_data_df.empty:
            continue
        
        print(f"  Applying outlier analysis for {class_str_name} (Label: {class_int_label}), samples: {len(class_data_df)}")

        for col_name in numerical_col_names:
            feature_data = class_data_df[col_name].astype(float) # Ensure numeric type
            
            if feature_data.empty or feature_data.nunique() == 1: # Skip if no data or all values are the same
                continue

            median_val = feature_data.median()
            abs_diff_from_median = np.abs(feature_data - median_val)
            mad = abs_diff_from_median.median()

            if mad == 0: # If MAD is 0, no spread by this measure, skip outlier removal for this feature/class
                # print(f"    MAD is 0 for feature '{col_name}' in {class_str_name}. Skipping outlier removal for this feature.")
                continue

            sigma_hat = 1.4826 * mad
            threshold = 10 * sigma_hat
            
            # Identify outliers for the current feature in the current class
            # Outliers are those where |xi - median_val| > threshold
            outliers_in_feature = feature_data[abs_diff_from_median > threshold]
            
            if not outliers_in_feature.empty:
                # print(f"    Found {len(outliers_in_feature)} outliers in '{col_name}' for {class_str_name} (Threshold: {threshold:.2f}, MAD: {mad:.2f})")
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


def load_and_preprocess_nsl_kdd(base_path, train_filename, test_filename):
    print(f"Loading NSL-KDD data from {base_path} with Outlier Analysis...")
    train_path = os.path.join(base_path, train_filename)
    test_path = os.path.join(base_path, test_filename)

    column_names = [ # ... (full column names as before) ...
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
    categorical_feature_names = ['protocol_type', 'service', 'flag'] # Original categorical names

    try:
        df_train_raw_full = pd.read_csv(train_path, header=None, names=column_names)
        df_test_raw_full = pd.read_csv(test_path, header=None, names=column_names)
    except FileNotFoundError:
        print(f"Error: Ensure '{train_filename}' and '{test_filename}' exist in '{base_path}'")
        raise

    # Separate features (X) and attack_type strings (y_str)
    X_train_raw = df_train_raw_full.iloc[:, :-2]
    y_train_str = df_train_raw_full['attack_type']
    X_test_raw = df_test_raw_full.iloc[:, :-2]
    y_test_str = df_test_raw_full['attack_type']

    # Convert string labels to multi-class integer labels
    y_train_multiclass_series = y_train_str.apply(map_attack_to_multiclass)
    y_test_multiclass_series = y_test_str.apply(map_attack_to_multiclass)
    
    print(f"Original train shape: {X_train_raw.shape}, Original test shape: {X_test_raw.shape}")
    print(f"Unique multi-class labels in original training set: {np.unique(y_train_multiclass_series)}")
    print(f"Unique multi-class labels in original test set: {np.unique(y_test_multiclass_series)}")

    # Identify numerical column names from X_train_raw (excluding known categoricals)
    numerical_col_names = [col for col in X_train_raw.columns if col not in categorical_feature_names]
    
    # --- Outlier Analysis and Removal ---
    print("\nPerforming Outlier Analysis on Training Data...")
    X_train_cleaned, y_train_multiclass_series_cleaned = apply_outlier_removal(
        X_train_raw, y_train_multiclass_series, numerical_col_names, cfg.NSL_KDD_CLASS_NAMES_INT_TO_STR
    )
    print("\nPerforming Outlier Analysis on Test Data...")
    X_test_cleaned, y_test_multiclass_series_cleaned = apply_outlier_removal(
        X_test_raw, y_test_multiclass_series, numerical_col_names, cfg.NSL_KDD_CLASS_NAMES_INT_TO_STR
    )

    # Convert cleaned label series back to NumPy arrays
    y_train_multiclass = y_train_multiclass_series_cleaned.values
    y_test_multiclass = y_test_multiclass_series_cleaned.values

    # --- ColumnTransformer for One-Hot Encoding and Scaling (on cleaned data) ---
    # Get indices of categorical and numerical features for the ColumnTransformer
    # These indices are based on the columns of X_train_cleaned / X_test_cleaned
    
    # Since columns might have been dropped if all numerical, re-evaluate indices carefully
    # Better to use column names for ColumnTransformer if possible, or ensure indices are correct.
    # For simplicity, assuming X_train_cleaned still has the same columns as X_train_raw, just fewer rows.
    
    # Indices for ColumnTransformer based on original X_train_raw columns
    cat_indices_for_ct = [X_train_cleaned.columns.get_loc(col) for col in categorical_feature_names if col in X_train_cleaned.columns]
    num_indices_for_ct = [X_train_cleaned.columns.get_loc(col) for col in numerical_col_names if col in X_train_cleaned.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('scaler', MinMaxScaler())]), num_indices_for_ct),
            ('cat', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_indices_for_ct)
        ],
        remainder='passthrough' # Should be 'drop' if all columns are covered, or ensure correct indices
    )
    
    # Fit preprocessor on cleaned training data
    X_train_processed = preprocessor.fit_transform(X_train_cleaned).astype(np.float32)
    # Transform cleaned test data
    X_test_processed = preprocessor.transform(X_test_cleaned).astype(np.float32)

    num_features_processed = X_train_processed.shape[1]
    print(f"\nData preprocessed (after outlier removal, OHE, scaling). Final number of features: {num_features_processed}")
    print(f"Processed X_train shape: {X_train_processed.shape}, y_train_multiclass (cleaned) shape: {y_train_multiclass.shape}")
    print(f"Processed X_test shape: {X_test_processed.shape}, y_test_multiclass (cleaned) shape: {y_test_multiclass.shape}")

    X_train_original_tensor = torch.from_numpy(X_train_processed)
    y_train_original_multiclass_tensor = torch.from_numpy(y_train_multiclass).long().unsqueeze(1)
    
    X_test_original_tensor = torch.from_numpy(X_test_processed)
    y_test_original_multiclass_tensor = torch.from_numpy(y_test_multiclass).long().unsqueeze(1)
    
    return X_train_original_tensor, y_train_original_multiclass_tensor, \
           X_test_original_tensor, y_test_original_multiclass_tensor, \
           num_features_processed