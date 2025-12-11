# data_loader.py
import os
import io
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler
import configs as cfg

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
        return x

def map_attack_to_multiclass(attack_type_str):
    """Maps a raw attack string to its integer multi-class label using the config mapping."""
    return cfg.NSL_KDD_CLASS_MAPPING_STR_TO_INT.get(
        attack_type_str, 
        cfg.DEFAULT_ATTACK_LABEL_INT if attack_type_str != 'normal' else 0
    )

def learn_outlier_thresholds(df, label_series, numerical_col_names, class_map_int_to_str):
    """
    Learns MAD-based outlier thresholds ONLY from the provided training data.
    Returns a dictionary of thresholds.
    """
    thresholds = {}
    df_with_labels = df.copy()
    temp_label_col = '~class_label_temp~'
    df_with_labels[temp_label_col] = label_series

    for class_int_label, class_str_name in class_map_int_to_str.items():
        thresholds[class_int_label] = {}
        class_data_df = df_with_labels[df_with_labels[temp_label_col] == class_int_label]
        if class_data_df.empty:
            continue
        
        for col_name in numerical_col_names:
            feature_data = class_data_df[col_name].astype(float)
            if feature_data.empty or feature_data.nunique() < 2:
                continue
                
            median_val = feature_data.median()
            abs_diff_from_median = np.abs(feature_data - median_val)
            mad = abs_diff_from_median.median()
            
            if mad == 0:
                threshold = np.inf
            else:
                sigma_hat = 1.4826 * mad
                threshold = 10 * sigma_hat
            
            thresholds[class_int_label][col_name] = {'median': median_val, 'threshold': threshold}
            
    return thresholds

def remove_outliers_with_thresholds(df, label_series, numerical_col_names, thresholds):
    """
    Removes outliers from a dataset (train or test) using pre-computed thresholds.
    """
    rows_to_drop_indices = set()
    
    for index, row in df.iterrows():
        class_label = label_series.loc[index]
        if class_label not in thresholds:
            continue
            
        for col_name in numerical_col_names:
            if col_name not in thresholds[class_label]:
                continue
            
            class_thresholds = thresholds[class_label][col_name]
            median_val = class_thresholds['median']
            threshold = class_thresholds['threshold']
            
            if np.abs(row[col_name] - median_val) > threshold:
                rows_to_drop_indices.add(index)
                break 
    if rows_to_drop_indices:
        print(f"  Total unique rows to drop due to outliers: {len(rows_to_drop_indices)}")
        df_cleaned = df.drop(index=list(rows_to_drop_indices))
        label_series_cleaned = label_series.drop(index=list(rows_to_drop_indices))
        print(f"  Shape after outlier removal: {df_cleaned.shape}")
    else:
        print("  No outliers found or removed based on the learned thresholds.")
        df_cleaned, label_series_cleaned = df, label_series
        
    return df_cleaned, label_series_cleaned

def load_and_preprocess_nsl_kdd(base_path, train_filename, test_filename, perform_outlier_removal=False):
    """
    Loads and preprocesses NSL-KDD data from either TXT or ARFF files.
    """
    print(f"Loading NSL-KDD data from {base_path}...")
    print(f"Outlier removal: {'Enabled' if perform_outlier_removal else 'Disabled'}")
    train_path = os.path.join(base_path, train_filename)
    test_path = os.path.join(base_path, test_filename)
    df_train_raw_full, df_test_raw_full = None, None

    if train_filename.lower().endswith('.arff'):
        print(f"Parsing ARFF files: {train_filename}, {test_filename}")
        try:
            with open(train_path, 'r', encoding='utf-8') as f:
                train_content_str = f.read()
            with open(test_path, 'r', encoding='utf-8') as f:
                test_content_str = f.read()
            fixed_train_content = train_content_str.replace(",' 'icmp''", ",icmp")
            fixed_test_content = test_content_str.replace(",' 'icmp''", ",icmp")
            train_data, _ = arff.loadarff(io.StringIO(fixed_train_content))
            test_data, _ = arff.loadarff(io.StringIO(fixed_test_content))
            df_train_raw_full, df_test_raw_full = pd.DataFrame(train_data), pd.DataFrame(test_data)
            for df in [df_train_raw_full, df_test_raw_full]:
                for col in df.select_dtypes([object]).columns:
                    df[col] = df[col].str.decode('utf-8', errors='ignore')
                if 'class' in df.columns: df.rename(columns={'class': 'attack_type'}, inplace=True)
        except FileNotFoundError:
            print(f"Error: Ensure '{train_filename}' and '{test_filename}' exist in '{base_path}'")
            raise
    elif train_filename.lower().endswith('.txt'):
        print(f"Parsing TXT files: {train_filename}, {test_filename}")
        column_names = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','attack_type','difficulty_level']
        try:
            df_train_raw_full = pd.read_csv(train_path, header=None, names=column_names)
            df_test_raw_full = pd.read_csv(test_path, header=None, names=column_names)
        except FileNotFoundError:
            print(f"Error: Ensure '{train_filename}' and '{test_filename}' exist in '{base_path}'")
            raise
    else:
        raise ValueError(f"Unsupported file format for: {train_filename}")

    # --- Enforce numeric types BEFORE any other processing ---
    categorical_feature_names = ['protocol_type', 'service', 'flag']
    feature_cols = [col for col in df_train_raw_full.columns if col not in ['attack_type', 'difficulty_level', 'class']]
    numerical_col_names = [col for col in feature_cols if col not in categorical_feature_names]
    print("\nEnforcing numeric types on numerical columns...")
    for col in numerical_col_names:
        df_train_raw_full[col] = pd.to_numeric(df_train_raw_full[col], errors='coerce')
        df_test_raw_full[col] = pd.to_numeric(df_test_raw_full[col], errors='coerce')
    df_train_raw_full.fillna(0, inplace=True)
    df_test_raw_full.fillna(0, inplace=True)

    # --- Preprocessing Pipeline ---
    cols_to_drop = ['attack_type']
    if 'difficulty_level' in df_train_raw_full.columns: cols_to_drop.append('difficulty_level')
    X_train_raw, y_train_str = df_train_raw_full.drop(columns=cols_to_drop), df_train_raw_full['attack_type']
    X_test_raw, y_test_str = df_test_raw_full.drop(columns=cols_to_drop), df_test_raw_full['attack_type']
    
    is_binary_mode_data = train_filename.lower().endswith('.arff')
    if is_binary_mode_data:
        print("Applying BINARY label mapping (Normal: 0, Abnormal: 1)")
        y_train_labels, y_test_labels = y_train_str.apply(lambda x: 0 if x == 'normal' else 1).values, y_test_str.apply(lambda x: 0 if x == 'normal' else 1).values
    else:
        print("Applying MULTI-CLASS label mapping (Normal: 0, DoS: 1, ...)")
        y_train_labels, y_test_labels = y_train_str.apply(map_attack_to_multiclass).values, y_test_str.apply(map_attack_to_multiclass).values
    
    y_train_series, y_test_series = pd.Series(y_train_labels, index=X_train_raw.index), pd.Series(y_test_labels, index=X_test_raw.index)
    X_train_cleaned, y_train_series_cleaned = X_train_raw, y_train_series
    X_test_cleaned, y_test_series_cleaned = X_test_raw, y_test_series

    if perform_outlier_removal:
        class_map = cfg.NSL_KDD_CLASS_NAMES_INT_TO_STR if not is_binary_mode_data else {0: 'Normal', 1: 'Abnormal'}
        print("\nLearning Outlier Thresholds from Training Data ONLY...")
        outlier_thresholds = learn_outlier_thresholds(X_train_raw, y_train_series, numerical_col_names, class_map)
        print("\nApplying learned thresholds to remove outliers from Training Data...")
        X_train_cleaned, y_train_series_cleaned = remove_outliers_with_thresholds(X_train_raw, y_train_series, numerical_col_names, outlier_thresholds)
        print("\nApplying learned thresholds to remove outliers from Test Data...")
        X_test_cleaned, y_test_series_cleaned = remove_outliers_with_thresholds(X_test_raw, y_test_series, numerical_col_names, outlier_thresholds)
    else:
        print("\nSkipping Outlier Analysis.")

    y_train_final, y_test_final = y_train_series_cleaned.values, y_test_series_cleaned.values

    # Combine features and labels before pd.get_dummies
    X_train_cleaned['label'] = y_train_final
    X_test_cleaned['label'] = y_test_final
    combined_data = pd.concat([X_train_cleaned, X_test_cleaned], ignore_index=True)

    print("\nApplying one-hot encoding using pd.get_dummies()...")
    combined_data_ohe = pd.get_dummies(combined_data, columns=categorical_feature_names)

    X_train_ohe = combined_data_ohe.iloc[:len(X_train_cleaned)]
    X_test_ohe = combined_data_ohe.iloc[len(X_train_cleaned):]
    
    y_train_final = X_train_ohe.pop('label').values
    y_test_final = X_test_ohe.pop('label').values

    feature_names_after_ohe = X_train_ohe.columns.tolist()
    num_features_processed = len(feature_names_after_ohe)
    
    print("Applying MinMaxScaler to all features...")
    scaler = MinMaxScaler()
    X_train_processed = scaler.fit_transform(X_train_ohe).astype(np.float32)
    X_test_processed = scaler.transform(X_test_ohe).astype(np.float32)

    print(f"\nData preprocessed. Final number of features: {num_features_processed}")
    
    # Note: preprocessor object is now None, and feature_names are post-OHE
    return torch.from_numpy(X_train_processed), torch.from_numpy(y_train_final).long().unsqueeze(1), \
           torch.from_numpy(X_test_processed), torch.from_numpy(y_test_final).long().unsqueeze(1), \
           num_features_processed, None, feature_names_after_ohe