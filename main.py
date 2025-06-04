# main.py
import os
import time
import torch
import torch.nn as nn # For loss functions if not moved
from torch.utils.data import DataLoader
import numpy as np # For np.abs in BEGAN M calc, though torch.abs can be used
from sklearn.metrics import classification_report # For final report dict

# Import from our modules
import configs as cfg
from utils import TrainingLossEarlyStopper, format_time, save_results_table
from data_loader import NSLKDDDataset, load_and_preprocess_nsl_kdd
from models import Autoencoder, Generator, CNNClassifier
from training_logic import (
    train_autoencoder, train_began_for_class, train_classifier,
    extract_ae_features, augment_data_with_gan
)

if __name__ == '__main__':
    print(f"Using device: {cfg.DEVICE}")
    step_timings = {}
    
    ae_early_stopper = TrainingLossEarlyStopper(patience=cfg.EARLY_STOPPING_PATIENCE, rel_delta_threshold=cfg.EARLY_STOPPING_REL_DELTA)
    cnn_early_stopper = TrainingLossEarlyStopper(patience=cfg.EARLY_STOPPING_PATIENCE, rel_delta_threshold=cfg.EARLY_STOPPING_REL_DELTA)

    # --- Step 1: Load and preprocess original data ---
    print("\n--- Starting: Data Loading & Preprocessing ---")
    start_time = time.time()
    X_train_original_tensor, y_train_original_tensor, \
    X_test_tensor, y_test_tensor, NUM_FEATURES_PREPROCESSED = \
        load_and_preprocess_nsl_kdd(cfg.BASE_DATA_PATH, cfg.TRAIN_FILENAME, cfg.TEST_FILENAME)
    duration = time.time() - start_time
    step_timings['Data Loading & Preprocessing'] = format_time(duration)
    print(f"--- Finished: Data Loading & Preprocessing in {format_time(duration)} ---")
    print(f"Original training data: X={X_train_original_tensor.shape}, y={y_train_original_tensor.shape}")
    print(f"Original test data: X={X_test_tensor.shape}, y={y_test_tensor.shape}")
    print(f"Number of features preprocessed: {NUM_FEATURES_PREPROCESSED}")


    # --- Step 2: Class-wise BEGAN Training and Data Augmentation ---
    X_train_final_tensor_cpu_list = [X_train_original_tensor.cpu()]
    y_train_final_tensor_cpu_list = [y_train_original_tensor.cpu()]

    if cfg.USE_GAN_AUGMENTATION:
        print("\n--- Starting: Class-wise BEGAN Training & Augmentation ---")
        began_total_start_time = time.time()
        
        unique_classes = torch.unique(y_train_original_tensor.squeeze().cpu()).numpy()
        print(f"Found unique classes for BEGAN training: {unique_classes}")

        for class_label_float in unique_classes:
            class_label = int(class_label_float) # Ensure it's an int for dict keys etc.
            class_name_str = f"Class_{class_label}"
            print(f"\nProcessing BEGAN for {class_name_str}")

            class_indices = (y_train_original_tensor.squeeze().cpu() == class_label_float)
            X_class_data_cpu = X_train_original_tensor.cpu()[class_indices]

            if len(X_class_data_cpu) < cfg.GAN_BATCH_SIZE:
                print(f"Skipping BEGAN for {class_name_str}: Insufficient samples ({len(X_class_data_cpu)}). Needs at least {cfg.GAN_BATCH_SIZE}.")
                continue
            
            class_dataset = NSLKDDDataset(X_class_data_cpu) # Features only for BEGAN D training
            class_dataloader = DataLoader(class_dataset, batch_size=cfg.GAN_BATCH_SIZE, shuffle=True, drop_last=True)

            g_began_class = Generator(cfg.BEGAN_NOISE_DIM, cfg.BEGAN_GENERATOR_HIDDEN_DIM, NUM_FEATURES_PREPROCESSED)
            d_began_class_ae = Autoencoder(NUM_FEATURES_PREPROCESSED, cfg.AE_LATENT_DIM, cfg.AE_HIDDEN_DIM_1)
            
            print(f"  Generator for {class_name_str}:")
            # print(g_began_class) # Can be verbose
            print(f"  Discriminator (AE) for {class_name_str}:")
            # print(d_began_class_ae)

            trained_g_class = train_began_for_class(
                g_began_class, d_began_class_ae, class_dataloader,
                cfg.BEGAN_NOISE_DIM, cfg.BEGAN_MAX_EPOCHS_PER_CLASS, cfg.DEVICE, class_name_str,
                cfg.BEGAN_GAMMA, cfg.BEGAN_LAMBDA_K, cfg.BEGAN_K_T_INITIAL, cfg.BEGAN_M_THRESHOLD, cfg.BEGAN_LR
            )
            
            num_to_generate_for_class = cfg.NUM_SYNTHETIC_SAMPLES_PER_CLASS_MAP.get(class_label, 0)
            if num_to_generate_for_class > 0:
                print(f"Generating {num_to_generate_for_class} synthetic samples for {class_name_str}...")
                synthetic_class_features_cpu = augment_data_with_gan(trained_g_class, cfg.BEGAN_NOISE_DIM, 
                                                                  num_to_generate_for_class, cfg.DEVICE, cfg.GAN_BATCH_SIZE)
                if synthetic_class_features_cpu.size(0) > 0:
                    synthetic_class_labels_cpu = torch.full((synthetic_class_features_cpu.size(0), 1), 
                                                            float(class_label), dtype=torch.float) # Match y_train type
                    X_train_final_tensor_cpu_list.append(synthetic_class_features_cpu)
                    y_train_final_tensor_cpu_list.append(synthetic_class_labels_cpu)
                    print(f"  Added {synthetic_class_features_cpu.size(0)} synthetic samples for {class_name_str}.")
                else:
                    print(f"  No synthetic samples generated for {class_name_str}.")
            else:
                print(f"  Skipping synthetic sample generation for {class_name_str} (num_to_generate is 0).")

        step_timings['Class-wise BEGAN Training & Augmentation'] = format_time(time.time() - began_total_start_time)
    else:
        print("\n--- Skipping BEGAN-based Data Augmentation (USE_GAN_AUGMENTATION is False) ---")
        step_timings['Class-wise BEGAN Training & Augmentation'] = "Skipped"

    X_train_final_tensor = torch.cat(X_train_final_tensor_cpu_list, dim=0)
    y_train_final_tensor = torch.cat(y_train_final_tensor_cpu_list, dim=0)
    
    shuffle_indices = torch.randperm(X_train_final_tensor.size(0))
    X_train_final_tensor = X_train_final_tensor[shuffle_indices]
    y_train_final_tensor = y_train_final_tensor[shuffle_indices]
    print(f"Final training data shape after augmentation (if any): X={X_train_final_tensor.shape}, y={y_train_final_tensor.shape}")


    # --- Step 3: Feature Extraction Autoencoder (trained on final augmented data) ---
    current_extracted_features_file = cfg.EXTRACTED_FEATURES_FILE_GAN_AUGMENTED if cfg.USE_GAN_AUGMENTATION else cfg.EXTRACTED_FEATURES_FILE_ORIGINAL
    X_train_features_ae, X_test_features_ae = None, None
    actual_latent_dim_for_cnn = cfg.AE_LATENT_DIM # Default

    if os.path.exists(current_extracted_features_file):
        print(f"\n--- Loading pre-extracted AE features from {current_extracted_features_file} ---")
        start_time = time.time()
        try:
            loaded_data = torch.load(current_extracted_features_file, map_location=cfg.DEVICE)
            X_train_features_ae = loaded_data['train_features'].to(cfg.DEVICE)
            X_test_features_ae = loaded_data['test_features'].to(cfg.DEVICE)
            actual_latent_dim_for_cnn = loaded_data.get('latent_dim', X_train_features_ae.shape[1])
            if actual_latent_dim_for_cnn != cfg.AE_LATENT_DIM:
                 print(f"  Info: Loaded features have latent_dim {actual_latent_dim_for_cnn}, global AE_LATENT_DIM is {cfg.AE_LATENT_DIM}. Using {actual_latent_dim_for_cnn} for CNN.")
            step_timings['AE Feature Loading'] = format_time(time.time() - start_time)
            step_timings['Feature Extraction AE Training'] = "Skipped (loaded features)"
            step_timings['AE Feature Extraction (Train/Test)'] = "Skipped (loaded features)"
            print(f"  Loaded features: Train shape {X_train_features_ae.shape}, Test shape {X_test_features_ae.shape}")
        except Exception as e:
            print(f"  Error loading features: {e}. Will proceed to train AE and extract.")
            X_train_features_ae = None # Force re-extraction
            
    if X_train_features_ae is None:
        print("\n--- Starting: Feature Extraction AE Training (on final train data) ---")
        start_time = time.time()
        # This AE is for feature extraction, distinct from BEGAN's Discriminator AEs
        feature_extraction_ae = Autoencoder(NUM_FEATURES_PREPROCESSED, cfg.AE_LATENT_DIM, cfg.AE_HIDDEN_DIM_1)
        # print(feature_extraction_ae) # Can be verbose
        
        # AE uses X_train_final_tensor (potentially augmented)
        ae_train_dataset = NSLKDDDataset(X_train_final_tensor.to(cfg.DEVICE)) # Move data to device for AE
        ae_train_loader = DataLoader(ae_train_dataset, batch_size=cfg.AE_BATCH_SIZE, shuffle=True)
        train_autoencoder(feature_extraction_ae, ae_train_loader, cfg.DEFAULT_EPOCHS, 
                          cfg.AE_LEARNING_RATE, cfg.DEVICE, early_stopper=ae_early_stopper)
        step_timings['Feature Extraction AE Training'] = format_time(time.time() - start_time)

        print("\n--- Starting: AE Feature Extraction (Train/Test) ---")
        start_time = time.time()
        # For train features (from X_train_final_tensor)
        ae_train_extract_loader = DataLoader(NSLKDDDataset(X_train_final_tensor), batch_size=cfg.AE_BATCH_SIZE, shuffle=False)
        X_train_features_ae = extract_ae_features(feature_extraction_ae, ae_train_extract_loader, cfg.DEVICE, cfg.AE_BATCH_SIZE)
        
        # For test features (from original X_test_tensor)
        ae_test_extract_loader = DataLoader(NSLKDDDataset(X_test_tensor), batch_size=cfg.AE_BATCH_SIZE, shuffle=False)
        X_test_features_ae = extract_ae_features(feature_extraction_ae, ae_test_extract_loader, cfg.DEVICE, cfg.AE_BATCH_SIZE)
        
        actual_latent_dim_for_cnn = X_train_features_ae.shape[1] if X_train_features_ae.nelement() > 0 else cfg.AE_LATENT_DIM
        step_timings['AE Feature Extraction (Train/Test)'] = format_time(time.time() - start_time)
        
        print(f"  Saving AE features to {current_extracted_features_file}...")
        try:
            torch.save({
                'train_features': X_train_features_ae.cpu(),
                'test_features': X_test_features_ae.cpu(),
                'latent_dim': actual_latent_dim_for_cnn
            }, current_extracted_features_file)
            print(f"  Features saved.")
        except Exception as e:
            print(f"  Error saving features: {e}")

    if X_train_features_ae is None or X_test_features_ae is None or X_train_features_ae.nelement() == 0 or X_test_features_ae.nelement() == 0 :
        print("Critical Error: AE features for CNN are not available. Exiting.")
        exit()
    
    # Ensure features are on device for CNN
    X_train_features_ae = X_train_features_ae.to(cfg.DEVICE)
    X_test_features_ae = X_test_features_ae.to(cfg.DEVICE)
    print(f"AE features ready for CNN: Train shape {X_train_features_ae.shape}, Test shape {X_test_features_ae.shape}")
    print(f"Using latent dimension for CNN: {actual_latent_dim_for_cnn}")


    # --- Step 4: CNN Classifier Training ---
    cnn_train_dataset = NSLKDDDataset(X_train_features_ae, y_train_final_tensor.to(cfg.DEVICE)) # Ensure labels are also on device
    cnn_train_loader = DataLoader(cnn_train_dataset, batch_size=cfg.CNN_BATCH_SIZE, shuffle=True)
    cnn_test_dataset = NSLKDDDataset(X_test_features_ae, y_test_tensor.to(cfg.DEVICE))
    cnn_test_loader = DataLoader(cnn_test_dataset, batch_size=cfg.CNN_BATCH_SIZE, shuffle=False)

    cnn_model = CNNClassifier(
        input_dim_cnn=actual_latent_dim_for_cnn,
        cnn_filters=cfg.CNN_FILTERS, cnn_kernel_size=cfg.CNN_KERNEL_SIZE,
        cnn_pool_size=cfg.CNN_POOL_SIZE, cnn_pool_stride=cfg.CNN_POOL_STRIDE,
        cnn_fc_neurons=cfg.CNN_FC_NEURONS, num_classes=1 # Binary
    )
    # print(cnn_model) # Can be verbose

    print("\n--- Starting: CNN Classifier Training & Validation ---")
    start_time = time.time()
    trained_cnn_model = train_classifier(
        cnn_model, cnn_train_loader, cnn_test_loader, cfg.CNN_EPOCHS, 
        cfg.CNN_LEARNING_RATE, cfg.DEVICE, num_classes=1, early_stopper=cnn_early_stopper
    )
    step_timings['CNN Training & Validation'] = format_time(time.time() - start_time)


    # --- Step 5: Final Evaluation and Reporting ---
    print("\n--- Starting: Final CNN Evaluation on Test Data ---")
    start_time = time.time()
    trained_cnn_model.eval()
    all_final_preds_list, all_final_labels_list = [], []
    with torch.no_grad():
        for features, labels in cnn_test_loader:
            features, labels = features.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            outputs = trained_cnn_model(features)
            predicted_final = (torch.sigmoid(outputs) > 0.5).float()
            all_final_preds_list.extend(predicted_final.cpu().numpy().squeeze())
            all_final_labels_list.extend(labels.cpu().numpy().squeeze())

    final_report_dict = {}
    if len(all_final_labels_list) > 0:
        final_report_dict = classification_report(
            all_final_labels_list, all_final_preds_list, 
            target_names=['Normal (Class 0)', 'Abnormal (Class 1)'], 
            output_dict=True, zero_division=0
        )
    else:
        print("Warning: No data in final test evaluation to generate report.")

    step_timings['Final CNN Evaluation'] = format_time(time.time() - start_time)
    
    print("\n--- Final Test Performance Report (Percentage Format) ---")
    if 'accuracy' in final_report_dict: print(f"Overall Accuracy: {final_report_dict['accuracy']*100:.2f}%")
    print("-" * 50)
    for class_name_report in ['Normal (Class 0)', 'Abnormal (Class 1)']:
        if class_name_report in final_report_dict:
            metrics = final_report_dict[class_name_report]
            print(f"Class: {class_name_report}")
            print(f"  Precision: {metrics['precision']*100:.2f}% | Recall: {metrics['recall']*100:.2f}% | F1-Score: {metrics['f1-score']*100:.2f}% | Support: {metrics['support']}")
            print("-" * 30)
    for avg_type in ['macro avg', 'weighted avg']:
        if avg_type in final_report_dict:
            metrics = final_report_dict[avg_type]
            print(f"{avg_type.replace('avg', 'Average').title()}:")
            print(f"  Precision: {metrics['precision']*100:.2f}% | Recall: {metrics['recall']*100:.2f}% | F1-Score: {metrics['f1-score']*100:.2f}%")
            if 'support' in metrics: print(f"  Support: {metrics['support']}")
            print("-" * 30)
            
    table_output_file = cfg.TABLE_OUTPUT_FILENAME_GAN if cfg.USE_GAN_AUGMENTATION else cfg.TABLE_OUTPUT_FILENAME_ORIGINAL
    save_results_table(final_report_dict, step_timings, output_path=table_output_file)

    print("\nProcess finished.")
    print("\n--- Summary of Step Timings ---")
    for step, time_taken in step_timings.items():
        print(f"Time for {step}: {time_taken}")