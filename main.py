# main.py
import os
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report
import argparse # Import argparse

# Import from our modules
import configs as cfg
from utils import TrainingLossEarlyStopper, format_time, save_results_table
from data_loader import NSLKDDDataset, load_and_preprocess_nsl_kdd
from models import Autoencoder, Generator, CNNClassifier
from training_logic import (
    train_autoencoder, train_began_for_class, train_classifier,
    extract_ae_features, augment_data_with_gan
)

def main(args): # Main logic will be in a function
    print(f"Using device: {cfg.DEVICE}")
    print(f"Running with arguments: GAN Augmentation={'Enabled' if args.use_gan else 'Disabled'}, Classifier Mode='{args.classifier_mode}'")
    step_timings = {}
    
    ae_early_stopper = TrainingLossEarlyStopper(patience=cfg.EARLY_STOPPING_PATIENCE, rel_delta_threshold=cfg.EARLY_STOPPING_REL_DELTA)
    cnn_early_stopper = TrainingLossEarlyStopper(patience=cfg.EARLY_STOPPING_PATIENCE, rel_delta_threshold=cfg.EARLY_STOPPING_REL_DELTA)

    # --- Step 1: Load and preprocess original data (provides multi-class labels) ---
    print("\n--- Starting: Data Loading & Preprocessing ---")
    start_time_s1 = time.time()
    X_train_original_tensor, y_train_original_multiclass_tensor, \
    X_test_original_tensor, y_test_original_multiclass_tensor, \
    NUM_FEATURES_PREPROCESSED = \
        load_and_preprocess_nsl_kdd(cfg.BASE_DATA_PATH, cfg.TRAIN_FILENAME, cfg.TEST_FILENAME)
    duration_s1 = time.time() - start_time_s1
    step_timings['Data Loading & Preprocessing'] = format_time(duration_s1)
    print(f"--- Finished: Data Loading & Preprocessing in {format_time(duration_s1)} ---")

    # --- Step 2: Class-wise BEGAN Training and Data Augmentation (Conditional) ---
    X_train_final_tensor_cpu_list = [X_train_original_tensor.cpu()]
    y_train_final_multiclass_tensor_cpu_list = [y_train_original_multiclass_tensor.cpu()] # Store multi-class labels


    if args.use_gan:
        print("\n--- Starting: Class-wise BEGAN Training & Augmentation ---")
        began_total_start_time = time.time()
        unique_classes = torch.unique(y_train_original_multiclass_tensor.squeeze().cpu()).numpy()
        for class_label_val in unique_classes:
            class_label = int(class_label_val)
            class_name_str = f"Class_{class_label}"
            print(f"\nProcessing BEGAN for {class_name_str}")

            class_indices = (y_train_original_multiclass_tensor.squeeze().cpu() == class_label_val)
            X_class_data_cpu = X_train_original_tensor.cpu()[class_indices]

            if len(X_class_data_cpu) < cfg.GAN_BATCH_SIZE:
                print(f"  Skipping BEGAN for {class_name_str}: Insufficient samples ({len(X_class_data_cpu)}). Needs at least {cfg.GAN_BATCH_SIZE}.")
                continue
            
            class_dataset = NSLKDDDataset(X_class_data_cpu) # Features only
            class_dataloader = DataLoader(class_dataset, batch_size=cfg.GAN_BATCH_SIZE, shuffle=True, drop_last=True)

            g_began_class = Generator(cfg.BEGAN_NOISE_DIM, cfg.BEGAN_GENERATOR_HIDDEN_DIM, NUM_FEATURES_PREPROCESSED)
            d_began_class_ae = Autoencoder(NUM_FEATURES_PREPROCESSED, cfg.AE_LATENT_DIM, cfg.AE_HIDDEN_DIM_1)
            
            trained_g_class = train_began_for_class(
                g_began_class, d_began_class_ae, class_dataloader,
                cfg.BEGAN_NOISE_DIM, cfg.BEGAN_MAX_EPOCHS_PER_CLASS, cfg.DEVICE, class_name_str,
                cfg.BEGAN_GAMMA, cfg.BEGAN_LAMBDA_K, cfg.BEGAN_K_T_INITIAL, cfg.BEGAN_M_THRESHOLD, cfg.BEGAN_LR
            )
            
            num_to_generate_for_class = cfg.NUM_SYNTHETIC_SAMPLES_PER_CLASS
            
            if num_to_generate_for_class > 0 and trained_g_class is not None:
                 print(f"  Generating {num_to_generate_for_class} synthetic samples for {class_name_str}...")
                 synthetic_class_features_cpu = augment_data_with_gan(
                     trained_g_class, cfg.BEGAN_NOISE_DIM, 
                     num_to_generate_for_class, cfg.DEVICE, cfg.GAN_BATCH_SIZE
                 )
                 if synthetic_class_features_cpu.size(0) > 0:
                     synthetic_class_labels_cpu = torch.full(
                        (synthetic_class_features_cpu.size(0), 1), 
                        float(class_label), dtype=torch.long # Ensure it's long for potential CrossEntropy
                    )
                     X_train_final_tensor_cpu_list.append(synthetic_class_features_cpu)
                     y_train_final_multiclass_tensor_cpu_list.append(synthetic_class_labels_cpu)
                     print(f"    Added {synthetic_class_features_cpu.size(0)} synthetic samples for {class_name_str} (Target: {num_to_generate_for_class}).")
                 else:
                    print(f"    No synthetic samples generated for {class_name_str} despite target > 0.")
            elif trained_g_class is None:
                print(f"  Skipping synthetic sample generation for {class_name_str} (BEGAN training might have failed or generator is invalid).")
            else: # num_to_generate_for_class is 0
                print(f"  Skipping synthetic sample generation for {class_name_str} (num_to_generate is 0 based on config).")
        
        duration_s2 = time.time() - began_total_start_time
        step_timings['Class-wise BEGAN Training & Augmentation'] = format_time(time.time() - began_total_start_time)
        print(f"--- Finished: Class-wise BEGAN Augmentation in {format_time(duration_s2)} ---")
    else:
        print("\n--- Skipping BEGAN-based Data Augmentation (Launch parameter --use_gan not set) ---")
        step_timings['Class-wise BEGAN Training & Augmentation'] = "Skipped (Launch Param)"

    X_train_final_tensor = torch.cat(X_train_final_tensor_cpu_list, dim=0)
    y_train_final_multiclass_tensor = torch.cat(y_train_final_multiclass_tensor_cpu_list, dim=0)
    
    shuffle_indices = torch.randperm(X_train_final_tensor.size(0))
    X_train_final_tensor = X_train_final_tensor[shuffle_indices]
    y_train_final_multiclass_tensor = y_train_final_multiclass_tensor[shuffle_indices]
    print(f"Final training data (multi-class labels) shape: X={X_train_final_tensor.shape}, y={y_train_final_multiclass_tensor.shape}")

    # --- Prepare labels for Downstream Classifier based on classifier_mode ---
    num_classes_for_cnn = 0
    y_train_for_cnn_tensor = None
    y_test_for_cnn_tensor = None
    target_names_for_report = None

    if args.classifier_mode == "binary":
        num_classes_for_cnn = 1
        y_train_for_cnn_tensor = (y_train_final_multiclass_tensor.squeeze() != 0).float().unsqueeze(1) # Normal (0) vs Attack (1)
        y_test_for_cnn_tensor = (y_test_original_multiclass_tensor.squeeze() != 0).float().unsqueeze(1)
        target_names_for_report = ['Normal (Class 0)', 'Abnormal (Class 1)']
        print(f"Classifier mode: Binary. Target labels prepared.")
    elif args.classifier_mode == "multiclass":
        num_classes_for_cnn = cfg.NUM_ORIGINAL_CLASSES
        y_train_for_cnn_tensor = y_train_final_multiclass_tensor.squeeze().long() # Use original multi-class labels, ensure long type
        y_test_for_cnn_tensor = y_test_original_multiclass_tensor.squeeze().long()
        target_names_for_report = [cfg.NSL_KDD_CLASS_NAMES_INT_TO_STR[i] for i in range(cfg.NUM_ORIGINAL_CLASSES)]
        print(f"Classifier mode: Multi-class ({num_classes_for_cnn} classes). Target labels prepared.")
    else:
        raise ValueError(f"Invalid classifier_mode: {args.classifier_mode}")
    
    print(f"Shape of training labels for CNN: {y_train_for_cnn_tensor.shape}")
    print(f"Shape of test labels for CNN: {y_test_for_cnn_tensor.shape}")
    

    # --- Step 3: Feature Extraction Autoencoder ---
    current_extracted_features_file = cfg.EXTRACTED_FEATURES_FILE_GAN_AUGMENTED if args.use_gan else cfg.EXTRACTED_FEATURES_FILE_ORIGINAL
    X_train_features_ae, X_test_features_ae = None, None
    actual_latent_dim_for_cnn = cfg.AE_LATENT_DIM # Default

    if os.path.exists(current_extracted_features_file):
        print(f"\n--- Loading pre-extracted AE features from {current_extracted_features_file} ---")
        start_time_s3load = time.time()
        try:
            loaded_data = torch.load(current_extracted_features_file, map_location=cfg.DEVICE)
            X_train_features_ae = loaded_data['train_features'].to(cfg.DEVICE)
            X_test_features_ae = loaded_data['test_features'].to(cfg.DEVICE)
            actual_latent_dim_for_cnn = loaded_data.get('latent_dim', X_train_features_ae.shape[1])
            if actual_latent_dim_for_cnn != cfg.AE_LATENT_DIM:
                 print(f"  Info: Loaded features have latent_dim {actual_latent_dim_for_cnn}, global AE_LATENT_DIM is {cfg.AE_LATENT_DIM}. Using {actual_latent_dim_for_cnn}.")
            duration_s3load = time.time() - start_time_s3load
            step_timings['AE Feature Loading'] = format_time(duration_s3load)
            step_timings['Feature Extraction AE Training'] = "Skipped (loaded features)"
            step_timings['AE Feature Extraction (Train/Test)'] = "Skipped (loaded features)"
            print(f"  Loaded features: Train shape {X_train_features_ae.shape}, Test shape {X_test_features_ae.shape}")
        except Exception as e:
            print(f"  Error loading features from {current_extracted_features_file}: {e}. Will proceed to train AE and extract.")
            X_train_features_ae = None 
            
    if X_train_features_ae is None:
        print("\n--- Starting: Feature Extraction AE Training (on final train data) ---")
        start_time_s3train_ae = time.time()
        feature_extraction_ae = Autoencoder(NUM_FEATURES_PREPROCESSED, cfg.AE_LATENT_DIM, cfg.AE_HIDDEN_DIM_1)
        
        ae_train_dataset_for_fe = NSLKDDDataset(X_train_final_tensor.to(cfg.DEVICE)) # Use final (possibly augmented) X_train
        ae_train_loader_for_fe = DataLoader(ae_train_dataset_for_fe, batch_size=cfg.AE_BATCH_SIZE, shuffle=True)
        
        ae_early_stopper.reset() # Ensure stopper is fresh
        train_autoencoder(feature_extraction_ae, ae_train_loader_for_fe, cfg.DEFAULT_EPOCHS, 
                          cfg.AE_LEARNING_RATE, cfg.DEVICE, early_stopper=ae_early_stopper)
        duration_s3train_ae = time.time() - start_time_s3train_ae
        step_timings['Feature Extraction AE Training'] = format_time(duration_s3train_ae)

        print("\n--- Starting: AE Feature Extraction (Train/Test) ---")
        start_time_s3extract = time.time()
        # For train features (from X_train_final_tensor)
        ae_train_extract_loader = DataLoader(NSLKDDDataset(X_train_final_tensor), batch_size=cfg.AE_BATCH_SIZE, shuffle=False)
        X_train_features_ae = extract_ae_features(feature_extraction_ae, ae_train_extract_loader, cfg.DEVICE, cfg.AE_BATCH_SIZE)
        
        # For test features (from original X_test_original_tensor)
        ae_test_extract_loader = DataLoader(NSLKDDDataset(X_test_original_tensor), batch_size=cfg.AE_BATCH_SIZE, shuffle=False)
        X_test_features_ae = extract_ae_features(feature_extraction_ae, ae_test_extract_loader, cfg.DEVICE, cfg.AE_BATCH_SIZE)
        
        actual_latent_dim_for_cnn = X_train_features_ae.shape[1] if X_train_features_ae.nelement() > 0 else cfg.AE_LATENT_DIM
        duration_s3extract = time.time() - start_time_s3extract
        step_timings['AE Feature Extraction (Train/Test)'] = format_time(duration_s3extract)
        
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
    
    X_train_features_ae = X_train_features_ae.to(cfg.DEVICE)
    X_test_features_ae = X_test_features_ae.to(cfg.DEVICE)
    print(f"AE features ready for CNN: Train shape {X_train_features_ae.shape}, Test shape {X_test_features_ae.shape}")
    print(f"Using latent dimension for CNN: {actual_latent_dim_for_cnn}")


    # --- Step 4: CNN Classifier Training ---
    # Ensure labels are on device for DataLoader
    cnn_train_dataset = NSLKDDDataset(X_train_features_ae, y_train_for_cnn_tensor.to(cfg.DEVICE))
    cnn_train_loader = DataLoader(cnn_train_dataset, batch_size=cfg.CNN_BATCH_SIZE, shuffle=True)
    cnn_test_dataset = NSLKDDDataset(X_test_features_ae, y_test_for_cnn_tensor.to(cfg.DEVICE))
    cnn_test_loader = DataLoader(cnn_test_dataset, batch_size=cfg.CNN_BATCH_SIZE, shuffle=False)

    cnn_model = CNNClassifier(
        input_dim_cnn=actual_latent_dim_for_cnn, # This should be correctly determined
        cnn_filters=cfg.CNN_FILTERS, cnn_kernel_size=cfg.CNN_KERNEL_SIZE,
        cnn_pool_size=cfg.CNN_POOL_SIZE, cnn_pool_stride=cfg.CNN_POOL_STRIDE,
        cnn_fc_neurons=cfg.CNN_FC_NEURONS, 
        num_classes=num_classes_for_cnn # Dynamic based on arg
    )
    
    print(f"\n--- Starting: CNN Classifier Training & Validation (Mode: {args.classifier_mode}, Classes: {num_classes_for_cnn}) ---")
    start_time_s4 = time.time()
    cnn_early_stopper.reset()
    trained_cnn_model = train_classifier(
        cnn_model, cnn_train_loader, cnn_test_loader, cfg.CNN_EPOCHS, 
        cfg.CNN_LEARNING_RATE, cfg.DEVICE, 
        num_classes=num_classes_for_cnn, # Pass dynamic num_classes
        early_stopper=cnn_early_stopper,
        target_names_report=target_names_for_report # Pass dynamic target names
    )
    duration_s4 = time.time() - start_time_s4
    step_timings['CNN Training & Validation'] = format_time(duration_s4)
    print(f"--- Finished: CNN Training & Validation in {format_time(duration_s4)} ---")


    # --- Step 5: Final Evaluation and Reporting ---
    print("\n--- Starting: Final CNN Evaluation on Test Data ---")
    start_time_s5 = time.time()
    trained_cnn_model.eval()
    all_final_preds_list, all_final_labels_list = [], []
    
    # Ensure cnn_test_loader is correctly defined with X_test_features_ae and y_test_binary_for_cnn_tensor
    with torch.no_grad():
        for features, labels in cnn_test_loader:
            features = features.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE) # labels shape (B,1) for binary, (B,) for multiclass
            outputs = trained_cnn_model(features) # outputs shape (B,1) for binary, (B, num_classes) for multiclass

            if num_classes_for_cnn == 1: # num_classes_for_cnn is set based on args.classifier_mode
                predicted_final = (torch.sigmoid(outputs) > 0.5).float() # Shape (B, 1)
            else: # For multi-class
                _, predicted_final = torch.max(outputs.data, 1) # Shape (B,)
            
            # Robustly flatten to 1D before converting to numpy and extending
            all_final_preds_list.extend(predicted_final.view(-1).cpu().numpy())
            all_final_labels_list.extend(labels.view(-1).cpu().numpy()) # .view(-1) handles both (B,1) and (B,) correctly


    final_report_dict = {}
    # final_report_dict generation using target_names_for_report
    if len(all_final_labels_list) > 0 and len(all_final_preds_list) > 0 :
        final_report_dict = classification_report(
            all_final_labels_list, all_final_preds_list, 
            target_names=target_names_for_report, # Use dynamic target names
            output_dict=True, zero_division=0
        )
    else:
        print("Warning: No data/predictions in final test evaluation to generate report.")

    duration_s5 = time.time() - start_time_s5
    step_timings['Final CNN Evaluation'] = format_time(duration_s5)
    print(f"--- Finished: Final CNN Evaluation in {format_time(duration_s5)} ---")
    
    print("\n--- Final Test Performance Report (Percentage Format) ---")
    if not final_report_dict:
        print("No evaluation data to report.")
    else:
        if 'accuracy' in final_report_dict: print(f"Overall Accuracy: {final_report_dict['accuracy']*100:.2f}%")
        else: print("Overall Accuracy: N/A")
        print("-" * 70) 
        for class_name_key in target_names_for_report: # Iterate using the provided target names
            if class_name_key in final_report_dict:
                metrics = final_report_dict[class_name_key]
                print(f"Class: {class_name_key}")
                print(f"  Recall: {metrics.get('recall',0)*100:.2f}% | Precision: {metrics.get('precision',0)*100:.2f}% | "
                      f"F1-Score: {metrics.get('f1-score',0)*100:.2f}% | Support: {metrics.get('support',0)}")
                print("-" * 40)
        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in final_report_dict:
                metrics = final_report_dict[avg_type]
                print(f"{avg_type.replace('avg', 'Average').title()}:")
                print(f"  Recall: {metrics.get('recall',0)*100:.2f}% | Precision: {metrics.get('precision',0)*100:.2f}% | "
                      f"F1-Score: {metrics.get('f1-score',0)*100:.2f}%")
                if 'support' in metrics and metrics['support'] is not None : print(f"  Support: {metrics['support']}")
                print("-" * 40)
            
    table_output_file = f"{cfg.TABLE_OUTPUT_FILENAME_PREFIX}_{'gan' if args.use_gan else 'orig'}_{args.classifier_mode}.png"
    save_results_table(final_report_dict, step_timings, 
                       output_path=table_output_file, 
                       classifier_mode=args.classifier_mode, 
                       class_names=target_names_for_report if args.classifier_mode == "multiclass" else None)

    print("\nProcess finished.")
    print("\n--- Summary of Step Timings ---")
    for step, time_taken in step_timings.items():
        print(f"Time for {step}: {time_taken}")
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NSL-KDD Intrusion Detection with PyTorch, AE, CNN, and optional BEGAN augmentation.")
    parser.add_argument('--use_gan', action='store_true', help="Enable GAN (BEGAN) for data augmentation.")
    parser.add_argument('--classifier_mode', type=str, default='binary', choices=['binary', 'multiclass'],
                        help="Set the classifier mode: 'binary' (Normal vs Attack) or 'multiclass' (Normal, DoS, Probe, etc.). Default: binary.")
    
    cli_args = parser.parse_args()
    main(cli_args)