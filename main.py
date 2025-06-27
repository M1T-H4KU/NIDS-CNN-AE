# main.py
import argparse
import sys
import time
import os

def main(args):
    """
    Main execution function that runs the entire pipeline based on command-line arguments.
    """
    
    # --- Step 0: Imports and Initial Setup ---
    print("Imports loaded. Starting main execution...")
    
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import classification_report
    
    import configs as cfg
    from utils import ValidationLossEarlyStopper, format_time, save_results_table, display_data_samples
    from data_loader import NSLKDDDataset, load_and_preprocess_nsl_kdd
    from models import Autoencoder, Generator, CNNClassifier
    from training_logic import (
        train_autoencoder, train_began_for_class, train_classifier,
        extract_ae_features, augment_data_with_gan
    )
    
    
    print(f"Using device: {cfg.DEVICE}")
    print(f"Running with arguments: GAN Augmentation={'Enabled' if args.use_gan else 'Disabled'}, "
          f"Outlier Removal={'Enabled' if args.use_outlier_removal else 'Disabled'}, "
          f"Classifier Mode='{args.classifier_mode}', Model Pipeline='{args.model}'")
    step_timings = {}
    
    # Initialize early stoppers that monitor validation loss
    ae_early_stopper = ValidationLossEarlyStopper(patience=cfg.EARLY_STOPPING_PATIENCE, min_delta=cfg.EARLY_STOPPING_MIN_DELTA_VAL)
    classifier_early_stopper = ValidationLossEarlyStopper(patience=cfg.EARLY_STOPPING_PATIENCE, min_delta=cfg.EARLY_STOPPING_MIN_DELTA_VAL)

    # --- Step 1: Load and preprocess original data ---
    if args.classifier_mode == 'binary':
        train_file, test_file = cfg.TRAIN_FILENAME_ARFF, cfg.TEST_FILENAME_ARFF
    else: # 'multiclass'
        train_file, test_file = cfg.TRAIN_FILENAME_TXT, cfg.TEST_FILENAME_TXT
    
    print("\n--- Starting: Data Loading & Preprocessing ---")
    start_time_s1 = time.time()
    X_train_original_tensor, y_train_original_labels_tensor, \
    X_test_original_tensor, y_test_original_labels_tensor, \
    NUM_FEATURES_PREPROCESSED = \
        load_and_preprocess_nsl_kdd(cfg.BASE_DATA_PATH, train_file, test_file, perform_outlier_removal=args.use_outlier_removal)
    duration_s1 = time.time() - start_time_s1
    step_timings['Data Loading & Preprocessing'] = format_time(duration_s1)
    print(f"--- Finished: Data Loading & Preprocessing in {format_time(duration_s1)} ---")
    print("\n--- Verifying Original Preprocessed Data ---")
    # Determine the class names based on the mode data was loaded for
    is_binary_data = train_file.lower().endswith('.arff')
    class_map_for_display = {0: 'Normal', 1: 'Abnormal'} if is_binary_data else cfg.NSL_KDD_CLASS_NAMES_INT_TO_STR
    
    for class_label, class_name in class_map_for_display.items():
        class_indices = (y_train_original_labels_tensor.squeeze() == class_label)
        data_for_class = X_train_original_tensor[class_indices]
        display_data_samples(data_for_class, f"Original '{class_name}' Data (Class {class_label})")
        
        
    # --- Step 2: Class-wise BEGAN Training and Data Augmentation ---
    X_train_final_tensor = X_train_original_tensor
    y_train_final_labels_tensor = y_train_original_labels_tensor
    if args.use_gan:
        print("\n--- Starting: BEGAN Data Augmentation ---")
        began_total_start_time = time.time()
        
        X_train_final_tensor_cpu_list = [X_train_original_tensor.cpu()]
        y_train_final_labels_cpu_list = [y_train_original_labels_tensor.cpu()]
        
        # <<< FIX STARTS HERE: Conditional GAN Logic >>>
        classes_to_augment, num_samples_to_generate, class_name_map = [], 0, {}
        
        if args.classifier_mode == 'multiclass':
            print("Applying MULTI-CLASS augmentation strategy (minority classes only).")
            classes_to_augment = cfg.NSL_KDD_MINORITY_CLASSES_TO_AUGMENT
            num_samples_to_generate = cfg.NUM_SYNTHETIC_SAMPLES_PER_MINORITY_CLASS
            class_name_map = cfg.NSL_KDD_CLASS_NAMES_INT_TO_STR
        else: # binary
            print("Applying BINARY augmentation strategy (all classes found).")
            classes_to_augment = torch.unique(y_train_original_labels_tensor.squeeze().cpu()).numpy()
            num_samples_to_generate = cfg.NUM_SYNTHETIC_SAMPLES_PER_CLASS
            class_name_map = {0: 'Normal', 1: 'Abnormal'}
        
        existing_classes_in_data = torch.unique(y_train_original_labels_tensor.squeeze().cpu()).numpy()

        for class_label in classes_to_augment:
            if int(class_label) not in existing_classes_in_data:
                print(f"\nSkipping BEGAN for Class_{int(class_label)}: Not present in the loaded training data.")
                continue

            class_name_str = f"Class_{int(class_label)} ({class_name_map.get(int(class_label), 'Unknown')})"
            print(f"\nProcessing BEGAN for {class_name_str}")

            class_indices = (y_train_original_labels_tensor.squeeze().cpu() == class_label)
            X_class_data_cpu = X_train_original_tensor.cpu()[class_indices]

            effective_batch_size = min(len(X_class_data_cpu), cfg.GAN_BATCH_SIZE)
            if effective_batch_size == 0:
                print(f"  Skipping BEGAN for {class_name_str}: No data samples.")
                continue

            class_dataset = NSLKDDDataset(X_class_data_cpu)
            class_dataloader = DataLoader(class_dataset, batch_size=effective_batch_size, shuffle=True, drop_last=False)
            
            if len(class_dataloader) == 0:
                print(f"  Skipping BEGAN for {class_name_str}: DataLoader is empty.")
                continue

            g_began_class = Generator(cfg.BEGAN_NOISE_DIM, cfg.BEGAN_GENERATOR_HIDDEN_DIM, NUM_FEATURES_PREPROCESSED)
            d_began_class_ae = Autoencoder(NUM_FEATURES_PREPROCESSED, cfg.AE_LATENT_DIM, cfg.AE_HIDDEN_DIM_1)
            
            trained_g_class = train_began_for_class(
                g_began_class, d_began_class_ae, class_dataloader,
                cfg.BEGAN_NOISE_DIM, cfg.BEGAN_MAX_EPOCHS_PER_CLASS, cfg.DEVICE, class_name_str,
                cfg.BEGAN_GAMMA, cfg.BEGAN_LAMBDA_K, cfg.BEGAN_K_T_INITIAL, cfg.BEGAN_M_THRESHOLD, cfg.BEGAN_LR
            )
            
            if num_samples_to_generate > 0 and trained_g_class is not None:
                print(f"  Generating {num_samples_to_generate} synthetic samples for {class_name_str}...")
                synthetic_class_features_cpu = augment_data_with_gan(trained_g_class, cfg.BEGAN_NOISE_DIM, num_samples_to_generate, cfg.DEVICE, cfg.GAN_BATCH_SIZE)
                display_data_samples(synthetic_class_features_cpu, f"Generated '{class_name_str}' Data")
                if synthetic_class_features_cpu.size(0) > 0:
                    synthetic_class_labels_cpu = torch.full((synthetic_class_features_cpu.size(0), 1), float(class_label), dtype=torch.long)
                    X_train_final_tensor_cpu_list.append(synthetic_class_features_cpu)
                    y_train_final_labels_cpu_list.append(synthetic_class_labels_cpu)
                    print(f"    Added {synthetic_class_features_cpu.size(0)} synthetic samples for {class_name_str} (Target: {num_samples_to_generate}).")
        
        step_timings['BEGAN Augmentation'] = format_time(time.time() - began_total_start_time)
        X_train_final_tensor = torch.cat(X_train_final_tensor_cpu_list, dim=0)
        y_train_final_labels_tensor = torch.cat(y_train_final_labels_cpu_list, dim=0)
        # <<< FIX ENDS HERE >>>
    else:
        print("\n--- Skipping BEGAN-based Data Augmentation ---")
        step_timings['BEGAN Augmentation'] = "Skipped"

    # Shuffle the final combined training dataset
    shuffle_indices = torch.randperm(X_train_final_tensor.size(0))
    X_train_final_tensor, y_train_final_labels_tensor = X_train_final_tensor[shuffle_indices], y_train_final_labels_tensor[shuffle_indices]
    print(f"Final training data shape: X={X_train_final_tensor.shape}, y={y_train_final_labels_tensor.shape}")

    # --- Prepare labels for the selected classifier mode ---
    if args.classifier_mode == "binary":
        num_classes_for_cnn = 1
        y_train_for_cnn_tensor = (y_train_final_labels_tensor.squeeze() != 0).float().unsqueeze(1)
        y_test_for_cnn_tensor = (y_test_original_labels_tensor.squeeze() != 0).float().unsqueeze(1)
        target_names_for_report = ['Normal (Class 0)', 'Abnormal (Class 1)']
        print(f"Classifier mode: Binary. Target labels prepared.")
    else: # multiclass
        num_classes_for_cnn = cfg.NUM_ORIGINAL_CLASSES
        y_train_for_cnn_tensor = y_train_final_labels_tensor.squeeze().long()
        y_test_for_cnn_tensor = y_test_original_labels_tensor.squeeze().long()
        target_names_for_report = [cfg.NSL_KDD_CLASS_NAMES_INT_TO_STR[i] for i in range(cfg.NUM_ORIGINAL_CLASSES)]
        print(f"Classifier mode: Multi-class ({num_classes_for_cnn} classes). Target labels prepared.")

    # --- MODEL PIPELINE SELECTION ---
    X_train_for_cnn, X_test_for_cnn, input_dim_for_cnn = None, None, 0
    actual_latent_dim_for_cnn = cfg.AE_LATENT_DIM

    # 'ae' logic for models that use the autoencoder
    if 'ae' in args.model:
        print(f"\n--- Model Pipeline: {args.model.upper()} (Autoencoder for Feature Extraction) ---")
        current_extracted_features_file = cfg.EXTRACTED_FEATURES_FILE_GAN_AUGMENTED if args.use_gan else cfg.EXTRACTED_FEATURES_FILE_ORIGINAL
        
        X_train_features_ae, X_test_features_ae = None, None
        if os.path.exists(current_extracted_features_file):
            print(f"\n--- Loading pre-extracted AE features from {current_extracted_features_file} ---")
            start_time_s3load = time.time()
            try:
                loaded_data = torch.load(current_extracted_features_file, map_location=cfg.DEVICE)
                if loaded_data['train_features'].shape[0] != len(X_train_final_tensor):
                    raise ValueError(f"Stale cache detected. Expected {len(X_train_final_tensor)} samples, but cache has {loaded_data['train_features'].shape[0]}.")
                
                print(f"  Cache file validated. Loading features.")
                X_train_features_ae, X_test_features_ae = loaded_data['train_features'].to(cfg.DEVICE), loaded_data['test_features'].to(cfg.DEVICE)
                actual_latent_dim_for_cnn = loaded_data.get('latent_dim', X_train_features_ae.shape[1])
                step_timings['AE Feature Loading'] = format_time(time.time() - start_time_s3load)
                step_timings['Feature Extraction AE Training'] = "Skipped (loaded features)"
            except Exception as e:
                print(f"  Warning: {e}. Re-training AE.")
                X_train_features_ae = None
        
        if X_train_features_ae is None:
            print("\n--- Starting: Feature Extraction AE Training ---")
            start_time_s3train_ae = time.time()
            feature_extraction_ae = Autoencoder(NUM_FEATURES_PREPROCESSED, cfg.AE_LATENT_DIM, cfg.AE_HIDDEN_DIM_1)
            ae_train_dataset_for_fe = NSLKDDDataset(X_train_final_tensor)
            ae_train_loader_for_fe = DataLoader(ae_train_dataset_for_fe, batch_size=cfg.AE_BATCH_SIZE, shuffle=True)
            ae_early_stopper.reset()
            train_autoencoder(feature_extraction_ae, ae_train_loader_for_fe, cfg.DEFAULT_EPOCHS, cfg.AE_LEARNING_RATE, cfg.DEVICE, early_stopper=ae_early_stopper)
            step_timings['Feature Extraction AE Training'] = format_time(time.time() - start_time_s3train_ae)

            print("\n--- Starting: AE Feature Extraction ---")
            start_time_s3extract = time.time()
            ae_train_extract_loader, ae_test_extract_loader = DataLoader(NSLKDDDataset(X_train_final_tensor), batch_size=cfg.AE_BATCH_SIZE, shuffle=False), DataLoader(NSLKDDDataset(X_test_original_tensor), batch_size=cfg.AE_BATCH_SIZE, shuffle=False)
            X_train_features_ae, X_test_features_ae = extract_ae_features(feature_extraction_ae, ae_train_extract_loader, cfg.DEVICE, cfg.AE_BATCH_SIZE), extract_ae_features(feature_extraction_ae, ae_test_extract_loader, cfg.DEVICE, cfg.AE_BATCH_SIZE)
            actual_latent_dim_for_cnn = X_train_features_ae.shape[1] if X_train_features_ae.nelement() > 0 else cfg.AE_LATENT_DIM
            step_timings['AE Feature Extraction (Train/Test)'] = format_time(time.time() - start_time_s3extract)
            
            print(f"  Saving AE features to {current_extracted_features_file}...")
            torch.save({'train_features': X_train_features_ae.cpu(), 'test_features': X_test_features_ae.cpu(), 'latent_dim': actual_latent_dim_for_cnn}, current_extracted_features_file)

        X_train_for_cnn, X_test_for_cnn, input_dim_for_cnn = X_train_features_ae, X_test_features_ae, actual_latent_dim_for_cnn
    
    # Logic for models that do not use the autoencoder
    else:
        print(f"\n--- Model Pipeline: {args.model.upper()} (Directly on preprocessed data) ---")
        step_timings['Feature Extraction AE Training'] = f"Skipped (Direct {args.model.upper()} Model)"
        X_train_for_cnn, X_test_for_cnn = X_train_final_tensor, X_test_original_tensor
        input_dim_for_cnn = NUM_FEATURES_PREPROCESSED
    
    # --- Step 4: Classifier Training ---
    print(f"\nData ready for Classifier. Input Dim: {input_dim_for_cnn}, Train Shape: {X_train_for_cnn.shape}, Test Shape: {X_test_for_cnn.shape}")
    cnn_train_dataset = NSLKDDDataset(X_train_for_cnn, y_train_for_cnn_tensor.to(cfg.DEVICE))
    cnn_train_loader = DataLoader(cnn_train_dataset, batch_size=cfg.CNN_BATCH_SIZE, shuffle=True)
    cnn_test_dataset = NSLKDDDataset(X_test_for_cnn, y_test_for_cnn_tensor.to(cfg.DEVICE))
    cnn_test_loader = DataLoader(cnn_test_dataset, batch_size=cfg.CNN_BATCH_SIZE, shuffle=False)

    print(f"\n--- Instantiating Classifier Model: {args.model.upper()} ---")
    classifier_model = None
    if args.model in ['cnn', 'cnnae']:
        classifier_model = CNNClassifier(
            input_dim_cnn=input_dim_for_cnn, cnn_filters=cfg.CNN_FILTERS, cnn_kernel_size=cfg.CNN_KERNEL_SIZE,
            cnn_pool_size=cfg.CNN_POOL_SIZE, cnn_pool_stride=cfg.CNN_POOL_STRIDE,
            cnn_fc_neurons=cfg.CNN_FC_NEURONS, num_classes=num_classes_for_cnn
        )
    elif args.model in ['dnn', 'dnnae', 'lstm', 'lstmae']:
        print(f"Error: Model '{args.model}' is not yet implemented. Exiting.")
        sys.exit(1)
    
    classifier_model.to(cfg.DEVICE)
    print("Model Architecture:"); print(classifier_model)
    
    print(f"\n--- Starting: Classifier Training & Validation ({args.model.upper()}) ---")
    start_time_s4 = time.time()
    classifier_early_stopper.reset()
    trained_classifier_model = train_classifier(
        classifier_model, cnn_train_loader, cnn_test_loader, cfg.CNN_EPOCHS, 
        cfg.CNN_LEARNING_RATE, cfg.DEVICE, 
        num_classes=num_classes_for_cnn, early_stopper=classifier_early_stopper,
        target_names_report=target_names_for_report
    )
    step_timings['Classifier Training & Validation'] = format_time(time.time() - start_time_s4)

    # --- Step 5: Final Evaluation and Reporting ---
    print("\n--- Starting: Final CNN Evaluation on Test Data ---")
    start_time_s5 = time.time()
    trained_classifier_model.eval()
    all_final_preds_list, all_final_labels_list = [], []
    with torch.no_grad():
        for features, labels in cnn_test_loader:
            features, labels = features.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            outputs = trained_classifier_model(features)
            if num_classes_for_cnn == 1:
                predicted_final = (torch.sigmoid(outputs) > 0.5).float()
            else:
                _, predicted_final = torch.max(outputs.data, 1)
            all_final_preds_list.extend(predicted_final.view(-1).cpu().numpy())
            all_final_labels_list.extend(labels.view(-1).cpu().numpy())

    final_report_dict = {}
    if all_final_labels_list:
        final_report_dict = classification_report(
            all_final_labels_list, all_final_preds_list, 
            target_names=target_names_for_report, output_dict=True, zero_division=0
        )
    step_timings['Final CNN Evaluation'] = format_time(time.time() - start_time_s5)
    
    print("\n--- Final Test Performance Report (Percentage Format) ---")
    if 'accuracy' in final_report_dict: print(f"Overall Accuracy: {final_report_dict['accuracy']*100:.2f}%")
    else: print("Overall Accuracy: N/A")
    print("-" * 70)
    for class_name_key in target_names_for_report:
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
            if 'support' in metrics: print(f"  Support: {metrics.get('support',0)}")
            print("-" * 40)
            
    table_output_file = f"{cfg.TABLE_OUTPUT_FILENAME_PREFIX}_{args.model}_{'gan' if args.use_gan else 'orig'}_{'outlier' if args.use_outlier_removal else 'no_outlier'}_{args.classifier_mode}.png"
    save_results_table(final_report_dict, step_timings, 
                       output_path=table_output_file, 
                       classifier_mode=args.classifier_mode, 
                       class_names=target_names_for_report)
    
    print("\nProcess finished.")
    print("\n--- Summary of Step Timings ---")
    for step, time_taken in step_timings.items():
        print(f"Time for {step}: {time_taken}")

# --- Main entry point with command-line argument parsing ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="NSL-KDD Intrusion Detection with PyTorch, AE, CNN, and optional BEGAN augmentation.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    parser.add_argument('--use_outlier_removal', action='store_true', help="Enable MAD-based outlier removal.\n(Default: Disabled).")
    parser.add_argument('--use_gan', action='store_true', help="Enable BEGAN for data augmentation.\n(Default: Disabled).")
    parser.add_argument('--classifier_mode', type=str, required=True, choices=['binary', 'multiclass'], help="Set the classifier mode (REQUIRED).")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['cnn', 'dnn', 'lstm', 'cnnae', 'dnnae', 'lstmae'],
        help="Select the model pipeline to use (REQUIRED):\n"
             "  'cnn':   Directly train CNN on preprocessed data.\n"
             "  'cnnae': Use Autoencoder for feature extraction before CNN.\n"
             "  'dnn':   Directly train DNN on preprocessed data.\n"
             "  'dnnae': Use Autoencoder for feature extraction before DNN.\n"
             "  'lstm':  Directly train LSTM on preprocessed data.\n"
             "  'lstmae': Use Autoencoder for feature extraction before LSTM."
    )

    cli_args = parser.parse_args()
    try:
        main(cli_args)
    except (SystemExit, KeyboardInterrupt):
        print("\nProcess interrupted by user. Exiting.")
    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()