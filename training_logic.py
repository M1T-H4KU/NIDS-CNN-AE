# training_logic.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

def train_autoencoder(model, dataloader, epochs, learning_rate, device, early_stopper=None):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {'loss': []}
    print(f"\n--- Training Feature Extraction Autoencoder on {device} ---")
    if early_stopper: early_stopper.reset()

    for epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0.0

        for batch_features in dataloader:
            if isinstance(batch_features, list) or isinstance(batch_features, tuple):
                batch_features = batch_features[0]
            batch_features = batch_features.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_features)
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item() * batch_features.size(0)
        
        avg_epoch_loss = 0.0
        if hasattr(dataloader, 'dataset') and len(dataloader.dataset) > 0:
            avg_epoch_loss = epoch_loss_sum / len(dataloader.dataset)
        elif len(dataloader) > 0: 
             avg_epoch_loss = epoch_loss_sum / len(dataloader)

        history['loss'].append(avg_epoch_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"AE Epoch [{epoch+1}/{epochs}], Training Loss: {avg_epoch_loss:.6f}")

        if early_stopper and early_stopper(avg_epoch_loss):
            break 
    print("Feature Extraction Autoencoder training finished.")
    return model, history

def train_began_for_class(
    generator, discriminator_ae, class_dataloader,
    noise_dim, epochs, device, class_name,
    gamma, lambda_k, k_t_initial, m_threshold, lr
):
    generator.to(device)
    discriminator_ae.to(device)
    ae_loss_fn = nn.L1Loss()

    optimizer_g = optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = optim.Adam(discriminator_ae.parameters(), lr=lr)
    k_t = k_t_initial
    
    print(f"\n--- Training BEGAN for Class: {class_name} on {device} ---")
    for epoch in range(epochs):
        generator.train()
        discriminator_ae.train()
        epoch_loss_d_real_sum, epoch_loss_d_fake_sum, epoch_loss_g_sum, epoch_m_sum = 0,0,0,0
        num_batches = 0

        for real_data_batch in class_dataloader:
            real_data_batch = real_data_batch.to(device)
            current_batch_size = real_data_batch.size(0)

            optimizer_d.zero_grad()
            reconstructed_real = discriminator_ae(real_data_batch)
            loss_d_real = ae_loss_fn(reconstructed_real, real_data_batch)
            noise = torch.randn(current_batch_size, noise_dim, device=device)
            fake_data_batch = generator(noise).detach()
            reconstructed_fake = discriminator_ae(fake_data_batch)
            loss_d_fake = ae_loss_fn(reconstructed_fake, fake_data_batch)
            loss_d = loss_d_real - k_t * loss_d_fake
            loss_d.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            noise_g = torch.randn(current_batch_size, noise_dim, device=device)
            fake_data_for_g = generator(noise_g)
            reconstructed_fake_for_g = discriminator_ae(fake_data_for_g)
            loss_g = ae_loss_fn(reconstructed_fake_for_g, fake_data_for_g)
            loss_g.backward()
            optimizer_g.step()

            balance = (gamma * loss_d_real.item() - loss_d_fake.item())
            k_t = np.clip(k_t + lambda_k * balance, 0, 1)
            current_m = loss_d_real.item() + np.abs(gamma * loss_d_real.item() - loss_g.item())

            epoch_loss_d_real_sum += loss_d_real.item()
            epoch_loss_d_fake_sum += loss_d_fake.item()
            epoch_loss_g_sum += loss_g.item()
            epoch_m_sum += current_m
            num_batches += 1
        
        if num_batches > 0:
            avg_loss_d_real = epoch_loss_d_real_sum / num_batches
            avg_loss_g = epoch_loss_g_sum / num_batches
            avg_m = epoch_m_sum / num_batches
            
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs -1:
                print(f"  BEGAN Epoch [{epoch+1}/{epochs}] for '{class_name}': M={avg_m:.4f}, L(x)={avg_loss_d_real:.4f}, "
                      f"L(G(z))={avg_loss_g:.4f}, k_t={k_t:.4f}")
            if avg_m < m_threshold and epoch > 10:
                print(f"BEGAN convergence for class '{class_name}' reached at M={avg_m:.4f} (Threshold: {m_threshold})")
                break

    print(f"BEGAN training for class '{class_name}' finished.")
    return generator

def train_cgan(generator, discriminator, dataloader, noise_dim, epochs, device, lr, beta1, beta2):
    """
    Trains a Conditional GAN (cGAN) on the provided dataset with labels.
    """
    generator.to(device)
    discriminator.to(device)
    
    # Binary Cross Entropy Loss (with logits for stability)
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    
    print(f"\n--- Training cGAN on {device} for {epochs} epochs ---")
    
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        
        loss_d_sum = 0.0
        loss_g_sum = 0.0
        num_batches = 0
        
        for real_features, real_labels in dataloader:
            real_features = real_features.to(device)
            real_labels = real_labels.to(device).long().squeeze()
            batch_size = real_features.size(0)
            
            # --- Train Discriminator ---
            optimizer_d.zero_grad()
            
            # Real Data
            valid_targets = torch.ones(batch_size, 1, device=device)
            pred_real = discriminator(real_features, real_labels)
            loss_real = criterion(pred_real, valid_targets)
            
            # Fake Data
            noise = torch.randn(batch_size, noise_dim, device=device)
            # We use the same labels as real data to condition the generator
            fake_features = generator(noise, real_labels)
            fake_targets = torch.zeros(batch_size, 1, device=device)
            pred_fake = discriminator(fake_features.detach(), real_labels)
            loss_fake = criterion(pred_fake, fake_targets)
            
            loss_d = (loss_real + loss_fake) / 2
            loss_d.backward()
            optimizer_d.step()
            
            # --- Train Generator ---
            optimizer_g.zero_grad()
            
            # Generator wants discriminator to predict 'valid' (1)
            pred_fake_for_g = discriminator(fake_features, real_labels)
            loss_g = criterion(pred_fake_for_g, valid_targets)
            
            loss_g.backward()
            optimizer_g.step()
            
            loss_d_sum += loss_d.item()
            loss_g_sum += loss_g.item()
            num_batches += 1
            
        if num_batches > 0:
            avg_loss_d = loss_d_sum / num_batches
            avg_loss_g = loss_g_sum / num_batches
            
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"  cGAN Epoch [{epoch+1}/{epochs}]: Loss D: {avg_loss_d:.4f}, Loss G: {avg_loss_g:.4f}")
                
    print("cGAN training finished.")
    return generator

def augment_data_with_gan(generator, noise_dim, num_samples, device, gan_batch_size):
    generator.eval()
    generated_data_list = []
    if num_samples == 0:
        out_features = generator.main[-2].out_features if hasattr(generator, 'main') and len(generator.main) > 2 else 0
        if out_features == 0: 
            print("Warning: Could not determine generator output features for empty tensor.")
            return torch.empty(0, 1, device='cpu')
        return torch.empty(0, out_features, device='cpu')

    with torch.no_grad():
        for _ in range(int(np.ceil(num_samples / gan_batch_size))):
            current_batch_gen_size = min(gan_batch_size, num_samples - len(generated_data_list))
            if current_batch_gen_size <= 0: break
            noise = torch.randn(current_batch_gen_size, noise_dim, device=device)
            synthetic_samples = generator(noise).cpu()
            generated_data_list.append(synthetic_samples)
    
    if not generated_data_list:
        out_features = generator.main[-2].out_features
        return torch.empty(0, out_features, device='cpu')
        
    generated_data = torch.cat(generated_data_list, dim=0)
    return generated_data[:num_samples]

def augment_data_with_cgan(generator, noise_dim, num_samples, target_label_int, device, gan_batch_size):
    """
    Generates synthetic data for a specific class using a trained Conditional Generator.
    """
    generator.eval()
    generated_data_list = []
    
    if num_samples == 0:
        return torch.empty(0, 1, device='cpu') # Placeholder return size might need adjustment

    with torch.no_grad():
        for _ in range(int(np.ceil(num_samples / gan_batch_size))):
            current_batch_gen_size = min(gan_batch_size, num_samples - len(generated_data_list))
            if current_batch_gen_size <= 0: break
            
            noise = torch.randn(current_batch_gen_size, noise_dim, device=device)
            labels = torch.full((current_batch_gen_size,), target_label_int, dtype=torch.long, device=device)
            
            synthetic_samples = generator(noise, labels).cpu()
            generated_data_list.append(synthetic_samples)
            
    if not generated_data_list:
        # Try to infer output dim from model structure (assuming Linear layer last before sigmoid)
        # cGAN model structure: model -> Sequential -> Linear ...
        try:
            out_features = generator.model[-2].out_features
        except:
            out_features = 1
        return torch.empty(0, out_features, device='cpu')

    generated_data = torch.cat(generated_data_list, dim=0)
    return generated_data[:num_samples]

def extract_ae_features(encoder_model, dataloader, device, batch_size):
    encoder_model.to(device)
    encoder_model.eval()
    all_features_list = []
    # MODIFIED: 确保我们使用的是传入的 dataloader
    temp_dataloader = dataloader 

    with torch.no_grad():
        for batch_data in temp_dataloader:
            if isinstance(batch_data, list) or isinstance(batch_data, tuple):
                batch_data = batch_data[0]
            batch_data = batch_data.to(device)
            encoded_features = encoder_model.encode(batch_data)
            all_features_list.append(encoded_features.cpu())
    if not all_features_list:
        print("Warning: No data processed during feature extraction.")
        try:
            out_dim = encoder_model.encoder[-2].out_features
        except:
            out_dim = 1 # Fallback
        return torch.empty(0, out_dim, device='cpu')
        
    return torch.cat(all_features_list, dim=0)

def train_classifier(model, train_loader, val_loader, epochs, learning_rate, device, 
                     num_classes=1, early_stopper=None, target_names_report=None, class_weights=None):
    model.to(device)
    
    if num_classes == 1:
        criterion = nn.BCEWithLogitsLoss().to(device)
    else: # Multi-class
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    if target_names_report is None:
        target_names_report = ['Normal (Class 0)', 'Abnormal (Class 1)'] if num_classes == 1 else [f'Class_{i}' for i in range(num_classes)]

    print(f"\n--- Training Classifier on {device} (Classes: {num_classes}) ---")
    if early_stopper: early_stopper.reset()
    
    history = {'loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0
        all_train_preds_list, all_train_labels_list = [], []

        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            
            if num_classes > 1:
                batch_labels = batch_labels.long()
            
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * batch_features.size(0)

            if num_classes == 1:
                predicted = (torch.sigmoid(outputs) > 0.5).float()
            else:
                _, predicted = torch.max(outputs.data, 1)
            
            all_train_preds_list.append(predicted.view(-1))
            all_train_labels_list.append(batch_labels.view(-1))
            
        avg_train_loss = 0.0
        if hasattr(train_loader, 'dataset') and len(train_loader.dataset) > 0:
            avg_train_loss = train_loss_sum / len(train_loader.dataset)

        train_accuracy_epoch = 0.0
        if all_train_labels_list:
            try:
                all_train_preds_np = torch.cat(all_train_preds_list).cpu().numpy()
                all_train_labels_np = torch.cat(all_train_labels_list).cpu().numpy()
                train_accuracy_epoch = accuracy_score(all_train_labels_np, all_train_preds_np) * 100
            except Exception as e:
                print(f"Warning: Could not calculate train accuracy. {e}")

        model.eval()
        val_loss_sum = 0
        all_val_preds_list, all_val_labels_list = [], []
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                
                if num_classes > 1:
                    batch_labels = batch_labels.long()
                    
                outputs = model(batch_features)
                loss_val = criterion(outputs, batch_labels)
                val_loss_sum += loss_val.item() * batch_features.size(0)
                
                if num_classes == 1:
                    predicted_val = (torch.sigmoid(outputs) > 0.5).float()
                else:
                    _, predicted_val = torch.max(outputs.data, 1)
                
                all_val_preds_list.append(predicted_val.view(-1))
                all_val_labels_list.append(batch_labels.view(-1))

        avg_val_loss = 0.0
        if hasattr(val_loader, 'dataset') and len(val_loader.dataset) > 0:
            avg_val_loss = val_loss_sum / len(val_loader.dataset)
        
        report_dict_val = {}
        val_accuracy = 0.0
        all_val_preds_np, all_val_labels_np = None, None
        
        if all_val_labels_list:
            try:
                all_val_preds_np = torch.cat(all_val_preds_list).cpu().numpy()
                all_val_labels_np = torch.cat(all_val_labels_list).cpu().numpy()

                report_dict_val = classification_report(all_val_labels_np, all_val_preds_np, 
                                                    target_names=target_names_report, 
                                                    output_dict=True, zero_division=0)
                if isinstance(report_dict_val, dict) and 'accuracy' in report_dict_val:
                    try:
                        val_accuracy = float(report_dict_val.get('accuracy', 0)) * 100
                    except (TypeError, ValueError):
                        val_accuracy = 0.0
            except Exception as e:
                 print(f"Warning: Could not calculate validation metrics. {e}")

        history['loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy_epoch)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"Classifier Epoch [{epoch+1}/{epochs}]:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy_epoch:.2f}%")
            print(f"  Val Loss  : {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            if report_dict_val:
                for class_name_in_report in target_names_report: 
                    if class_name_in_report in report_dict_val:
                        metrics = report_dict_val.get(class_name_in_report, {})
                        if not isinstance(metrics, dict):
                            metrics = {}
                        recall = float(metrics.get('recall', 0)) if isinstance(metrics, dict) else 0.0
                        precision = float(metrics.get('precision', 0)) if isinstance(metrics, dict) else 0.0
                        f1 = float(metrics.get('f1-score', 0)) if isinstance(metrics, dict) else 0.0
                        print(f"    {class_name_in_report}: "
                              f"Recall: {recall*100:.2f}%, "
                              f"Precision: {precision*100:.2f}%, "
                              f"F1: {f1*100:.2f}%")
        
        if early_stopper and early_stopper(avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch+1} with validation loss: {avg_val_loss:.6f}")
            break
            
    print("Classifier training finished.")
    return model, history