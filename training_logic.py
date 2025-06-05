# training_logic.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
# from utils import TrainingLossEarlyStopper # Will be imported in main.py and instance passed

def train_autoencoder(model, dataloader, epochs, learning_rate, device, early_stopper=None):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\n--- Training Feature Extraction Autoencoder on {device} ---")
    if early_stopper: early_stopper.reset()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_features in dataloader:
            batch_features = batch_features.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_features)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs -1:
            print(f"AE Epoch [{epoch+1}/{epochs}], Training Loss: {avg_epoch_loss:.6f}")

        if early_stopper and early_stopper(avg_epoch_loss):
            break 
    print("Feature Extraction Autoencoder training finished.")

def train_began_for_class(
    generator, discriminator_ae, class_dataloader,
    noise_dim, epochs, device, class_name,
    gamma, lambda_k, k_t_initial, m_threshold, lr
):
    generator.to(device)
    discriminator_ae.to(device)
    ae_loss_fn = nn.L1Loss() # Or nn.MSELoss()

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

            balance = (gamma * loss_d_real.item() - loss_d_fake.item()) # use .item() for scalars
            k_t = np.clip(k_t + lambda_k * balance, 0, 1)
            current_m = loss_d_real.item() + np.abs(gamma * loss_d_real.item() - loss_g.item())

            epoch_loss_d_real_sum += loss_d_real.item()
            epoch_loss_d_fake_sum += loss_d_fake.item()
            epoch_loss_g_sum += loss_g.item()
            epoch_m_sum += current_m
            num_batches += 1
        
        if num_batches > 0:
            avg_loss_d_real = epoch_loss_d_real_sum / num_batches
            avg_loss_g = epoch_loss_g_sum / num_batches # L(G(z)) from G's perspective
            avg_m = epoch_m_sum / num_batches
            
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs -1:
                print(f"  BEGAN Epoch [{epoch+1}/{epochs}] for '{class_name}': M={avg_m:.4f}, L(x)={avg_loss_d_real:.4f}, "
                      f"L(G(z))={avg_loss_g:.4f}, k_t={k_t:.4f}")
            if avg_m < m_threshold and epoch > 10: # Check M not too early
                print(f"BEGAN convergence for class '{class_name}' reached at M={avg_m:.4f} (Threshold: {m_threshold})")
                break
        else:
            print(f"No batches processed for BEGAN class {class_name} in epoch {epoch+1}. Skipping logging for this epoch.")


    print(f"BEGAN training for class '{class_name}' finished.")
    return generator

def augment_data_with_gan(generator, noise_dim, num_samples, device, gan_batch_size):
    generator.eval()
    generated_data_list = []
    if num_samples == 0:
        return torch.empty(0, generator.main[-2].out_features, device='cpu') # Ensure correct output dimension

    with torch.no_grad():
        for _ in range(int(np.ceil(num_samples / gan_batch_size))):
            current_batch_gen_size = min(gan_batch_size, num_samples - len(generated_data_list))
            if current_batch_gen_size <= 0: break
            noise = torch.randn(current_batch_gen_size, noise_dim, device=device)
            synthetic_samples = generator(noise).cpu() # Move to CPU after generation
            generated_data_list.append(synthetic_samples)
    
    if not generated_data_list:
        return torch.empty(0, generator.main[-2].out_features, device='cpu')
        
    generated_data = torch.cat(generated_data_list, dim=0)
    return generated_data[:num_samples]

def extract_ae_features(encoder_model, dataloader, device, batch_size): # encoder_model is the full AE, we use its .encode
    encoder_model.to(device)
    encoder_model.eval()
    all_features_list = []
    temp_dataloader = DataLoader(dataloader.dataset, batch_size=batch_size, shuffle=False) # Use consistent batch size

    with torch.no_grad():
        for batch_data in temp_dataloader:
             # NSLKDDDataset returns x or (x,y), handle both for general use
            if isinstance(batch_data, list) or isinstance(batch_data, tuple):
                batch_data = batch_data[0] # Take only features
            batch_data = batch_data.to(device)
            encoded_features = encoder_model.encode(batch_data)
            all_features_list.append(encoded_features.cpu()) # Move to CPU
    if not all_features_list:
        # Determine expected feature dimension if dataloader was empty
        # This is tricky; for now, assume it won't be empty if called.
        # Or pass expected_latent_dim
        print("Warning: No data processed during feature extraction.")
        return torch.empty(0, device='cpu') # Or handle as error
        
    return torch.cat(all_features_list, dim=0)


def train_classifier(model, train_loader, val_loader, epochs, learning_rate, device, 
                     num_classes=1, early_stopper=None, target_names_report=None):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    if target_names_report is None:
        target_names_report = ['Normal (Class 0)', 'Abnormal (Class 1)'] if num_classes == 1 else [f'Class_{i}' for i in range(num_classes)]

    print(f"\n--- Training CNN Classifier on {device} (Classes: {num_classes}) ---")
    if early_stopper: early_stopper.reset()
        
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0
        all_train_preds_list = [] # For epoch's training accuracy
        all_train_labels_list = [] # For epoch's training accuracy

        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * batch_features.size(0)

            if num_classes == 1:
                predicted = (torch.sigmoid(outputs) > 0.5).float() # Shape (B,1)
            else:
                _, predicted = torch.max(outputs.data, 1) # Shape (B,)
            
            # Use view(-1) for robust flattening for training metrics
            all_train_preds_list.extend(predicted.view(-1).cpu().numpy())
            all_train_labels_list.extend(batch_labels.view(-1).cpu().numpy())
            
        avg_train_loss = train_loss_sum / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0
        train_accuracy_epoch = accuracy_score(all_train_labels_list, all_train_preds_list) * 100 if len(all_train_labels_list) > 0 else 0.0
        
        # Validation
        model.eval()
        val_loss_sum = 0
        all_val_preds_list = [] # For epoch's validation metrics
        all_val_labels_list = [] # For epoch's validation metrics
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                loss_val = criterion(outputs, batch_labels)
                val_loss_sum += loss_val.item() * batch_features.size(0)
                
                if num_classes == 1:
                    predicted_val = (torch.sigmoid(outputs) > 0.5).float() # Shape (B,1)
                else:
                    _, predicted_val = torch.max(outputs.data, 1) # Shape (B,)
                
                # Use view(-1) for robust flattening for validation metrics
                all_val_preds_list.extend(predicted_val.view(-1).cpu().numpy())
                all_val_labels_list.extend(batch_labels.view(-1).cpu().numpy())

        avg_val_loss = val_loss_sum / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0
        
        report_dict_val = {}
        val_accuracy = 0.0
        if len(all_val_labels_list) > 0:
            report_dict_val = classification_report(all_val_labels_list, all_val_preds_list, 
                                                target_names=target_names_report, 
                                                output_dict=True, zero_division=0)
            if 'accuracy' in report_dict_val:
                val_accuracy = report_dict_val['accuracy'] * 100
        
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs -1 :
            print(f"CNN Epoch [{epoch+1}/{epochs}]:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy_epoch:.2f}%")
            print(f"  Val Loss  : {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            if report_dict_val:
                for class_name_in_report in target_names_report: 
                    if class_name_in_report in report_dict_val:
                        metrics = report_dict_val[class_name_in_report]
                        print(f"    {class_name_in_report}: " # Indent for validation epoch log
                              f"P: {metrics.get('precision',0)*100:.2f}%, "
                              f"R: {metrics.get('recall',0)*100:.2f}%, "
                              f"F1: {metrics.get('f1-score',0)*100:.2f}%")
        if early_stopper and early_stopper(avg_train_loss):
            break
    print("CNN training finished.")
    return model