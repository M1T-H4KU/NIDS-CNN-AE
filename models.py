# models.py
import torch
import torch.nn as nn

class Autoencoder(nn.Module): # Also serves as BEGAN Discriminator's AE part
    def __init__(self, input_dim, latent_dim, hidden_dim_1):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(True),
            nn.Linear(hidden_dim_1, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(True),
            nn.Linear(hidden_dim_1, input_dim),
            nn.Sigmoid() # To output features in [0,1] range
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

class Generator(nn.Module): # For BEGAN
    def __init__(self, noise_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid() 
        )

    def forward(self, noise):
        return self.main(noise)

class ConditionalGenerator(nn.Module): # For cGAN
    def __init__(self, noise_dim, num_classes, label_emb_dim, hidden_dim, output_dim):
        super(ConditionalGenerator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, label_emb_dim)
        
        self.model = nn.Sequential(
            nn.Linear(noise_dim + label_emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Sigmoid() # Assuming normalized features [0, 1]
        )

    def forward(self, noise, labels):
        # noise: (batch_size, noise_dim)
        # labels: (batch_size,) or (batch_size, 1)
        c = self.label_embedding(labels).squeeze() # (batch, emb_dim)
        if c.dim() == 1: c = c.unsqueeze(0) # handle batch_size=1
        
        x = torch.cat([noise, c], dim=1)
        out = self.model(x)
        return out

class ConditionalDiscriminator(nn.Module): # For cGAN
    def __init__(self, input_dim, num_classes, label_emb_dim, hidden_dim):
        super(ConditionalDiscriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, label_emb_dim)
        
        self.model = nn.Sequential(
            nn.Linear(input_dim + label_emb_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            # No Sigmoid here if using BCEWithLogitsLoss
        )

    def forward(self, features, labels):
        c = self.label_embedding(labels).squeeze()
        if c.dim() == 1: c = c.unsqueeze(0)
        
        x = torch.cat([features, c], dim=1)
        out = self.model(x)
        return out

class CNNClassifier(nn.Module):
    def __init__(self, input_dim_cnn, cnn_filters, cnn_kernel_size, 
                 cnn_pool_size, cnn_pool_stride, cnn_fc_neurons, num_classes=1):
        super(CNNClassifier, self).__init__()
        
        # --- Architecture aligned with the original paper's description ---
        
        # First convolutional block
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=cnn_filters, kernel_size=cnn_kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool1d(kernel_size=cnn_pool_size, stride=cnn_pool_stride)

        # Second convolutional block
        self.conv2 = nn.Conv1d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=cnn_kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(cnn_filters)
        self.relu2 = nn.ReLU(True)
        
        self.flatten = nn.Flatten()
        
        # Helper to calculate the flattened size after conv/pool layers
        def _get_output_size(input_len, pool_size, pool_stride):
            # Conv1D with 'same' padding doesn't change length before pooling
            len_after_pool1 = (input_len - pool_size) // pool_stride + 1
            # Second Conv1D also has 'same' padding, so length is preserved from previous layer's output
            return len_after_pool1

        flattened_features_len = _get_output_size(input_dim_cnn, cnn_pool_size, cnn_pool_stride)
        self.flattened_features = cnn_filters * flattened_features_len
        
        # Fully connected layer
        self.fc1 = nn.Linear(self.flattened_features, cnn_fc_neurons)
        self.relu3 = nn.ReLU(True)
        
        # Output layer
        self.fc_output = nn.Linear(cnn_fc_neurons, num_classes)

    def forward(self, x):
        # x shape: (batch_size, feature_dim)
        x = x.unsqueeze(1) # Reshape to (batch_size, 1, feature_dim) for Conv1D

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu3(x)
        
        x = self.fc_output(x)
        return x
    
class DNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_layer_1, hidden_layer_2, num_classes=1):
        super(DNNClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_1),
            nn.BatchNorm1d(hidden_layer_1),
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(hidden_layer_1, hidden_layer_2),
            nn.BatchNorm1d(hidden_layer_2),
            nn.ReLU(True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_layer_2, num_classes)
        )

    def forward(self, x):
        return self.layers(x)