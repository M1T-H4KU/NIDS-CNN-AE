# models.py
import torch
import torch.nn as nn

class Autoencoder(nn.Module): # Also serves as BEGAN Discriminator's AE part
    def __init__(self, input_dim, latent_dim, hidden_dim_1):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.encoder_bn1 = nn.BatchNorm1d(hidden_dim_1)
        self.encoder_relu1 = nn.ReLU(True) # Inplace ReLU
        self.encoder_fc2 = nn.Linear(hidden_dim_1, latent_dim)
        self.encoder_bn2 = nn.BatchNorm1d(latent_dim)
        self.encoder_relu2 = nn.ReLU(True)

        # Decoder
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim_1)
        self.decoder_bn1 = nn.BatchNorm1d(hidden_dim_1)
        self.decoder_relu1 = nn.ReLU(True)
        self.decoder_fc2 = nn.Linear(hidden_dim_1, input_dim)
        self.decoder_sigmoid = nn.Sigmoid()

    def encode(self, x):
        x = self.encoder_fc1(x)
        x = self.encoder_bn1(x)
        x = self.encoder_relu1(x)
        x = self.encoder_fc2(x)
        x = self.encoder_bn2(x)
        x = self.encoder_relu2(x)
        return x

    def decode(self, x):
        x = self.decoder_fc1(x)
        x = self.decoder_bn1(x)
        x = self.decoder_relu1(x)
        x = self.decoder_fc2(x)
        x = self.decoder_sigmoid(x)
        return x

    def forward(self, x): # For AE training & BEGAN D's reconstruction
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

class CNNClassifier(nn.Module):
    def __init__(self, input_dim_cnn, cnn_filters, cnn_kernel_size, 
                 cnn_pool_size, cnn_pool_stride, cnn_fc_neurons, num_classes=1):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=cnn_filters,
                               kernel_size=cnn_kernel_size, padding=(cnn_kernel_size - 1) // 2)
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool1d(kernel_size=cnn_pool_size, stride=cnn_pool_stride)

        self.conv2 = nn.Conv1d(in_channels=cnn_filters, out_channels=cnn_filters,
                               kernel_size=cnn_kernel_size, padding=(cnn_kernel_size - 1) // 2)
        self.bn2 = nn.BatchNorm1d(cnn_filters)
        self.relu2 = nn.ReLU(True)
        self.flatten = nn.Flatten()

        def _get_conv_output_size(input_len, kernel, stride, padding=0, dilation=1):
            return (input_len + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

        conv1_out_len = input_dim_cnn # Since padding='same' for conv
        pool1_out_len = _get_conv_output_size(conv1_out_len, cnn_pool_size, cnn_pool_stride)
        conv2_out_len = pool1_out_len # Since padding='same' for conv
        
        self.flattened_features = cnn_filters * conv2_out_len
        
        self.fc1 = nn.Linear(self.flattened_features, cnn_fc_neurons)
        self.relu3 = nn.ReLU(True)
        self.fc_output = nn.Linear(cnn_fc_neurons, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
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