# configs.py
import torch

# --- 0. Configuration & Device Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths
BASE_DATA_PATH = "datasets/nsl-kdd/"
TRAIN_FILENAME = "KDDTrain+.txt"
TEST_FILENAME = "KDDTest+.txt"

# Feature caching
EXTRACTED_FEATURES_FILE_ORIGINAL = "ae_features_original.pt"
EXTRACTED_FEATURES_FILE_GAN_AUGMENTED = "ae_features_gan_augmented.pt"

# General training
DEFAULT_EPOCHS = 300 # For AE and CNN if not specified otherwise by BEGAN paper info
AE_BATCH_SIZE = 64
CNN_BATCH_SIZE = 64

# Early Stopping for AE and CNN (based on training loss)
EARLY_STOPPING_PATIENCE = 35
EARLY_STOPPING_REL_DELTA = 1e-6

# Autoencoder (for feature extraction, and also as D in BEGAN)
AE_LATENT_DIM = 50
AE_HIDDEN_DIM_1 = 80
AE_LEARNING_RATE = 1e-3

# CNN Classifier
CNN_FILTERS = 32
CNN_KERNEL_SIZE = 5
CNN_POOL_SIZE = 3
CNN_POOL_STRIDE = 3 # Assuming stride = pool_size based on common practice
CNN_FC_NEURONS = 16
CNN_LEARNING_RATE = 1e-3
CNN_EPOCHS = 50 # Or use DEFAULT_EPOCHS

# GAN (BEGAN) Augmentation
USE_GAN_AUGMENTATION = True # Master switch for GAN part
BEGAN_NOISE_DIM = 50
BEGAN_GENERATOR_HIDDEN_DIM = 80 # G's hidden layer
# BEGAN Discriminator (Autoencoder) uses AE_LATENT_DIM and AE_HIDDEN_DIM_1
BEGAN_MAX_EPOCHS_PER_CLASS = 250
BEGAN_LR = 0.0001
BEGAN_GAMMA = 0.7        # Diversity ratio for BEGAN [0.5, 0.75 from paper example]
BEGAN_LAMBDA_K = 0.001   # Learning rate for k_t in BEGAN
BEGAN_K_T_INITIAL = 0.0
BEGAN_M_THRESHOLD = 0.058 # Convergence threshold
GAN_BATCH_SIZE = 64      # Batch size for BEGAN training
# Number of synthetic samples to generate per class.
# This might be a fixed number, or a multiplier of original class size.
# For now, a fixed number. You might want different numbers for different classes.
NUM_SYNTHETIC_SAMPLES_PER_CLASS_MAP = {
    0: 2000,  # Example: Augment 'normal' (class 0) by 2000 samples
    1: 5000   # Example: Augment 'attack' (class 1) by 5000 samples
    # Add more if you have more classes from NSL-KDD in a multi-class setup
}
# Or a general value if you want the same for all augmented classes:
# NUM_SYNTHETIC_SAMPLES_TO_GENERATE = 5000

# Reporting
TABLE_OUTPUT_FILENAME_ORIGINAL = "nslkdd_pytorch_results_original.png"
TABLE_OUTPUT_FILENAME_GAN = "nslkdd_pytorch_results_gan_augmented.png"