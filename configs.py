# configs.py
import torch

# --- 0. Configuration & Device Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths
BASE_DATA_PATH = "datasets/nsl-kdd/"
# Files for Multi-class classification (requires multi-class labels)
TRAIN_FILENAME_TXT = "KDDTrain+.txt"
TEST_FILENAME_TXT = "KDDTest+.txt"
# Files for Binary classification (can have any labels, will be mapped to binary)
TRAIN_FILENAME_ARFF = "KDDTrain+.arff"
TEST_FILENAME_ARFF = "KDDTest+.arff"

# Feature caching (filenames are now determined in main.py)
EXTRACTED_FEATURES_FILE_GAN_AUGMENTED = "ae_features_gan_augmented.pt"
EXTRACTED_FEATURES_FILE_ORIGINAL = "ae_features_original.pt"

# General training
DEFAULT_EPOCHS = 300 
AE_BATCH_SIZE = 64
CNN_BATCH_SIZE = 64

# Early Stopping
EARLY_STOPPING_PATIENCE = 35
EARLY_STOPPING_MIN_DELTA_VAL = 1e-6 # For validation loss

# Autoencoder (for feature extraction, and also as D in BEGAN)
AE_LATENT_DIM = 50
AE_HIDDEN_DIM_1 = 80
AE_LEARNING_RATE = 1e-3

# CNN Classifier
CNN_FILTERS = 32
CNN_KERNEL_SIZE = 5
CNN_POOL_SIZE = 3
CNN_POOL_STRIDE = 3
CNN_FC_NEURONS = 16
CNN_LEARNING_RATE = 1e-3
CNN_EPOCHS = 50 

# GAN (BEGAN) Augmentation
BEGAN_NOISE_DIM = 50
BEGAN_GENERATOR_HIDDEN_DIM = 80
BEGAN_MAX_EPOCHS_PER_CLASS = 250
BEGAN_LR = 0.0001
BEGAN_GAMMA = 0.7
BEGAN_LAMBDA_K = 0.001
BEGAN_K_T_INITIAL = 0.0
BEGAN_M_THRESHOLD = 0.058
GAN_BATCH_SIZE = 64
NUM_SYNTHETIC_SAMPLES_PER_CLASS = 35000

# Class Definitions
NSL_KDD_CLASS_MAPPING_STR_TO_INT = {
    'normal': 0, 'back': 1, 'land': 1, 'neptune': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,
    'mailbomb': 1, 'processtable': 1, 'udpstorm': 1, 'apache2': 1, 'worm': 1,
    'satan': 2, 'ipsweep': 2, 'nmap': 2, 'portsweep': 2, 'mscan': 2, 'saint': 2,
    'guess_passwd': 3, 'ftp_write': 3, 'imap': 3, 'phf': 3, 'multihop': 3,
    'warezmaster': 3, 'warezclient': 3, 'spy': 3, 'xlock': 3, 'xsnoop': 3,
    'snmpguess': 3, 'snmpgetattack': 3, 'httptunnel': 3, 'sendmail': 3, 'named': 3,
    'buffer_overflow': 4, 'loadmodule': 4, 'perl': 4, 'rootkit': 4,
    'sqlattack': 4, 'xterm': 4, 'ps': 4
}
NSL_KDD_CLASS_NAMES_INT_TO_STR = {
    0: 'Normal', 1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'
}
NUM_ORIGINAL_CLASSES = len(NSL_KDD_CLASS_NAMES_INT_TO_STR)
DEFAULT_ATTACK_LABEL_INT = 1

# Reporting
TABLE_OUTPUT_FILENAME_PREFIX = "nslkdd_pytorch_results"