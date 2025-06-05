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
DEFAULT_EPOCHS = 300 
AE_BATCH_SIZE = 64
CNN_BATCH_SIZE = 64

# Early Stopping for AE and CNN
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
CNN_POOL_STRIDE = 3
CNN_FC_NEURONS = 16
CNN_LEARNING_RATE = 1e-3
CNN_EPOCHS = 50 # Or use DEFAULT_EPOCHS

# GAN (BEGAN) Augmentation (Defaults if not overridden by cmd line where applicable)
# USE_GAN_AUGMENTATION will be a command line argument
BEGAN_NOISE_DIM = 50
BEGAN_GENERATOR_HIDDEN_DIM = 80
BEGAN_MAX_EPOCHS_PER_CLASS = 250 # Max epochs for BEGAN training per class
BEGAN_LR = 0.0001
BEGAN_GAMMA = 0.7
BEGAN_LAMBDA_K = 0.001
BEGAN_K_T_INITIAL = 0.0
BEGAN_M_THRESHOLD = 0.058
GAN_BATCH_SIZE = 64
NUM_SYNTHETIC_SAMPLES_PER_CLASS = 35000

# Class Definitions (for multi-class BEGAN and potentially multi-class classifier)
# Mapping from attack string to integer label for multi-class processing
NSL_KDD_CLASS_MAPPING_STR_TO_INT = {
    'normal': 0,
    'back': 1, 'land': 1, 'neptune': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,
    'mailbomb': 1, 'processtable': 1, 'udpstorm': 1, 'apache2': 1, 'worm': 1,
    'satan': 2, 'ipsweep': 2, 'nmap': 2, 'portsweep': 2, 'mscan': 2, 'saint': 2,
    'guess_passwd': 3, 'ftp_write': 3, 'imap': 3, 'phf': 3, 'multihop': 3,
    'warezmaster': 3, 'warezclient': 3, 'spy': 3, 'xlock': 3, 'xsnoop': 3,
    'snmpguess': 3, 'snmpgetattack': 3, 'httptunnel': 3, 'sendmail': 3, 'named': 3,
    'buffer_overflow': 4, 'loadmodule': 4, 'perl': 4, 'rootkit': 4,
    'sqlattack': 4, 'xterm': 4, 'ps': 4
}
# Mapping from integer label to class name string
NSL_KDD_CLASS_NAMES_INT_TO_STR = {
    0: 'Normal', 1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'
}
NUM_ORIGINAL_CLASSES = len(NSL_KDD_CLASS_NAMES_INT_TO_STR) # Should be 5
DEFAULT_ATTACK_LABEL_INT = 1 # Default to DoS if specific attack not in map

# Reporting
TABLE_OUTPUT_FILENAME_PREFIX = "nslkdd_pytorch_results" # Will append GAN/classifier mode