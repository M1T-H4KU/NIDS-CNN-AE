import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, BatchNormalization, ReLU,
                                     Conv1D, MaxPooling1D, Flatten, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- 0. Configuration & Placeholders ---
# These should be adjusted based on your actual NSL-KDD preprocessing
# For NSL-KDD, after one-hot encoding, the number of features can be around 122.
# For demonstration, let's assume a placeholder number of features and samples.
NUM_FEATURES_PREPROCESSED = 122 # Example: Adjust this after your preprocessing
LATENT_DIM = 50
AE_HIDDEN_DIM_1 = 80
AE_EPOCHS = 300 # As per paper
AE_BATCH_SIZE = 64 # Common batch size, can be tuned
AE_RECONSTRUCTION_TARGET = 0.97 # For reference, training uses fixed epochs

CNN_FILTERS = 32
CNN_KERNEL_SIZE = 5
CNN_POOL_SIZE = 3
CNN_FC_NEURONS = 16
CNN_EPOCHS = 50 # Placeholder, adjust as needed
CNN_BATCH_SIZE = 64

# --- 1. NSL-KDD Data Loading and Preprocessing (Placeholder) ---
def load_and_preprocess_nsl_kdd(train_path="KDDTrain+.txt", test_path="KDDTest+.txt"):
    """
    Placeholder function for loading and preprocessing NSL-KDD data.
    You MUST replace this with your actual NSL-KDD loading and preprocessing logic.

    NSL-KDD has 41 features + 1 label (+1 difficulty score not usually used).
    Features include categorical (protocol_type, service, flag) and numerical.
    Preprocessing should involve:
    1. Loading data (e.g., from CSV or ARFF).
    2. Separating features and labels.
    3. Identifying categorical and numerical features.
    4. One-hot encoding categorical features.
    5. Scaling all features (e.g., MinMaxScaler to [0, 1]).
    """
    print("Loading and preprocessing NSL-KDD data (Placeholder)...")

    # Example: Column names for NSL-KDD (shortened for brevity, ensure you use all)
    # Refer to NSL-KDD documentation for full column names and types
    column_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
        'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
        'attack_type', 'difficulty_level'
    ]
    
    # Simulating loading data (replace with actual pd.read_csv or similar)
    # For demonstration, we'll create dummy data.
    # In reality, you would load KDDTrain+.txt and KDDTest+.txt
    num_train_samples = 125973 # Example
    num_test_samples = 22544   # Example
    
    # Create dummy data matching the *expected structure after some preprocessing*
    # This is highly simplified.
    # For the autoencoder, we only need X_train_processed.
    # For the classifier, we need X_train_processed, y_train, X_test_processed, y_test.

    # Let's assume KDDTrain+.txt has been loaded into a DataFrame `df_train`
    # And KDDTest+.txt into `df_test`
    
    # Example: if you had actual dataframes:
    # df_train = pd.read_csv(train_path, header=None, names=column_names)
    # df_test = pd.read_csv(test_path, header=None, names=column_names)

    # For this placeholder, generate random data:
    X_train_raw = np.random.rand(num_train_samples, 41) 
    X_test_raw = np.random.rand(num_test_samples, 41)
    
    # Simulate labels: 0 for normal, 1 for attack (binary)
    # In NSL-KDD, 'normal' is one class, others are attacks.
    y_train_labels = np.random.randint(0, 2, num_train_samples) 
    y_test_labels = np.random.randint(0, 2, num_test_samples)

    # Simulate categorical and numerical features for preprocessing pipeline
    # These indices are based on the standard 41 features BEFORE one-hot encoding
    categorical_features_indices = [1, 2, 3] # protocol_type, service, flag
    numerical_features_indices = [i for i in range(41) if i not in categorical_features_indices]

    # Create preprocessing pipelines for numerical and categorical features
    numerical_pipeline = Pipeline([
        ('scaler', MinMaxScaler())
    ])

    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False for dense array
    ])

    # Create a column transformer to apply transformations
    # Note: The indices here are for the dummy X_train_raw (41 features)
    # Ensure these match your actual raw data structure.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features_indices),
            ('cat', categorical_pipeline, categorical_features_indices)
        ], 
        remainder='passthrough' # In case some columns are not listed, though all should be
    )

    # Fit preprocessor on training data and transform both train and test
    # Important: Only fit on training data to prevent data leakage
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)
    
    # Update NUM_FEATURES_PREPROCESSED based on the actual shape after preprocessing
    global NUM_FEATURES_PREPROCESSED 
    NUM_FEATURES_PREPROCESSED = X_train_processed.shape[1]
    print(f"Data preprocessed. Number of features after OHE and scaling: {NUM_FEATURES_PREPROCESSED}")

    # For labels, usually, they are strings like 'normal.', 'neptune.', etc.
    # You'd map them to integers (e.g., 0 for normal, 1 for any attack for binary)
    # y_train = df_train['attack_type'].apply(lambda x: 0 if x == 'normal' else 1).values
    # y_test = df_test['attack_type'].apply(lambda x: 0 if x == 'normal' else 1).values
    y_train = y_train_labels # Using the dummy labels
    y_test = y_test_labels   # Using the dummy labels

    print(f"X_train_processed shape: {X_train_processed.shape}, y_train shape: {y_train.shape}")
    print(f"X_test_processed shape: {X_test_processed.shape}, y_test shape: {y_test.shape}")
    
    return X_train_processed, y_train, X_test_processed, y_test


# --- 2. Autoencoder (AE) for Feature Extraction ---
def build_autoencoder(input_dim, latent_dim, hidden_dim_1):
    # Encoder
    input_layer = Input(shape=(input_dim,), name="AE_Input")
    
    # First hidden layer
    encoded = Dense(hidden_dim_1, activation='linear', name="Encoder_Hidden1_Dense")(input_layer)
    encoded = BatchNormalization(name="Encoder_Hidden1_BN")(encoded)
    encoded = ReLU(name="Encoder_Hidden1_ReLU")(encoded)
    
    # Latent space
    latent_space = Dense(latent_dim, activation='linear', name="Latent_Space_Dense")(encoded)
    latent_space = BatchNormalization(name="Latent_Space_BN")(latent_space)
    latent_space_output = ReLU(name="Latent_Space_ReLU")(latent_space) # This is the output of the encoder part
    
    encoder = Model(input_layer, latent_space_output, name="Encoder")

    # Decoder
    decoder_input = Input(shape=(latent_dim,), name="Decoder_Input")
    
    # Symmetrically first decoder hidden layer
    decoded = Dense(hidden_dim_1, activation='linear', name="Decoder_Hidden1_Dense")(decoder_input)
    decoded = BatchNormalization(name="Decoder_Hidden1_BN")(decoded)
    decoded = ReLU(name="Decoder_Hidden1_ReLU")(decoded)
    
    # Output layer (reconstruction)
    # Sigmoid activation is common if inputs are scaled to [0,1]
    reconstructed_output = Dense(input_dim, activation='sigmoid', name="Decoder_Output_Dense")(decoded)
    
    decoder = Model(decoder_input, reconstructed_output, name="Decoder")

    # Autoencoder (Encoder + Decoder)
    autoencoder_output = decoder(encoder.output)
    autoencoder = Model(input_layer, autoencoder_output, name="Autoencoder")

    autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')
    print("\n--- Autoencoder Architecture ---")
    autoencoder.summary()
    print("\n--- Encoder part Architecture ---")
    encoder.summary()
    return autoencoder, encoder

# --- 3. CNN Classifier ---
def build_cnn_classifier(input_dim_cnn, num_classes=1):
    # input_dim_cnn is the latent_dim from AE
    input_layer = Input(shape=(input_dim_cnn,), name="CNN_Input_Raw")
    
    # Reshape for Conv1D: (batch_size, steps, channels)
    # Here, steps = input_dim_cnn (number of features from AE), channels = 1
    reshaped_input = Reshape((input_dim_cnn, 1), name="CNN_Reshape")(input_layer)

    # First Convolutional Layer
    conv1 = Conv1D(filters=CNN_FILTERS, kernel_size=CNN_KERNEL_SIZE, padding='same', activation='linear', name="CNN_Conv1_Dense")(reshaped_input)
    conv1 = BatchNormalization(name="CNN_Conv1_BN")(conv1)
    conv1 = ReLU(name="CNN_Conv1_ReLU")(conv1)
    conv1_pooled = MaxPooling1D(pool_size=CNN_POOL_SIZE, padding='same', name="CNN_MaxPool1")(conv1)

    # Second Convolutional Layer
    conv2 = Conv1D(filters=CNN_FILTERS, kernel_size=CNN_KERNEL_SIZE, padding='same', activation='linear', name="CNN_Conv2_Dense")(conv1_pooled)
    conv2 = BatchNormalization(name="CNN_Conv2_BN")(conv2)
    conv2 = ReLU(name="CNN_Conv2_ReLU")(conv2)

    # Flatten before Fully Connected Layer
    flattened = Flatten(name="CNN_Flatten")(conv2)

    # Fully Connected Layer
    fc1 = Dense(CNN_FC_NEURONS, activation='relu', name="CNN_FC1")(flattened)

    # Output Layer (Binary Classification)
    if num_classes == 1:
        output_activation = 'sigmoid'
        loss_function = 'binary_crossentropy'
        output_neurons = 1
    else: # Multi-class
        output_activation = 'softmax'
        loss_function = 'categorical_crossentropy'
        output_neurons = num_classes
        
    output_layer = Dense(output_neurons, activation=output_activation, name="CNN_Output")(fc1)

    cnn_model = Model(input_layer, output_layer, name="CNN_Classifier")
    cnn_model.compile(optimizer=Adam(), loss=loss_function, metrics=['accuracy'])
    
    print("\n--- CNN Classifier Architecture ---")
    cnn_model.summary()
    return cnn_model

# --- 4. Main Workflow ---
if __name__ == '__main__':
    # Step 1: Load and preprocess data (Replace with your actual implementation)
    dataset_nsl_kdd_train_path = "datasets/nsl-kdd/KDDTrain+.txt"
    dataset_nsl_kdd_test_path = "datasets/nsl-kdd/KDDTest+.txt"
    # For the AE, we primarily need X_train_processed.
    # For the CNN, we will need X_train_features (from AE), y_train, X_test_features, y_test.
    X_train_processed, y_train, X_test_processed, y_test = load_and_preprocess_nsl_kdd(dataset_nsl_kdd_train_path, dataset_nsl_kdd_test_path)

    # Ensure NUM_FEATURES_PREPROCESSED is correctly set by the preprocessing function
    if X_train_processed.shape[1] != NUM_FEATURES_PREPROCESSED:
        print(f"Warning: NUM_FEATURES_PREPROCESSED might be inconsistent. Actual features: {X_train_processed.shape[1]}")
        NUM_FEATURES_PREPROCESSED = X_train_processed.shape[1]

    # Step 2.1: Build and Train Autoencoder
    print("\nBuilding and Training Autoencoder...")
    autoencoder, encoder = build_autoencoder(
        input_dim=NUM_FEATURES_PREPROCESSED,
        latent_dim=LATENT_DIM,
        hidden_dim_1=AE_HIDDEN_DIM_1
    )
    
    # Training the autoencoder
    # Note: The paper mentions stopping when "reconstruction accuracy was above 0.97".
    # This is hard to implement directly without a clear definition.
    # We will train for the specified number of epochs.
    # Consider using EarlyStopping on validation loss if you have a validation set.
    print(f"Training Autoencoder for {AE_EPOCHS} epochs...")
    history_ae = autoencoder.fit(
        X_train_processed, X_train_processed, # AE learns to reconstruct its input
        epochs=AE_EPOCHS,
        batch_size=AE_BATCH_SIZE,
        shuffle=True,
        verbose=1 # Set to 1 or 2 for progress, 0 for silent
    )
    print("Autoencoder training completed.")
    final_ae_loss = history_ae.history['loss'][-1]
    print(f"Final Autoencoder Reconstruction MSE Loss: {final_ae_loss:.4f}")

    # Step 2.2: Extract Features using the trained Encoder part of the AE
    print("\nExtracting features using the trained Autoencoder's Encoder part...")
    X_train_features_ae = encoder.predict(X_train_processed)
    X_test_features_ae = encoder.predict(X_test_processed)
    print(f"Shape of original training data: {X_train_processed.shape}")
    print(f"Shape of AE-extracted training features: {X_train_features_ae.shape}")
    print(f"Shape of original test data: {X_test_processed.shape}")
    print(f"Shape of AE-extracted test features: {X_test_features_ae.shape}")
    
    # Step 3: Train CNN Classifier with AE features
    # The paper implies the classifier is trained on augmented data + AE features.
    # For now, we use original data processed by AE.
    print("\nBuilding and Training CNN Classifier...")
    
    # Assuming binary classification (normal vs. attack)
    # If y_train/y_test are not already 0/1, you need to convert them.
    # If it's multi-class, adjust `num_classes` and ensure y_train/y_test are one-hot encoded.
    num_classes_for_cnn = 1 # For binary
    # If multiclass, determine number of unique classes in y_train:
    # num_classes_for_cnn = len(np.unique(y_train)) 
    # And ensure y_train/y_test are one-hot encoded for 'categorical_crossentropy'

    cnn_classifier = build_cnn_classifier(input_dim_cnn=LATENT_DIM, num_classes=num_classes_for_cnn)

    print(f"Training CNN Classifier for {CNN_EPOCHS} epochs...")
    history_cnn = cnn_classifier.fit(
        X_train_features_ae, y_train,
        epochs=CNN_EPOCHS,
        batch_size=CNN_BATCH_SIZE,
        validation_data=(X_test_features_ae, y_test), # Use test set for validation here
        shuffle=True,
        verbose=1
    )
    print("CNN Classifier training completed.")

    # Evaluate the CNN classifier
    print("\nEvaluating CNN Classifier on test data...")
    loss_cnn, accuracy_cnn = cnn_classifier.evaluate(X_test_features_ae, y_test, verbose=0)
    print(f"CNN Test Loss: {loss_cnn:.4f}")
    print(f"CNN Test Accuracy: {accuracy_cnn:.4f}")

    print("\nProcess finished.")
    print("Next steps would involve implementing the GAN for data augmentation (Step 1 from paper)")
    print("and then retraining the AE/CNN pipeline with augmented data if specified.")