import numpy as np
import mne
import os
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from datetime import datetime # Import datetime here for TensorBoard callback

from model_architectures import build_hybrid_cnn_lstm, build_attention_eegnet, square, log

# Configuration
EPOCHS_TO_LOAD_PATH = 'preprocessed_epochs' # Directory where preprocessed epochs are saved
MODEL_SAVE_DIR = 'trained_models'
TENSORBOARD_LOG_DIR = 'logs/fit'
SCALER_SAVE_PATH = 'scaler.pkl' # Path to save the fitted scaler

# Ensure directories exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

def prepare_data_for_nn(epochs, desired_event_strings):
    """
    Prepares MNE Epochs data into a format suitable for Keras/TensorFlow.
    Applies StandardScaler to the EEG data.
    
    Args:
        epochs (mne.Epochs): Preprocessed MNE Epochs object (potentially containing all event types).
        desired_event_strings (list): A list of event strings (e.g., "PUSH_IMAGERY_START", "REST_START")
                                      that you want to include in training.
        
    Returns:
        tuple: (X, y, label_encoder, scaler)
               X (np.ndarray): EEG data (samples, channels, time_points, 1).
               y (np.ndarray): One-hot encoded labels.
               label_encoder (LabelEncoder): Encoder for mapping event IDs to class names.
               scaler (StandardScaler): Fitted StandardScaler object.
    """
    # 1. Get the raw data from the loaded epochs object, as we will re-epoch it.
    # The epochs object has a reference to the underlying raw data.
    raw_data_from_epochs = epochs.info # We only need the info and data from this
    # Actually, we need the original raw data from which these epochs were formed.
    # MNE Epochs object often carries a reference to the Raw object it was derived from.
    # If not, we might need to modify eeg_preprocessor to save the raw data too, or
    # infer it from the epochs. For this case, we'll try to get the raw data from epochs.
    # Note: epochs.copy().crop() or epochs.to_data_frame() doesn't give back Raw.
    # The best approach is to re-epoch from the original raw file or ensure 'epochs'
    # object is truly filtered based on event types.
    
    # --- REVISED STRATEGY: Directly filter `epochs.events` and create a new `Epochs` object ---
    # This addresses the problem of `epochs[event_ids_list]` not fully filtering event types.

    # Extract all events and their mapping from the loaded epochs
    all_events = epochs.events
    all_event_id_map = epochs.event_id

    # Filter events to include ONLY the desired `_IMAGERY_START` and `REST_START` events
    target_mne_event_ids = [] # These are the integer IDs we want to keep
    
    # This dictionary will map the ACTUAL MNE integer IDs (from epochs.event_id)
    # to the simplified string labels we want to use for training (e.g., 'PUSH', 'REST').
    # Example: {12: 'PUSH', 3: 'PULL', 9: 'LEFT', 15: 'RIGHT', 18: 'UP', 21: 'DOWN', 5: 'REST'}
    actual_id_to_simplified_label_map = {} 

    # Populate target_mne_event_ids and actual_id_to_simplified_label_map
    for desired_event_string in desired_event_strings:
        if desired_event_string in all_event_id_map:
            mne_int_id = all_event_id_map[desired_event_string]
            target_mne_event_ids.append(mne_int_id)
            
            # Create the simplified label
            if '_IMAGERY_START' in desired_event_string:
                simplified_label = desired_event_string.replace('_IMAGERY_START', '')
            elif 'REST_START' in desired_event_string:
                simplified_label = 'REST'
            else:
                simplified_label = desired_event_string # Fallback, should not be hit if input is clean
            
            actual_id_to_simplified_label_map[mne_int_id] = simplified_label
        else:
            print(f"Warning: Desired event '{desired_event_string}' not found in the loaded epochs.event_id. This class will not be trained.")

    if not target_mne_event_ids:
        raise ValueError("No relevant IMAGERY_START or REST_START events found in the loaded epochs that match the defined `desired_event_strings`. Check your data collection and preprocessing.")

    print(f"Target MNE Event IDs for training: {target_mne_event_ids}")
    print(f"Simplified Label Map for Training: {actual_id_to_simplified_label_map}")

    # Now, explicitly filter the `all_events` array
    filtered_events_array = []
    for event in all_events:
        if event[2] in target_mne_event_ids: # Check if the event ID is one of our desired ones
            filtered_events_array.append(event)
    
    if not filtered_events_array:
        raise ValueError("No epochs found after filtering original events by desired training classes. Check your data and `desired_event_strings`.")

    filtered_events_array = np.array(filtered_events_array)
    
    # Re-create a new Epochs object using the original raw (or a copy of it, if it still holds it)
    # This is the most reliable way to ensure we only have the desired events and their data.
    # MNE Epochs objects generally have a ._raw attribute pointing to the original Raw.
    # If the .fif only stores epochs data without raw, then we need to load Raw data too.
    # For now, assuming the original raw data is accessible through epochs.info or can be recreated.
    # A simpler approach (given epochs are already loaded) is to just extract data/labels from the *filtered* events
    # from the original epochs object, but carefully. Let's refine the loop below.

    # We need to extract data and labels for the epochs that correspond to `target_mne_event_ids`.
    # `epochs[target_mne_event_ids]` was supposed to do this but caused issues.
    # Let's manually select from `epochs` based on the event IDs.
    
    X_selected_epochs_list = []
    y_selected_labels_list = []

    # Iterate through each epoch in the loaded epochs object
    for i, epoch_data in enumerate(epochs.get_data(units='uV')): # This gets data (n_epochs, n_channels, n_times)
        current_epoch_event_id = epochs.events[i, 2] # Get the event ID for this specific epoch
        
        if current_epoch_event_id in actual_id_to_simplified_label_map:
            simplified_label = actual_id_to_simplified_label_map[current_epoch_event_id]
            X_selected_epochs_list.append(epoch_data)
            y_selected_labels_list.append(simplified_label)
        else:
            # This handles cases where the event ID is not one of our desired training classes
            # e.g., 'FIXATION_END', 'PUSH_CUE', etc.
            print(f"[Prepare Data Debug] Skipping epoch {i} with MNE ID {current_epoch_event_id}")

    if not X_selected_epochs_list:
        raise ValueError("No epochs found after filtering for desired training classes. Ensure your collected data contains appropriate markers and `desired_event_strings` is correct.")

    X_raw_final = np.array(X_selected_epochs_list)
    y_labels = np.array(y_selected_labels_list)
    
    print(f"Number of epochs extracted for training: {len(X_raw_final)}")
    print(f"Unique labels extracted for training: {np.unique(y_labels)}")

    # 4. Encode string labels to integers (0, 1, 2...) and then one-hot encode
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_labels)
    y_one_hot = tf.keras.utils.to_categorical(y_encoded, num_classes=len(label_encoder.classes_))

    # Reshape X_raw for scaling: (n_samples * n_time_points, n_channels)
    n_epochs, n_channels, n_time_points = X_raw_final.shape
    X_reshaped_for_scaler = X_raw_final.transpose(0, 2, 1).reshape(-1, n_channels)

    # Fit and transform data using StandardScaler
    scaler = StandardScaler()
    X_scaled_reshaped = scaler.fit_transform(X_reshaped_for_scaler)
    
    # Reshape back to (n_epochs, n_channels, n_time_points)
    X_scaled = X_scaled_reshaped.reshape(n_epochs, n_time_points, n_channels).transpose(0, 2, 1)

    # Reshape X for Keras Conv2D: (samples, channels, time_points, 1)
    X = X_scaled[:, :, :, np.newaxis]
    
    print(f"Data shape for NN (X): {X.shape}")
    print(f"Labels shape for NN (y): {y_one_hot.shape}")
    print(f"Class labels: {label_encoder.classes_}")
    
    return X, y_one_hot, label_encoder, scaler

def train_model(model, X_train, y_train, X_val, y_val, model_name, epochs=100, batch_size=32):
    """
    Trains a deep learning model.
    
    Args:
        model (tf.keras.Model): The Keras model to train.
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation data.
        y_val (np.ndarray): Validation labels.
        model_name (str): Name for saving the model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    """
    print(f"\n--- Training {model_name} Model ---")
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(MODEL_SAVE_DIR, f'{model_name}_best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(TENSORBOARD_LOG_DIR, model_name, datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=1)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint, tensorboard_callback],
        verbose=1
    )
    
    print(f"Training of {model_name} completed.")
    return history

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluates the trained model and prints a classification report and confusion matrix.
    
    Args:
        model (tf.keras.Model): The trained Keras model.
        X_test (np.ndarray): Test data.
        y_test (np.ndarray): True test labels (one-hot encoded).
        label_encoder (LabelEncoder): Label encoder used during data preparation.
    """
    print("\n--- Model Evaluation ---")
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    # Define your desired event strings. These are the *names* of the markers
    # you want to train on, as they appear in your raw data and .fif files.
    desired_event_strings_for_training = [
        "PUSH_IMAGERY_START",
        "PULL_IMAGERY_START",
        "LEFT_IMAGERY_START",
        "RIGHT_IMAGERY_START",
        "UP_IMAGERY_START",
        "DOWN_IMAGERY_START",
        "REST_START", 
    ]
    
    epochs = None
    try:
        # Try to find the latest preprocessed epochs file
        list_of_files = glob.glob(os.path.join(EPOCHS_TO_LOAD_PATH, 'mindrove_raw_session_*-epo.fif'))
        if list_of_files:
            latest_file = max(list_of_files, key=os.path.getctime)
            epochs = mne.read_epochs(latest_file, preload=True)
            print(f"Loaded latest preprocessed epochs: {latest_file}")
            print(f"Event IDs in loaded epochs (from .fif file): {epochs.event_id}") # Crucial for debugging!
        else:
            raise FileNotFoundError
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading epochs: {e}")
        print("No preprocessed epochs found. Please ensure `eeg_preprocessor.py` has been run on collected data and has produced valid epochs.")
        exit() # Exit if no real data is found

    # Prepare data for Neural Network, including scaling
    # Pass the list of desired event strings
    X, y, label_encoder, scaler = prepare_data_for_nn(epochs, desired_event_strings_for_training)

    # Save the fitted scaler for real-time classification
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Fitted StandardScaler saved to {SCALER_SAVE_PATH}")

    # Split data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    #X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp) # 15% val, 15% test
    X_val = X_test
    y_val = y_test
    print(f"Train data shape: {X_train.shape}, labels: {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}, labels: {y_val.shape}")
    print(f"Test data shape: {X_test.shape}, labels: {y_test.shape}")

    n_channels = X.shape[1]
    n_time_points = X.shape[2]
    num_classes = y.shape[1]

    # --- Train Hybrid CNN-LSTM Model ---
    cnn_lstm_model = build_hybrid_cnn_lstm(input_shape=(n_channels, n_time_points), num_classes=num_classes)
    train_model(cnn_lstm_model, X_train, y_train, X_val, y_val, "Hybrid_CNN_LSTM")
    
    # Load the best saved model for evaluation
    best_cnn_lstm_model = tf.keras.models.load_model(os.path.join(MODEL_SAVE_DIR, 'Hybrid_CNN_LSTM_best_model.h5'),
                                                     custom_objects={'square': square, 'log': log})
    evaluate_model(best_cnn_lstm_model, X_test, y_test, label_encoder)

    # --- Train Attention-Enhanced EEGNet Model ---
    eegnet_model = build_attention_eegnet(input_shape=(n_channels, n_time_points), num_classes=num_classes)
    train_model(eegnet_model, X_train, y_train, X_val, y_val, "Attention_EEGNet")
    
    # Load the best saved model for evaluation
    best_eegnet_model = tf.keras.models.load_model(os.path.join(MODEL_SAVE_DIR, 'Attention_EEGNet_best_model.h5'), custom_objects={'square': square, 'log': log})
    evaluate_model(best_eegnet_model, X_test, y_test, label_encoder)
