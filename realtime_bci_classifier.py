import pylsl
import numpy as np
import time
import tensorflow as tf
import mne # MNE is not strictly needed for real-time processing here, but good for context
import threading
from collections import deque
from sklearn.preprocessing import StandardScaler
import os
import sys

# Add the directory containing model_architectures.py to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_architectures import square, log # Import custom activation functions if used

# Configuration
EEG_INPUT_STREAM_NAME = 'MindRove_Filtered_EEG' # Now receiving already filtered data
IMU_INPUT_STREAM_NAME = 'MindRove_Raw_IMU' # Still receiving raw IMU
PRETRAINED_MODEL_PATH = os.path.join('trained_models', 'Attention_EEGNet_best_model.h5') # Or Hybrid_CNN_LSTM_best_model.h5
CLASSIFICATION_INTERVAL_SEC = 0.1 # Classify every 100ms
EPOCH_DURATION_SEC = 1.0 # Duration of the window to classify (e.g., 1 second)
CLASSIFICATION_OVERLAP_SEC = 0.5 # Overlap between classification windows
EEG_CHANNELS = 6 # MindRove EEG channels (adjust if your board has different number)
EEG_SRATE = 500 # Hz (MindRove's sampling rate)

# LSL Outlet for Classified Commands (for game_controller.py)
COMMAND_STREAM_NAME = 'BCI_Commands'
command_outlet = None

# Global buffer for EEG data
eeg_buffer = deque()
last_classification_time = 0

# Placeholder for Standardization parameters (fit on training data)
scaler = None # A StandardScaler object from sklearn.preprocessing

# Load the trained model and label encoder
model = None
label_encoder = None
# IMPORTANT: Ensure LABELS matches the order of classes used during training!
# FIX: Added 'REST' to the labels
LABELS = ['DOWN', 'LEFT', 'PULL', 'PUSH', 'REST', 'RIGHT', 'UP'] 

def load_bci_resources():
    """Loads the pre-trained model and sets up label encoder."""
    global model, label_encoder, scaler
    try:
        # Load model with custom objects if any
        model = tf.keras.models.load_model(PRETRAINED_MODEL_PATH, custom_objects={'square': square, 'log': log})
        print(f"Loaded model from {PRETRAINED_MODEL_PATH}")

        # Recreate label encoder. Ensure LABELS matches order from training.
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        label_encoder.fit(LABELS) # Fit with the exact labels from training
        print(f"Label encoder fitted with classes: {label_encoder.classes_}")

        # IMPORTANT: In a real system, you MUST save and load the scaler fitted on your training data.
        # For this example, we'll just initialize a StandardScaler without fitting, meaning no real scaling occurs
        # if you don't explicitly load a fitted scaler.
        scaler_path = 'scaler.pkl' # Assuming this is where model_trainer saves the scaler
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print("Loaded StandardScaler.")
        else:
            print("WARNING: Scaler not found at 'scaler.pkl'. Real-time performance may be impacted without proper scaling.")
            scaler = StandardScaler() # Initialize an empty scaler if not found

    except Exception as e:
        print(f"Error loading BCI resources: {e}")
        model = None
        label_encoder = None
        scaler = None
        # Do NOT exit here, allow the program to run and report issues.
        # exit() 

def init_command_outlet():
    """Initializes the LSL outlet for sending classified commands."""
    global command_outlet
    info = pylsl.StreamInfo(COMMAND_STREAM_NAME, 'Commands', 1, 0, 'string', 'bci_commands_uid')
    command_outlet = pylsl.StreamOutlet(info)
    print(f"LSL command stream '{COMMAND_STREAM_NAME}' ready.")

def real_time_preprocess(eeg_chunk):
    """
    Performs real-time preprocessing on a chunk of EEG data.
    Assumes initial filtering (bandpass, notch, detrend) has already been applied
    by the data collection script (mindrove_data_collector.py).
    
    Args:
        eeg_chunk (np.ndarray): A 2D numpy array (samples x channels) of already filtered EEG data.
        
    Returns:
        np.ndarray: Processed EEG data, ready for classification.
    """
    # 1. Standardization (if scaler is fitted)
    if scaler is not None:
        try:
            # Ensure the chunk has at least as many samples as features (channels) for scaling
            # if the scaler was fitted on (samples, features)
            if eeg_chunk.shape[0] < eeg_chunk.shape[1] and eeg_chunk.shape[0] > 0:
                print("Warning: EEG chunk has fewer samples than channels for scaling. This might indicate insufficient data for classification window.")
                # You might want to pad or handle this case specifically, or just return original.
                # For now, if the shape is problematic for scaler, it will likely raise an error.
            scaled_chunk = scaler.transform(eeg_chunk) 
        except Exception as e:
            print(f"Error during scaling: {e}. Skipping scaling for this chunk.")
            scaled_chunk = eeg_chunk
    else:
        scaled_chunk = eeg_chunk

    # Ensure shape for model input: (1, channels, time_points, 1)
    # Data is (samples, channels), need (channels, samples) for model
    # Then reshape to (1, channels, time_points, 1)
    processed_chunk = scaled_chunk.T # (channels, samples)
    processed_chunk = processed_chunk[np.newaxis, :, :, np.newaxis] # (1, channels, time_points, 1)
    
    return processed_chunk

def classify_eeg_chunk(eeg_chunk):
    """
    Classifies a preprocessed EEG chunk using the loaded deep learning model.
    
    Args:
        eeg_chunk (np.ndarray): Preprocessed EEG data (1, channels, time_points, 1).
        
    Returns:
        str: Predicted command (e.g., "LEFT", "PUSH", "REST") or "NO_COMMAND".
    """
    if model is None or label_encoder is None:
        # print("Model or label encoder not loaded. Cannot classify.") # Avoid spamming console
        return "NO_COMMAND"
    
    try:
        predictions = model.predict(eeg_chunk, verbose=0)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_command = label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # Optional: Confidence thresholding
        confidence = np.max(predictions)
        if confidence < 0.6: # Example threshold, adjust based on performance
            return "NO_COMMAND"
        
        return predicted_command
    except Exception as e:
        print(f"Error during classification: {e}")
        return "NO_COMMAND"

def lsl_listener_thread():
    """
    Thread function to listen for EEG data from LSL and buffer it.
    """
    global eeg_buffer, last_classification_time

    print(f"Looking for {EEG_INPUT_STREAM_NAME} stream...")
    eeg_streams = pylsl.resolve_byprop('name', EEG_INPUT_STREAM_NAME, timeout=30)
    if not eeg_streams:
        print(f"No {EEG_INPUT_STREAM_NAME} stream found. Is `mindrove_data_collector.py` running and streaming?")
        return
    
    eeg_inlet = pylsl.StreamInlet(eeg_streams[0])
    print(f"Connected to {EEG_INPUT_STREAM_NAME} stream. Starting data acquisition.")

    samples_per_epoch = int(EPOCH_DURATION_SEC * EEG_SRATE)
    samples_per_interval = int(CLASSIFICATION_INTERVAL_SEC * EEG_SRATE)
    
    while True:
        # Pull data in chunks (non-blocking)
        samples, timestamps = eeg_inlet.pull_chunk(timeout=0.0)
        
        if samples:
            # Add new samples to the buffer
            for sample_row in samples:
                eeg_buffer.append(sample_row[:EEG_CHANNELS]) # Only take EEG channels

            # Manage buffer size to avoid excessive memory usage
            # Keep enough data for the current epoch duration + some overlap history
            max_buffer_size = int(EPOCH_DURATION_SEC * EEG_SRATE) + int(CLASSIFICATION_OVERLAP_SEC * EEG_SRATE) # Ensures enough history
            while len(eeg_buffer) > max_buffer_size:
                eeg_buffer.popleft() # Remove oldest samples
            
            # --- Real-time Classification Logic ---
            current_time = time.time()
            if (current_time - last_classification_time) >= CLASSIFICATION_INTERVAL_SEC:
                # Check if enough data for an epoch is available
                if len(eeg_buffer) >= samples_per_epoch:
                    # Take the last `samples_per_epoch` samples from the buffer
                    eeg_segment = np.array(list(eeg_buffer)[-samples_per_epoch:])
                    
                    # Preprocess the segment
                    preprocessed_segment = real_time_preprocess(eeg_segment)
                    
                    # Classify
                    predicted_command = classify_eeg_chunk(preprocessed_segment)
                    
                    if predicted_command != "NO_COMMAND":
                        print(f"Predicted Command: {predicted_command}")
                        if command_outlet:
                            command_outlet.push_sample([predicted_command])
                    
                    last_classification_time = current_time
                else:
                    # print(f"Not enough data in buffer for an epoch. Need {samples_per_epoch}, have {len(eeg_buffer)}.") # Debugging
                    pass # Not enough data for a full epoch yet
        
        time.sleep(0.001) # Small sleep to prevent busy-waiting

if __name__ == "__main__":
    # Ensure this script runs *after* model_trainer.py has saved a scaler and a model.
    load_bci_resources()
    init_command_outlet()

    # Start the LSL listener in a separate thread
    listener_thread = threading.Thread(target=lsl_listener_thread)
    listener_thread.daemon = True # Daemonize thread so it exits with main program
    listener_thread.start()

    print("\nReal-time BCI classifier running. Press Ctrl+C to exit.")
    print(f"Listening for EEG on LSL stream: '{EEG_INPUT_STREAM_NAME}'")
    print(f"Sending commands on LSL stream: '{COMMAND_STREAM_NAME}'")

    try:
        while True:
            time.sleep(1) # Main thread keeps running
    except KeyboardInterrupt:
        print("\nExiting real-time BCI classifier.")

