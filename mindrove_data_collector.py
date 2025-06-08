import pylsl
import numpy as np
import time
import os
import logging
from datetime import datetime
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from mindrove.data_filter import DataFilter, FilterTypes, DetrendOperations
import pickle # Added this import

# Configuration for LSL output streams
EEG_OUTPUT_STREAM_NAME = 'MindRove_Filtered_EEG'
IMU_OUTPUT_STREAM_NAME = 'MindRove_Raw_IMU'
MARKER_INPUT_STREAM_NAME = 'MindRove_Markers' # From Pygame paradigm
OUTPUT_DIR = 'raw_eeg_data' # Directory to save raw data (useful for debugging/archiving)

# MindRove BoardShim logging
BoardShim.enable_dev_board_logger()
logging.basicConfig(level=logging.DEBUG)

def setup_mindrove_board():
    """
    Sets up and initializes the MindRove BoardShim.
    
    Returns:
        tuple: (board_shim, exg_channels, accel_channels, gyro_channels, sampling_rate)
               Returns None for board_shim if connection fails.
    """
    params = MindRoveInputParams()
    # You might need to set params.ip_address or params.mac_address if not using auto-discovery
    # params.ip_address = 'YOUR_MINDROVE_IP'
    # params.mac_address = 'YOUR_MINDROVE_MAC'

    board_shim = None
    try:
        board_shim = BoardShim(BoardIds.MINDROVE_WIFI_BOARD, params)
        board_shim.prepare_session()
        board_shim.start_stream()
        
        board_id = board_shim.get_board_id()
        exg_channels = BoardShim.get_exg_channels(board_id)
        accel_channels = BoardShim.get_accel_channels(board_id) # Assuming IMU channels exist
        gyro_channels = BoardShim.get_gyro_channels(board_id)
        sampling_rate = BoardShim.get_sampling_rate(board_id)

        print(f"MindRove Board connected. Board ID: {board_id}")
        print(f"EEG Channels: {exg_channels}, Sampling Rate: {sampling_rate} Hz")
        print(f"Accelerometer Channels: {accel_channels}, Gyroscope Channels: {gyro_channels}")
        
        return board_shim, exg_channels, accel_channels, gyro_channels, sampling_rate

    except Exception as e:
        logging.error(f"Error setting up MindRove Board: {e}")
        if board_shim is not None and board_shim.is_prepared():
            board_shim.release_session()
        return None, [], [], [], 0

def create_lsl_outlets(eeg_channels_count, imu_channels_count, srate):
    """
    Creates LSL Outlets for EEG and IMU data.
    
    Args:
        eeg_channels_count (int): Number of EEG channels.
        imu_channels_count (int): Number of IMU channels (accel + gyro).
        srate (int): Sampling rate for the streams.
        
    Returns:
        tuple: (eeg_outlet, imu_outlet)
    """
    eeg_info = pylsl.StreamInfo(EEG_OUTPUT_STREAM_NAME, 'EEG', eeg_channels_count, srate, 'float32', 'mindrove_eeg_uid')
    eeg_outlet = pylsl.StreamOutlet(eeg_info)
    print(f"LSL EEG stream '{EEG_OUTPUT_STREAM_NAME}' ready.")

    imu_info = pylsl.StreamInfo(IMU_OUTPUT_STREAM_NAME, 'IMU', imu_channels_count, srate, 'float32', 'mindrove_imu_uid')
    imu_outlet = pylsl.StreamOutlet(imu_info)
    print(f"LSL IMU stream '{IMU_OUTPUT_STREAM_NAME}' ready.")
    
    return eeg_outlet, imu_outlet

def collect_and_stream_data():
    """
    Collects data from MindRove board, applies filters, and streams via LSL.
    Also listens for markers from Pygame and saves them.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    board_shim, exg_channels, accel_channels, gyro_channels, sampling_rate = setup_mindrove_board()
    if board_shim is None:
        print("Failed to connect to MindRove board. Exiting.")
        return

    eeg_outlet, imu_outlet = create_lsl_outlets(len(exg_channels), len(accel_channels) + len(gyro_channels), sampling_rate)

    # LSL Inlet for Markers from Pygame
    marker_inlet = None
    try:
        print(f"Attempting to find and connect to LSL marker stream: '{MARKER_INPUT_STREAM_NAME}'...")
        marker_streams = pylsl.resolve_byprop('name', MARKER_INPUT_STREAM_NAME, timeout=10) # Increased timeout
        if marker_streams:
            marker_inlet = pylsl.StreamInlet(marker_streams[0])
            print(f"Successfully connected to LSL marker stream: '{MARKER_INPUT_STREAM_NAME}'.")
        else:
            print(f"WARNING: LSL marker stream '{MARKER_INPUT_STREAM_NAME}' not found after 10s. Markers will not be collected.")
            print("Please ensure `training_paradigm_pygame.py` is running BEFORE `mindrove_data_collector.py`.")
    except Exception as e:
        print(f"Error setting up marker inlet: {e}")

    # Data collection buffers for saving to file
    all_eeg_data = []
    all_eeg_timestamps = []
    all_imu_data = []
    all_imu_timestamps = []
    all_markers = [] # This list stores (marker_value, timestamp)

    print("Starting data collection and LSL streaming. Press Ctrl+C to stop.")
    start_time = time.time()

    try:
        marker_pull_count = 0 # Debug counter
        while True:
            # Get data from the board (non-blocking, pulls available data)
            data = board_shim.get_board_data() 
            
            if data.shape[1] == 0: # No new data from board
                time.sleep(0.001)
                continue

            # Extract EEG and IMU data
            eeg_data_chunk = data[exg_channels, :].T # Transpose to (samples, channels)
            
            imu_data_chunk = []
            if accel_channels:
                imu_data_chunk.append(data[accel_channels, :])
            if gyro_channels:
                imu_data_chunk.append(data[gyro_channels, :])
            imu_data_chunk = np.concatenate(imu_data_chunk, axis=0).T if imu_data_chunk else np.array([]) # (samples, imu_channels)

            # Get timestamp channel (assuming it's the last row in BrainFlow data)
            timestamp_channel_idx = BoardShim.get_num_rows(board_shim.get_board_id()) - 1
            timestamps_chunk = data[timestamp_channel_idx, :].flatten()

            # Apply filters to EEG data (channel by channel)
            processed_eeg_chunk = np.copy(eeg_data_chunk)
            for i in range(processed_eeg_chunk.shape[1]): # Iterate over channels
                # Detrend (re-enabled as it's standard practice)
                DataFilter.detrend(processed_eeg_chunk[:, i], DetrendOperations.CONSTANT.value)
                # Bandpass 51-100 Hz (as per user's snippet)
                #DataFilter.perform_bandpass(processed_eeg_chunk[:, i], sampling_rate, 10.0, 240.0, 2,
                #                            FilterTypes.BUTTERWORTH.value, 0)
                # Bandstop 50 Hz
                #DataFilter.perform_bandstop(processed_eeg_chunk[:, i], sampling_rate, 48.0, 52.0, 2,
                #                            FilterTypes.BUTTERWORTH.value, 0)
                # Bandstop 60 Hz
                #DataFilter.perform_bandstop(processed_eeg_chunk[:, i], sampling_rate, 58.0, 62.0, 2,
                #                            FilterTypes.BUTTERWORTH.value, 0)
            
            # Push processed EEG data to LSL
            for i in range(processed_eeg_chunk.shape[0]): # Iterate over samples
                eeg_outlet.push_sample(processed_eeg_chunk[i, :].tolist(), timestamps_chunk[i])

            # Push raw IMU data to LSL
            if imu_data_chunk.size > 0:
                for i in range(imu_data_chunk.shape[0]):
                    imu_outlet.push_sample(imu_data_chunk[i, :].tolist(), timestamps_chunk[i])

            # Collect markers from Pygame (non-blocking)
            if marker_inlet: # Only try to pull if inlet is established
                marker_samples, marker_ts = marker_inlet.pull_chunk(timeout=0.0)
                if marker_samples:
                    for s, ts in zip(marker_samples, marker_ts):
                        all_markers.append((s[0], ts)) # Marker is usually a single string/int
                        marker_pull_count += 1
                        # --- DEBUGGING MARKERS ---
                        if marker_pull_count % 10 == 0: # Print every 10th marker to avoid spam
                            print(f"  [Marker Debug] Pulled marker: {s[0]} at {ts}. Total collected: {len(all_markers)}")
                        # --- END DEBUGGING MARKERS ---

            # Append to buffers for saving (optional, for post-hoc analysis)
            all_eeg_data.append(eeg_data_chunk) # Save raw EEG for archiving
            all_eeg_timestamps.append(timestamps_chunk)
            if imu_data_chunk.size > 0:
                all_imu_data.append(imu_data_chunk)
                all_imu_timestamps.append(timestamps_chunk)

    except KeyboardInterrupt:
        print("\nStopping data collection.")
    except Exception as e:
        logging.error(f"An error occurred during streaming: {e}")
    finally:
        if board_shim is not None and board_shim.is_prepared():
            board_shim.stop_stream()
            board_shim.release_session()
            print("MindRove stream stopped and session released.")
        
        # LSL StreamOutlet and StreamInlet objects do not have a .close_stream() method.
        # They are typically closed when the object goes out of scope or the program terminates.

        # Save collected data to file (optional)
        if all_eeg_data:
            combined_eeg_data = np.concatenate(all_eeg_data, axis=0)
            combined_eeg_timestamps = np.concatenate(all_eeg_timestamps, axis=0)
            combined_imu_data = np.concatenate(all_imu_data, axis=0) if all_imu_data else np.array([])
            combined_imu_timestamps = np.concatenate(all_imu_timestamps, axis=0) if all_imu_timestamps else np.array([])

            collected_data = {
                'eeg_data': combined_eeg_data,
                'eeg_timestamps': combined_eeg_timestamps,
                'eeg_srate': sampling_rate,
                'eeg_channel_names': [f'EEG_{ch}' for ch in exg_channels],
                'imu_data': combined_imu_data,
                'imu_timestamps': combined_imu_timestamps,
                'imu_srate': sampling_rate, # Assuming IMU has same srate
                'imu_channel_names': [f'Accel_{ch}' for ch in accel_channels] + [f'Gyro_{ch}' for ch in gyro_channels],
                'markers': all_markers, # Save the collected markers
                'start_time_unix': start_time,
                'collection_duration': time.time() - start_time,
            }

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filepath = os.path.join(OUTPUT_DIR, f"mindrove_raw_session_{timestamp_str}.pkl")
            with open(output_filepath, 'wb') as f:
                pickle.dump(collected_data, f)
            print(f"Raw data saved to {output_filepath}")
        else:
            print("No EEG data was collected during this session. No .pkl file saved.")
        
        if not all_markers:
            print("WARNING: No markers were collected during this session. Check `training_paradigm_pygame.py` and LSL connection.")


if __name__ == "__main__":
    print("This script connects to MindRove, applies filters, and streams data via LSL.")
    print("Ensure `training_paradigm_pygame.py` is running BEFORE starting this script to send markers.")
    collect_and_stream_data()

