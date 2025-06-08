import mne
import numpy as np
import os
import pickle
from datetime import datetime, timezone
import glob # Used for finding latest files
import matplotlib.pyplot as plt # Import matplotlib

# --- Configuration (Adjust as needed) ---
RAW_DATA_DIR = 'raw_eeg_data'
PROCESSED_EPOCHS_DIR = 'preprocessed_epochs'

# Ensure output directory exists
os.makedirs(PROCESSED_EPOCHS_DIR, exist_ok=True)

# Define a mapping from the generic channel names (EEG_X) to standard 10-20 names
# Based on user clarification: 6 main electrodes along T3-T4, and 2 references at A1/A2.
# Assuming EEG_0 to EEG_5 are the active electrodes.
# Using T7/T8 for T3/T4 respectively, as T7/T8 are the current 10-20 standard names.
CHANNEL_MAP_1020 = {
    'EEG_0': 'T3',  # Left Temporal (formerly T3)
    'EEG_1': 'C3',  # Left Central (Motor Cortex)
    'EEG_2': 'Cz',  # Central (Midline Motor Cortex)
    'EEG_3': 'Pz',  # Right Central (Motor Cortex)
    'EEG_4': 'C4',  # Right Temporal (formerly T4)
    'EEG_5': 'T4',  # Midline Parietal (often a good central reference point or active channel)
    # Channels EEG_6 and EEG_7 (corresponding to A1/A2 references) will not be
    # included in this map, as they are not typically plotted as active EEG channels
    # for classification tasks with a standard montage. They will be implicitly excluded
    # by the channel picking logic.
}

def preprocess_eeg_data(raw_data_filepath):
    """
    Loads raw EEG data, preprocesses it (filtering, detrending, ICA),
    epochs it based on markers, and saves the epochs.

    Args:
        raw_data_filepath (str): Path to the raw .pkl data file.
    """
    print(f"\n--- Processing: {raw_data_filepath} ---")
    
    with open(raw_data_filepath, 'rb') as f:
        collected_data = pickle.load(f)

    eeg_data = collected_data['eeg_data'] # Shape (samples, channels)
    eeg_timestamps = collected_data['eeg_timestamps']
    eeg_srate = collected_data['eeg_srate']
    # eeg_channel_names from collected_data might contain more than 6 or have generic names
    original_eeg_channel_names = collected_data['eeg_channel_names'] 
    markers = collected_data['markers'] # List of (marker_string, timestamp) tuples

    print(f"Loaded EEG data: {eeg_data.shape[0]} samples, {eeg_data.shape[1]} channels, {eeg_srate} Hz")
    print(f"Original Channel Names from Data: {original_eeg_channel_names}")
    
    # --- DEBUGGING MARKERS---
    print(f"Markers loaded: {markers}") 
    print(f"Number of  markers loaded: {len(markers)}") 
    # --- END DEBUGGING MARKERS ---
    # --- DEBUGGING TimeStamps---
    print(f"TimeStamps loaded: {markers[0][1]}") 
    print(f"Number of  stamps loaded: {len(eeg_timestamps)}") 
    # --- END DEBUGGING MARKERS ---
    # MNE expects data in Volts (V), not microvolts (uV)
    # Convert from uV to V: 1 uV = 1e-6 V
    # Assuming eeg_data is (n_samples, n_channels) from the collector
    eeg_data_mne_format = eeg_data.T * 1e-6 

    # Convert Unix timestamp (float) to datetime object with UTC timezone
    measurement_date = datetime.fromtimestamp(markers[0][1], tz=timezone.utc)

    # Identify which of the original channels match our desired 10-20 map
    # and create the new info object with only those channels and their 10-20 names.
    
    new_eeg_channel_names = [] # Names for the new MNE info object (10-20 names)
    channel_indices_to_keep = [] # Indices from original_eeg_channel_names to select from eeg_data_mne_format

    for i, original_name in enumerate(original_eeg_channel_names):
        if original_name in CHANNEL_MAP_1020:
            new_eeg_channel_names.append(CHANNEL_MAP_1020[original_name])
            channel_indices_to_keep.append(i)
        else:
            print(f"Warning: Channel '{original_name}' not found in CHANNEL_MAP_1020. It will be excluded from MNE Raw object.")

    if not new_eeg_channel_names:
        raise ValueError("No EEG channels found that match the defined 10-20 mapping. Cannot create Raw object. Please check your collected data's channel names and CHANNEL_MAP_1020.")

    # Create MNE Info object with only the selected and renamed channels
    info = mne.create_info(
        ch_names=new_eeg_channel_names,
        sfreq=eeg_srate,
        ch_types='eeg'
    )
    
    
    # Pick the corresponding data columns (channels) from the original eeg_data_mne_format
    # eeg_data_mne_format is (n_channels, n_samples)
    # We need to select rows based on channel_indices_to_keep
    processed_eeg_data_for_raw = eeg_data_mne_format[channel_indices_to_keep, :]


    # Create Raw object with only the desired channels and their 10-20 names
    raw = mne.io.RawArray(processed_eeg_data_for_raw, info, first_samp=0, verbose=False) 
    print(f"MNE Raw object created with {len(raw.info['ch_names'])} channels: {raw.info['ch_names']}")
    
    # Set the standard 10-20 montage
    print("Setting standard_1020 montage...")
    try:
        # We can use on_missing='raise' now because we've carefully selected channels.
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_meas_date(measurement_date)
        raw.set_montage(montage, on_missing='raise', verbose=False)
        print("Montage set successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to set montage after channel renaming and picking: {e}")
        print("Please check CHANNEL_MAP_1020 and ensure its values (e.g., 'Fp1', 'C3') are valid 10-20 electrode names.")
        raise # Re-raise the exception as this is a fundamental issue for spatial analysis.

    # --- Capture data BEFORE full preprocessing for visualization ---
    # Create a copy of the raw object after montage for "before" comparison
    raw_before_full_processing = raw.copy()
    sfreq = raw.info['sfreq']
    duration_to_plot = min(5.0, raw.times[-1]) # Plot up to 5 seconds, or less if recording is shorter
    
    # Extract the data array for plotting, convert to microvolts for readability
    data_before = raw_before_full_processing.get_data(start=0, stop=int(duration_to_plot * sfreq), picks='eeg') * 1e6 
    times_before = raw_before_full_processing.times[0:int(duration_to_plot * sfreq)]


    # --- Preprocessing Steps ---

    # 1. Bandpass Filter (e.g., 0.5 - 40 Hz for typical EEG analysis)
    # The MindRove collector already applied 51-100Hz bandpass and 50/60Hz notch,
    # but a broader filter for analysis is often useful.
    # If the previous filtering was aggressive, this might be redundant or you might adjust
    print("Applying bandpass filter (0.5-240 Hz)...")
    raw.filter(0.5, 240., fir_design='firwin', verbose=False) # Changed 240. back to 40. Hz for typical EEG

    # 2. Re-referencing (e.g., to average reference)
    print("Applying average reference...")
    raw.set_eeg_reference('average', verbose=False)

    # 3. ICA for Artifact Removal (e.g., eye blinks, muscle artifacts)
    # ICA works best on continuous, filtered data.
    # It requires a sufficient amount of data and can be computationally intensive.
    print("Performing ICA for artifact removal (this may take a while)...")
    raw_for_ica = raw.copy().filter(1., None, fir_design='firwin', verbose=False) # Highpass for ICA
    ica = mne.preprocessing.ICA(n_components=0.95, random_state=42, verbose=False) # Keep 95% variance
    try:
        ica.fit(raw_for_ica)
        
        # --- IMPORTANT: Manual ICA component exclusion is crucial here ---
        # For a practical system, you would visually inspect ICA components using:
        ica.plot_sources(raw)
        ica.plot_components()
        ica.exclude = [list_of_bad_component_indices]
        # For an automated script, this is a placeholder. Without EOG channels,
        # automatic detection is harder.
        # For now, no automatic exclusion.
        # Example: ica.exclude = [0] # Exclude component 0 if it looks like an eye blink

        ica.apply(raw, verbose=False)
        print("ICA applied to raw data.")
    except Exception as e:
        print(f"ICA fitting failed or skipped: {e}. Proceeding without ICA artifact removal.")
        # This can happen if data quality is very poor or too short for ICA to converge.

    # --- Capture data AFTER full preprocessing for visualization ---
    # raw at this point is the fully preprocessed data
    data_after = raw.get_data(start=0, stop=int(duration_to_plot * sfreq), picks='eeg') * 1e6 # Convert back to microvolts
    times_after = raw.times[0:int(duration_to_plot * sfreq)]

    # --- Plotting ---
    print("Plotting data before and after preprocessing...")
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True, sharey=True)
    fig.suptitle('EEG Data Before and After Preprocessing (First few seconds)', fontsize=16)

    # Plot before processing
    for i, ch_name in enumerate(new_eeg_channel_names):
        # Offset channels for better visualization
        offset = i * 200 # Adjust offset as needed based on signal amplitude
        axes[0].plot(times_before, data_before[i, :] + offset, label=ch_name, linewidth=0.7)
    axes[0].set_title('Before Preprocessing (After Montage, Before Filtering/ICA)')
    axes[0].set_ylabel('Amplitude (µV)')
    axes[0].legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    axes[0].grid(True)

    # Plot after processing
    for i, ch_name in enumerate(new_eeg_channel_names):
        offset = i * 200 # Use same offset for comparison
        axes[1].plot(times_after, data_after[i, :] + offset, label=ch_name, linewidth=0.7)
    axes[1].set_title('After Preprocessing (Filtered, Re-referenced, ICA-cleaned)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude (µV)')
    axes[1].legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    axes[1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()


    # --- Epoching ---
    print("Epoching data based on markers...")
    
    # Convert markers to MNE events format
    events = []
    event_id = {}
    current_event_id = 1
    
    for marker_str, marker_ts in markers:
        # Ensure marker_str is a string
        marker_str = str(marker_str)
        
        # MNE expects event IDs as integers
        if marker_str not in event_id:
            event_id[marker_str] = current_event_id
            current_event_id += 1
        
        # Find sample index from timestamp relative to raw's start time
        time_relative_to_raw_start = marker_ts - markers[0][1]
        time_relative_to_raw_start = max(0, time_relative_to_raw_start) 

        # Calculate sample index
        # Ensure sample_index is not outside the bounds of raw data
        sample_index_float = time_relative_to_raw_start * raw.info['sfreq']
        sample_index = int(round(sample_index_float))

        # Check if sample_index is within the valid range of raw data
        if 0 <= sample_index < raw.n_times:
            events.append([sample_index, 0, event_id[marker_str]])
            # --- DEBUGGING MARKERS (from previous update) ---
            print(f"  [Epoching Debug] Processed marker: string='{marker_str}', timestamp={marker_ts}, sample_index={sample_index}, MNE_id={event_id[marker_str]}")
            # --- END DEBUGGING MARKERS ---
        else:
            print(f"  [Epoching Debug] WARNING: Marker at timestamp {marker_ts} (sample {sample_index}) is outside raw data range. Skipping.")


    if not events:
        print("No valid events found in markers. Skipping epoching.")
        return # Exit if no events to epoch
    
    events = np.array(events)
    print(f"Found {len(events)} events.")
    print(f"Event IDs: {event_id}")

    # Define epoching parameters
    tmin = -1.0  # Start time of epoch (e.g., 1 second before marker)
    tmax = 3.0   # End time of epoch (e.g., 3 seconds after marker)

    # Create Epochs - FIX: Added event_repeated='drop'
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        preload=True, verbose=False, baseline=(None, 0), event_repeated='drop') 
    print(f"Created {len(epochs)} epochs.")
    
    # Drop bad epochs (optional, based on amplitude thresholds)
    # epochs.drop_bad(reject=dict(eeg=150e-6)) # Example: reject if EEG exceeds 150uV peak-to-peak
    # print(f"Dropped {len(epochs.drop_log)} bad epochs.")

    # Save epochs
    output_filename = os.path.basename(raw_data_filepath).replace('.pkl', '-epo.fif')
    output_filepath_epochs = os.path.join(PROCESSED_EPOCHS_DIR, output_filename)
    epochs.save(output_filepath_epochs, overwrite=True, verbose=False)
    print(f"Epochs saved to {output_filepath_epochs}")


if __name__ == "__main__":
    print("Starting EEG Preprocessor.")
    
    # Process all .pkl files in the RAW_DATA_DIR
    raw_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.pkl')]
    
    if not raw_files:
        print(f"No .pkl files found in '{RAW_DATA_DIR}'. Please ensure data collector has run.")
    else:
        for filename in raw_files:
            full_path = os.path.join(RAW_DATA_DIR, filename)
            try:
                preprocess_eeg_data(full_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                import traceback
                traceback.print_exc()

