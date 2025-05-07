import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Define file paths
mat_file = r"C:\Users\dhara\Documents\PROJECT-2\EEG DATASET\MAT\EEG391.mat"
lay_file = r"C:\Users\dhara\Documents\PROJECT-2\EEG DATASET\LAY\EEG391.mat"
output_folder = r"C:\Users\dhara\Documents\PROJECT-2\PAC_Normalized_heatmap\PAC_Normalized_heatmap\EEG391"

# Load EEG Data
mat_data = scipy.io.loadmat(mat_file)
eeg_signal = mat_data['eegData']
fs = 250  # Sampling frequency (Hz)

# Load Task Start/End Times
lay_data = scipy.io.loadmat(lay_file)
task_times = lay_data['comExpControl']  # Assuming 'comExpControl' stores task times

# Extract Subject Name
subject_name = os.path.basename(mat_file).split('.')[0]

# Task Names
task_names = ['HVLT', 'Stroop', 'Symbol Digit', 'COWAT', 'Trail Making', 'Digit Span', 'Delayed HVLT', 'ModBent', 'Baseline']

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Loop through each task
for j, task in enumerate(task_names):
    start_time = task_times[j, 0]  # Start time (seconds)
    end_time = task_times[j, 1]    # End time (seconds)

    start_sample = int(np.squeeze(start_time) * fs)
    end_sample = int(np.squeeze(end_time) * fs)

    #Skip if invalid segment
    if start_sample >= end_sample or end_sample > eeg_signal.shape[1]:
        print(f"Warning: Skipping {task} (Start={start_sample}, End={end_sample})")
        continue

    #Skip if EEG segment is too short
    if (end_sample - start_sample) < 256:
        print(f"Warning: Skipping {task}: EEG segment too short ({end_sample - start_sample} samples)")
        continue

    #Compute spectrogram for each channel
    all_Sxx = []
    for i in range(eeg_signal.shape[0]):  # Loop over 16 EEG channels
        f, t, Sxx = spectrogram(eeg_signal[i, start_sample:end_sample], fs=fs, nperseg=256, noverlap=128)

        #Skip empty spectrograms
        if Sxx.size == 0:
            print(f"Warning: Skipping {task} (Channel {i}): Empty spectrogram")
            continue

        all_Sxx.append(Sxx)

    # Convert to NumPy Array
    if len(all_Sxx) == 0:
        print(f"Error: Skipping {task}: No valid spectrograms")
        continue

    all_Sxx = np.array(all_Sxx)  # Shape: (Valid Channels, freq_bins, time_bins)

    # Step 1: Average the Spectrograms Across Channels
    mean_Sxx = np.mean(all_Sxx, axis=0)

    # Step 2: Convert to dB Scale
    mean_Sxx_dB = 10 * np.log10(mean_Sxx + 1e-10)

    #Ensure correct dimensions before plotting
    if mean_Sxx_dB.shape[0] == 0 or mean_Sxx_dB.shape[1] == 0:
        print(f"Error: Skipping {task}: Spectrogram shape mismatch")
        continue

    # Plot and Save Spectrogram
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(t, f, mean_Sxx_dB, shading='gouraud', cmap='jet')
    plt.axis('off')  # Hide axis
    plt.gca().set_xticks([])  # Remove x-axis ticks
    plt.gca().set_yticks([])  # Remove y-axis ticks

    save_path = os.path.join(output_folder, f"{subject_name}_{task}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"saved mean spectrogram for {task} to {save_path}")

print("Spectrogram generation complete!")
