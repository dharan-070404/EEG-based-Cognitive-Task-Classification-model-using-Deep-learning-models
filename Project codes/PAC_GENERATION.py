import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Set file paths for EEG
mat_file = r"C:\Users\raja9\OneDrive\Desktop\Projects\PRJ-2 PROJECT\research papers\Cognitive tasks-database\Cognitive tasks-database\MAT\MAT\EEG01R1.mat"
lay_file = r"C:\Users\raja9\OneDrive\Desktop\Projects\PRJ-2 PROJECT\research papers\Cognitive tasks-database\Cognitive tasks-database\LAY\LAY\EEG01R1.mat"

# Load EEG Data
mat_data = scipy.io.loadmat(mat_file)
eeg_signal = mat_data['eegData']
fs = 250  # Sampling frequency (Hz)

# Load Task Start/End Times
lay_data = scipy.io.loadmat(lay_file)
task_times = lay_data['comExpControl']  # Assuming 'comExpControl' stores task times

# Extract Subject Name
subject_name = os.path.basename(mat_file).split('.')[0]  # Extracts EEG

# EEG Channel Labels
channel_labels = ['T5', 'T6', 'T3', 'T4', 'F7', 'F8', 'O1', 'O2', 'P3', 'P4', 'C3', 'C4', 'F3', 'F4', 'Fp1', 'Fp2']
task_names = ['HVLT', 'Stroop', 'Symbol Digit', 'COWAT', 'Trail Making', 'Digit Span', 'Delayed HVLT', 'ModBent', 'Baseline']

# Create Output Folder
output_folder = os.path.join(os.path.dirname(mat_file), f"Spectrograms_{subject_name}")
os.makedirs(output_folder, exist_ok=True)

# Loop through each channel and task
for i, channel in enumerate(channel_labels):
    for j, task in enumerate(task_names):
        start_time = task_times[j, 0]  # Start time (seconds)
        end_time = task_times[j, 1]    # End time (seconds)

        # Convert to sample indices
        start_sample = int(np.squeeze(start_time) * fs)
        end_sample = int(np.squeeze(end_time) * fs)

        # Extract EEG segment
        if start_sample >= end_sample or end_sample > eeg_signal.shape[1]:
            print(f"⚠️ Skipping {task} - {channel}: Invalid or empty EEG segment.")
            continue

        eeg_segment = eeg_signal[i, start_sample:end_sample]

        # Compute Spectrogram
        f, t, Sxx = spectrogram(eeg_segment, fs=fs, nperseg=256, noverlap=128)

        # Ensure dimensions match before plotting
        if Sxx.shape != (len(f), len(t)):
            print(f"⚠️ Dimension mismatch for {task} - {channel}. Skipping.")
            continue

        # Plot Spectrogram
        plt.figure(figsize=(8, 4))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='jet')
        plt.title(f"{subject_name} - {task} - {channel}")
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar(label='Power (dB)')
        
        # Save figure
        save_path = os.path.join(output_folder, f"{subject_name}_{task}_{channel}.png")
        plt.savefig(save_path)
        plt.close()

        print(f"✅ Saved {task} - {channel} spectrogram to {save_path}")

print("🎉 Spectrogram generation complete!")
