import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Define file paths
mat_file = r"C:\Users\raja9\OneDrive\Desktop\Projects\PRJ-2 PROJECT\research papers\Cognitive tasks-database\Cognitive tasks-database\MAT\MAT\EEG01R1.mat"
lay_file = r"C:\Users\raja9\OneDrive\Desktop\Projects\PRJ-2 PROJECT\research papers\Cognitive tasks-database\Cognitive tasks-database\LAY\LAY\EEG01R1.mat"

# Load EEG Data (MAT file)
mat_data = scipy.io.loadmat(mat_file)
EEG_signal = mat_data['eegData']  # Shape: (16, timepoints)
fs = 250  # Sampling frequency in Hz

# Load Task Timestamps (LAY file)
lay_data = scipy.io.loadmat(lay_file)
task_times = lay_data['comExpControl']  # Start-End times in seconds

# Define Channel Labels
channel_labels = ['T5', 'T6', 'T3', 'T4', 'F7', 'F8', 'O1', 'O2', 'P3', 'P4', 'C3', 'C4', 'F3', 'F4', 'Fp1', 'Fp2']

# Define Cognitive Tasks
task_names = ['HVLT', 'Stroop', 'Symbol Digit', 'COWAT', 'Trail Marking', 'Digit Span', 'HVLT Delayed', 'ModBent']

# Output Folder
output_dir = "Spectrograms_EEG01R1"
os.makedirs(output_dir, exist_ok=True)

# Generate Spectrograms for Each Task & Each Channel
for task_idx, task_name in enumerate(task_names):
    start_sample = int(task_times[task_idx, 0] * fs)
    end_sample = int(task_times[task_idx, 1] * fs)
    
    for ch_idx, ch_name in enumerate(channel_labels):
        eeg_segment = EEG_signal[ch_idx, start_sample:end_sample]  # Extract Task-Specific EEG Data
        
        # Generate Spectrogram
        f, t, Sxx = spectrogram(eeg_segment, fs, nperseg=256, noverlap=128)
        
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='jet')
        plt.colorbar(label='Power/Frequency (dB/Hz)')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title(f'{ch_name} - {task_name}')
        
        # Save Image
        save_path = os.path.join(output_dir, f'EEG01R1_{ch_name}_{task_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

print("Spectrogram generation completed!")
