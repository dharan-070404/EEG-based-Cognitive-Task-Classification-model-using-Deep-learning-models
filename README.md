# EEG-based-Cognitive-Task-Classification-model-using-Deep-learning-models

A deep learning framework for classifying cognitive tasks from EEG data using dual-input MobileNetV2 architecture. The model leverages both Phase-Amplitude Coupling (PAC) heatmaps and EEG time-frequency spectrograms to enhance classification performance across 9 cognitive tasks, including baseline. Features include PAC-based Modulation Index extraction, spectrogram generation, and a two-stage training pipeline for optimal model accuracy and generalization.

Key Features

📊 PAC Modulation Index computation for cognitive-relevant frequency bands

 Time-frequency spectrogram generation from 16-channel EEG data

🔀 Dual-branch MobileNetV2 for multimodal image input

🧪 Two-phase training: classification head → partial backbone fine-tuning

🧠 Supports classification across 9 tasks: HVLT, Stroop, TMT, etc.

Tech Stack

Python, NumPy, SciPy, Matplotlib

TensorFlow / Keras

EEG data preprocessed from .mat files with segment-level task separation

Second Approach

Implemented a numerical data input method where we incorporated the concept of PAC and calculated MI features and stored them in the Numpy Array for each eletrode for each task.

Used these MI features as input for a Bi-LSTM model to capture the features and temporal dependencies and finally classify the Task based on the input.

Acheiced an overall 76% accuracy for classifying all 9 tasks.
