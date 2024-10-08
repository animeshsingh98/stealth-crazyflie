import noisereduce as nr
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

# Load the audio file
audio_file = 'output_flying.wav'  # Replace with your file
y, sr = librosa.load(audio_file, sr=None)

# Extract the first second (used for noise profiling)
noise_profile = y[:sr]  # First second of audio

# Apply noise reduction
y_reduced_noise = nr.reduce_noise(y=y, sr=sr, y_noise=noise_profile)

# Save the resulting audio to a new file
output_file = 'noise_reduce_drone.wav'
sf.write(output_file, y_reduced_noise, sr)

# Plot original vs filtered waveforms
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.title('Original Audio Waveform')
librosa.display.waveshow(y, sr=sr)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.title('Filtered Audio Waveform (Noise Reduced)')
librosa.display.waveshow(y_reduced_noise, sr=sr)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

print(f"Filtered audio saved as {output_file}")
