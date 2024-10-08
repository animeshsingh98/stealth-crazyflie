import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

# Load the two audio files
drone_with_noise_file = 'drone_motor.wav'  # Replace with your file
ambient_noise_file = 'ambient_noise.wav'  # Replace with your file

# Load the audio files
y_drone, sr = librosa.load(drone_with_noise_file, sr=None)
y_ambient, sr_ambient = librosa.load(ambient_noise_file, sr=None)

# Ensure both audio files have the same sampling rate and length
if sr != sr_ambient:
    raise ValueError("Sampling rates of the two audio files do not match!")

min_length = min(len(y_drone), len(y_ambient))
y_drone = y_drone[:min_length]
y_ambient = y_ambient[:min_length]

# Compute the Short-Time Fourier Transform (STFT) for both signals
stft_drone = librosa.stft(y_drone)
stft_ambient = librosa.stft(y_ambient)

# Compute the magnitude of the STFT (spectrogram)
S_drone = np.abs(stft_drone)
S_ambient = np.abs(stft_ambient)

# Subtract the ambient noise spectrogram from the drone + noise spectrogram
S_result = np.maximum(S_drone - S_ambient, 0)  # Ensure non-negative magnitudes

# Recreate the phase information from the original drone sound
# This ensures the audio sounds more natural after subtraction
phase_drone = np.angle(stft_drone)

# Combine the magnitude with the phase to reconstruct the complex STFT
stft_result = S_result * np.exp(1j * phase_drone)

# Perform the inverse STFT to convert back to time-domain audio
y_result = librosa.istft(stft_result)

# Save the resulting audio to a new file
output_file = 'drone_without_ambient.wav'
sf.write(output_file, y_result, sr)

# Plot the original and processed spectrograms for comparison
plt.figure(figsize=(10, 6))

# Original drone + noise spectrogram
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(S_drone, ref=np.max),
                         sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Drone with Ambient Noise')

# Ambient noise spectrogram
plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(S_ambient, ref=np.max),
                         sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Ambient Noise')

# Resulting spectrogram after subtraction
plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(S_result, ref=np.max),
                         sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Resulting Sound (Drone without Ambient Noise)')

plt.tight_layout()
plt.show()

print(f'Resulting sound saved to {output_file}')

