import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Load the audio file
audio_file = 'real_time_filtered_output.wav'  # Replace with your file
y, sr = librosa.load(audio_file, sr=None)

# Plot the waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform of the Audio")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# Perform Fast Fourier Transform (FFT) to analyze frequency components
n = len(y)
frequencies = np.fft.rfftfreq(n, 1/sr)
magnitude = np.abs(np.fft.rfft(y))

# Plot the frequency spectrum
plt.figure(figsize=(10, 4))
plt.plot(frequencies, magnitude)
plt.title("Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 5000)  # Adjust based on expected drone frequencies
plt.grid()
plt.show()

# Plot the Spectrogram to see how frequencies change over time
plt.figure(figsize=(10, 6))
D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                         sr=sr, hop_length=512, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()
