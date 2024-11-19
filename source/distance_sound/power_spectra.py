import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft

def plot_power_spectra(file_paths, start_time=3, end_time=4):
    plt.figure(figsize=(10, 6))

    for file_path in file_paths:
        # Load the WAV file
        sample_rate, data = wavfile.read(file_path)
        print(file_path)

        # Select the channel if it's stereo
        if data.ndim > 1:
            data = data[:, 0]  # Take the first channel

        # Extract the segment from start_time to end_time
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        segment = data[start_sample:end_sample]

        # Calculate the Fourier Transform and the corresponding power spectrum
        N = len(segment)
        freqs = np.fft.fftfreq(N, 1 / sample_rate)
        fft_values = fft(segment)
        power_spectrum = np.abs(fft_values) ** 2

        # Plot the positive frequencies
        positive_freqs = freqs[:N // 2]
        positive_power = power_spectrum[:N // 2]

        plt.plot(positive_freqs, positive_power, label=file_path)

    # Customize the plot
    plt.title("Power Spectrum (3-4 seconds)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.legend()
    plt.xlim(4000, 16000)  # Adjust frequency range as needed
    plt.grid()
    plt.show()

# Usage
file_paths = ['free_field_cropped.wav', 'wall_cropped.wav', 'wall_close_cropped.wav']
plot_power_spectra(file_paths)
