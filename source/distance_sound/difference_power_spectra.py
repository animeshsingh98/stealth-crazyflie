import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq

def compute_power_spectrum(file_path, start_time=5, end_time=6):
    # Load the WAV file
    sample_rate, data = wavfile.read(file_path)

    # If stereo, take only the first channel
    if data.ndim > 1:
        data = data[:, 0]

    # Extract the segment from start_time to end_time
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    segment = data[start_sample:end_sample]

    # Perform FFT and compute the power spectrum
    N = len(segment)
    fft_values = fft(segment)
    power_spectrum = np.abs(fft_values[:N // 2]) ** 2  # Take positive frequencies only

    # Corresponding frequencies
    freqs = np.fft.fftfreq(N, 1 / sample_rate)[:N // 2]
    return freqs, power_spectrum

def analyze_spectrum_difference(file1, file2, start_time=3, end_time=4):
    # Compute power spectra for both files
    freqs1, power_spectrum1 = compute_power_spectrum(file1, start_time, end_time)
    freqs2, power_spectrum2 = compute_power_spectrum(file2, start_time, end_time)

    # Calculate the difference in power spectra
    power_diff = power_spectrum1 - power_spectrum2

    # Perform FFT on the power difference spectrum
    N = len(power_diff)
    fft_diff = fft(power_diff)
    freqs_diff = fftfreq(N, freqs1[1] - freqs1[0])[:N // 2]
    fft_diff_magnitude = np.abs(fft_diff[:N // 2])
    print(max(fft_diff_magnitude))

    # Plot the power spectrum difference
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(freqs1, power_diff, label='Difference in Power Spectrum', color='purple')
    plt.title("Difference in Power Spectrum between two files")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Difference")
    plt.legend()
    plt.grid()
    plt.xlim(4000, 16000)  # Adjust the frequency range as needed

    # Plot the FFT of the power spectrum difference
    plt.subplot(2, 1, 2)
    plt.plot(freqs_diff, fft_diff_magnitude, label='FFT of Power Spectrum Difference', color='blue')
    plt.title("FFT of the Difference in Power Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid()
    plt.xlim(0, 10)  # Adjust the frequency range as needed

    plt.tight_layout()
    plt.show()

# Usage
file1 = 'wall_cropped.wav'
file2 = 'wall_close_cropped.wav'
analyze_spectrum_difference(file1, file2)
