import pyaudio
import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Initialize PyAudio
p = pyaudio.PyAudio()

# Audio stream settings
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1              # Mono audio
RATE = 44100              # Sample rate (Hz)
CHUNK = 1024              # Samples per frame (size of buffer)

RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "output.wav"


# Butterworth filter for noise reduction (optional)
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Function to compute the RMS (Root Mean Square) value
def calculate_rms(block):
    return np.sqrt(np.mean(np.square(block)))


# Convert RMS to decibels (dB)
def rms_to_db(rms):
    return 20 * np.log10(rms + 1e-6)  # Add small offset to avoid log(0)


def audio_recording(filename=WAVE_OUTPUT_FILENAME):
    # Live plot settings
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'r-')
    ax.set_ylim(-120, 0)  # dB range for display
    ax.set_xlim(0, 10)    # Display the last 10 samples
    ax.set_title("Real-time Noise Level in dB")
    ax.set_ylabel("Noise Level (dB)")
    ax.set_xlabel("Time (s)")

    db_values = []  # Store dB values for plotting
    x_data = np.linspace(0, 10, num=10)

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        # data = stream.read(CHUNK)

        audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        frames.append(audio_data)

        # Apply bandpass filter (optional for noise reduction)
        # filtered_data = bandpass_filter(audio_data, lowcut=15, highcut=1000, fs=RATE)

        # Calculate RMS and dB
        rms_value = calculate_rms(audio_data)
        db_value = rms_to_db(rms_value)
        print(f"Noise level: {db_value:.2f} dB")

        # Append dB value for plotting
        db_values.append(db_value)

        # Keep the last 10 dB values in the plot
        if len(db_values) > 10:
            db_values.pop(0)

        # Update plot data
        line.set_xdata(x_data[:len(db_values)])
        line.set_ydata(db_values)
        ax.draw_artist(line)
        fig.canvas.draw()
        fig.canvas.flush_events()

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


if __name__ == '__main__':
    audio_recording()
