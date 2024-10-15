import pyaudio
import numpy as np
import noisereduce as nr
import soundfile as sf
from logging_config import setup_logging
import logging
from scipy.signal import butter, lfilter

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 20
WAVE_OUTPUT_FILENAME = "output.wav"
NOISE_PROFILE_DURATION = 2
logger = setup_logging(__name__)
logging.basicConfig(level=logging.ERROR)


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


def audio_recording(filename=WAVE_OUTPUT_FILENAME):
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording and filtering in real-time...")

    # Collect noise profile for the first second
    print("Capturing ambient noise profile...")
    noise_frames = []

    for _ in range(int(RATE / CHUNK * NOISE_PROFILE_DURATION)):
        data = stream.read(CHUNK)
        noise_frames.append(np.frombuffer(data, dtype=np.int16))

    # Combine the noise frames to create the noise profile
    noise_profile = np.hstack(noise_frames)

    # Real-time recording and filtering
    filtered_audio = []

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        audio_chunk = np.frombuffer(data, dtype=np.int16)

        # Apply noise reduction to the current chunk
        reduced_chunk = nr.reduce_noise(y=audio_chunk, sr=RATE, y_noise=noise_profile)
        reduced_chunk = bandpass_filter(reduced_chunk, lowcut=4000, highcut=16000, fs=RATE)
        rms = np.sqrt(np.mean(np.square(reduced_chunk)))
        logger.info(f"sound level {20 * np.log10(rms)}")

        # Store the filtered audio
        filtered_audio.extend(reduced_chunk)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Convert the filtered audio into a numpy array
    filtered_audio = np.array(filtered_audio, dtype=np.int16)

    # Save the filtered audio to a .wav file
    sf.write(WAVE_OUTPUT_FILENAME, filtered_audio, RATE)
