import pyaudio
import wave
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import numpy as np

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "output.wav"


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

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

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
