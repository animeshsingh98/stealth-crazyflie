import multiprocessing as mp
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.high_level_commander import HighLevelCommander
from cflib.utils import uri_helper

import pyaudio
import numpy as np
import noisereduce as nr
from scipy.signal import butter, lfilter
import soundfile as sf


URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')


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


def audio_recording():
    # Combine the noise frames to create the noise profile
    p = pyaudio.PyAudio()
    CHUNK = 2048
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    NOISE_PROFILE_DURATION = 2

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("Capturing ambient noise profile...")
    noise_frames = []

    for _ in range(int(RATE / CHUNK * NOISE_PROFILE_DURATION)):
        data = stream.read(CHUNK)
        noise_frames.append(np.frombuffer(data, dtype=np.int16))
    noise_profile = np.hstack(noise_frames)

    print("Sound level tracking initialized ...")
    filtered_audio = []

    try:
        while True:
            data = stream.read(CHUNK)
            audio_chunk = np.frombuffer(data, dtype=np.int16)

            # Apply noise reduction to the current chunk
            reduced_chunk = nr.reduce_noise(y=audio_chunk, sr=RATE, y_noise=noise_profile)
            reduced_chunk = bandpass_filter(reduced_chunk, lowcut=4000, highcut=16000, fs=RATE)
            filtered_audio.extend(reduced_chunk)
    except KeyboardInterrupt:
        print("Recording stopped.")

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Convert the filtered audio into a numpy array
        filtered_audio = np.array(filtered_audio, dtype=np.int16)

        # Save the filtered audio to a .wav file
        output_file = 'free_50cm.wav'
        sf.write(output_file, filtered_audio, RATE)

        print(f"Filtered audio saved to {output_file}")


def crazyflie_control():
    cflib.crtp.init_drivers()
    DEFAULT_HEIGHT = 0.5

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        print("Crazyflie initialized ...")
        commander = HighLevelCommander(scf.cf)
        commander.takeoff(DEFAULT_HEIGHT, 3)
        time.sleep(10)
        commander.land(0.0, 2.0)  # Land with a descent rate


if __name__ == '__main__':


    # Create the audio process
    audio_process = mp.Process(target=audio_recording, args=())

    # Create the Crazyflie control process
    crazyflie_process = mp.Process(target=crazyflie_control, args=())

    # Start both processes
    audio_process.start()
    crazyflie_process.start()

    # Join both processes to the main process
    audio_process.join()
    crazyflie_process.join()
