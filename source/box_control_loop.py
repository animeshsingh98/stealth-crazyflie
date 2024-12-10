import multiprocessing as mp
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper

import pyaudio
import numpy as np
import noisereduce as nr
from logging_config import setup_logging
import logging
from scipy.signal import butter, lfilter


logger = setup_logging(__name__)
logging.basicConfig(level=logging.ERROR)

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


def audio_recording(SL):
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

    while True:
        try:
            data = stream.read(CHUNK)
            audio_chunk = np.frombuffer(data, dtype=np.int16)

            # Apply noise reduction to the current chunk
            reduced_chunk = nr.reduce_noise(y=audio_chunk, sr=RATE, y_noise=noise_profile)
            reduced_chunk = bandpass_filter(reduced_chunk, lowcut=4000, highcut=16000, fs=RATE)
            rms = np.sqrt(np.mean(np.square(reduced_chunk)))
            sound_level = 20 * np.log10(rms)
            SL.value = sound_level
        except Exception as e:
            print("Error occured", e)


def crazyflie_control(SL):
    cflib.crtp.init_drivers()
    DEFAULT_HEIGHT = 0.5
    BOX_LIMIT = 1
    position_estimate = [0, 0]

    def log_pos_callback(timestamp, data, logconf):
        position_estimate[0] = round(data['stateEstimate.x'], 3)
        position_estimate[1] = round(data['stateEstimate.y'], 3)

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:

        logconf = LogConfig(name='Position', period_in_ms=10)
        logconf.add_variable('stateEstimate.x', 'float')
        logconf.add_variable('stateEstimate.y', 'float')
        scf.cf.log.add_config(logconf)
        logconf.data_received_cb.add_callback(log_pos_callback)

        logconf.start()
        print("Crazyflie initialized ...")
        with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
            body_x_cmd = 0.2
            body_y_cmd = 0.1
            max_vel = 0.5

            while True:
                sound_level = SL.value
                if sound_level > 20:
                    max_vel = max(0.1, max_vel - 0.1)
                    print("velocity decreased", sound_level, max_vel)
                elif sound_level < 18:
                    max_vel = min(0.5, max_vel + 0.1)
                    print("velocity increased", sound_level, max_vel)

                if position_estimate[0] > BOX_LIMIT:
                    body_x_cmd = -max_vel
                elif position_estimate[0] < -BOX_LIMIT:
                    body_x_cmd = max_vel
                if position_estimate[1] > BOX_LIMIT:
                    body_y_cmd = -max_vel
                elif position_estimate[1] < -BOX_LIMIT:
                    body_y_cmd = max_vel

                mc.start_linear_motion(body_x_cmd, 0, 0)

                time.sleep(0.1)
        # take_off_simple(scf)
        # logconf.stop()

if __name__ == '__main__':

    SL = mp.Value('d', 0.0)

    # Create the audio process
    audio_process = mp.Process(target=audio_recording, args=(SL,))

    # Create the Crazyflie control process
    crazyflie_process = mp.Process(target=crazyflie_control, args=(SL,))

    # Start both processes
    audio_process.start()
    crazyflie_process.start()

    # Join both processes to the main process
    audio_process.join()
    crazyflie_process.join()
