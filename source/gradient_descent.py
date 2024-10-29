import ctypes
import multiprocessing as mp
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.crazyflie.high_level_commander import HighLevelCommander
from cflib.utils import uri_helper

import pyaudio
import numpy as np
import noisereduce as nr
from logging_config import setup_logging
import logging
from scipy.signal import butter, lfilter

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

logger = setup_logging(__name__)
logging.basicConfig(level=logging.ERROR)
lock = mp.Lock()

class DynamicPathPlot3D:
    def __init__(self, position, SL):
        # Initialize the plot with a 3D grid
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(0, 2)
        self.ax.set_ylim(0, 2)
        self.ax.set_zlim(0, 30)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Value (Sound Level)")
        self.ax.set_title("Dynamic Path Plot (3D)")

        # Initialize the path with starting position
        self.positions = [(position[0], position[1], SL.value)]
        self.path, = self.ax.plot([], [], [], 'bo-', markersize=5, linewidth=2)  # 'bo-' is blue line with markers
        self.position = position  # Shared position
        self.SL = SL

    def update_path(self, frame):
        # Update the plot with the current position
        x, y, val = self.position[0], self.position[1], self.SL.value
        self.positions.append((x, y, val))

        # Update the path data
        xs, ys, zs = zip(*self.positions)
        self.path.set_data(xs, ys)
        self.path.set_3d_properties(zs)

    def animate(self, SL):
        self.SL = SL
        ani = FuncAnimation(self.fig, self.update_path, interval=1000)
        plt.show()

def plot_function(SL, position):
    plotter = DynamicPathPlot3D(position, SL)
    plotter.animate(SL)
    time.sleep(1)

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


def crazyflie_control(SL, position):
    cflib.crtp.init_drivers()
    DEFAULT_HEIGHT = 0.5
    BOX_LIMIT = 1
    position_estimate = [0, 0]

    def log_pos_callback(timestamp, data, logconf):
        position_estimate[0] = round(data['stateEstimate.x'], 3)
        position_estimate[1] = round(data['stateEstimate.y'], 3)
        position[0] = position_estimate[0]
        position[1] = position_estimate[1]

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:

        logconf = LogConfig(name='Position', period_in_ms=10)
        logconf.add_variable('stateEstimate.x', 'float')
        logconf.add_variable('stateEstimate.y', 'float')
        scf.cf.log.add_config(logconf)
        logconf.data_received_cb.add_callback(log_pos_callback)

        logconf.start()
        print("Crazyflie initialized ...")
        commander = HighLevelCommander(scf.cf)
        commander.takeoff(DEFAULT_HEIGHT, 3.0)
        time.sleep(3)
        try:
            min_val = 10000
            min_pos = []
            notdone = True
            while notdone:
                curr_pos = np.array(position)
                curr_sl = SL.value
                for i in [0,1,0,-1]:
                    for j in [1,0,-1,0]:
                        pos = curr_pos + np.array([i*0.2, j*0.2])
                        commander.go_to(pos[0], pos[1], DEFAULT_HEIGHT, 0, 0.25)
                        time.sleep(1)
                        k = SL.value
                        if min_val > k:
                            min_val = k
                            min_pos = pos
                        time.sleep(0.25)
                print(min_val, curr_sl)
                # if min_val < curr_sl:
                #     position[0] = min_pos[0]
                #     position[1] = min_pos[1]
                position[0] = curr_pos[0] + 0.2
                position[1] = curr_pos[1] + 0.2
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Initiating safe landing...")

        finally:
            # Ensure the Crazyflie lands safely
            commander.land(0.0, 2.0)  # Land with a descent rate
            time.sleep(3)  # Wait for landing to complete
            commander.stop()  # Stop the commander to clean up
            print("Landing completed.")


if __name__ == '__main__':

    SL = mp.Value('d', 5)
    position = mp.Array(ctypes.c_float, 2)
    position[0] = 0.1
    position[1] = 0.1

    # Create the audio process

    # Create the Crazyflie control process
    crazyflie_process = mp.Process(target=crazyflie_control, args=(SL,position))
    audio_process = mp.Process(target=audio_recording, args=(SL,))
    plot_process = mp.Process(target=plot_function, args=(SL,position))

    # Start both processes
    crazyflie_process.start()
    audio_process.start()
    plot_process.start()

    # Join both processes to the main process
    crazyflie_process.join()
    audio_process.join()
    plot_process.join()
