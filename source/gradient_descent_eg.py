import ctypes
import numpy as np
import multiprocessing as mp
import time

from logging_config import setup_logging
import logging


logger = setup_logging(__name__)
logging.basicConfig(level=logging.ERROR)
lock = mp.Lock()

# Define matrix size and center point
matrix_size = 100
center = (matrix_size // 2, matrix_size // 2)

# Initialize matrix with distances from the center
matrix = np.zeros((matrix_size, matrix_size))

for i in range(matrix_size):
    for j in range(matrix_size):
        # Calculate Euclidean distance from each cell to the center
        distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
        matrix[i, j] = distance

# Normalize the matrix to make the center exactly 0 and surrounding values positive
matrix -= matrix[center]

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata

class DynamicPathPlot:
    def __init__(self, position):
        # Initialize the plot with a grid
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_title("Dynamic Path Plot")
        self.positions = [(position[0], position[1])]
        self.path, = self.ax.plot([], [], 'bo-', markersize=5, linewidth=2)  # 'bo-' is blue line with markers
        self.position = position  # Shared position

    def update_path(self, frame):
        # Update the plot with the current position
        x, y = self.position[0], self.position[1]
        self.positions.append((x, y))
        self.path.set_data(*zip(*self.positions))

    def animate(self):
        ani = FuncAnimation(self.fig, self.update_path, interval=100)
        plt.show()

class DynamicPathPlot3D:
    def __init__(self, position, SL):
        # Initialize the plot with a 3D grid
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.set_zlim(0, 50)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Value (Sound Level / Altitude)")
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
        ani = FuncAnimation(self.fig, self.update_path, interval=100)
        plt.show()

def plot_function(SL, position):
    plotter = DynamicPathPlot3D(position, SL)
    plotter.animate(SL)

def crazyflie_control(SL, position):
    min_val = 10000
    min_pos = []
    notdone = True
    while notdone:
        curr_pos = np.array(position)
        curr_sl = matrix[int(curr_pos[0]*10)][int(curr_pos[0]*10)]
        SL.value = curr_sl
        for i in [0,1,0,-1]:
            for j in [1,0,-1,0]:
                pos = curr_pos + np.array([i*0.2, j*0.2])
                k = matrix[int(pos[0]*10)][int(pos[0]*10)]
                if min_val > k:
                    min_val = k
                    min_pos = pos
                time.sleep(0.025)
        print(min_val, curr_sl)
        if min_val < curr_sl:
            position[0] = min_pos[0]
            position[1] = min_pos[1]
        else:
            notdone = False


if __name__ == '__main__':

    SL = mp.Value('d', 5)
    position = mp.Array(ctypes.c_float, 2)
    position[0] = 0.1
    position[1] = 0.1

    # Create the audio process

    # Create the Crazyflie control process
    crazyflie_process = mp.Process(target=crazyflie_control, args=(SL,position))
    plot_process = mp.Process(target=plot_function, args=(SL,position))

    # Start both processes
    crazyflie_process.start()
    plot_process.start()

    # Join both processes to the main process
    crazyflie_process.join()
    plot_process.join()
