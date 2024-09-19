import logging
import sys
import time
import threading

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper
from audio_logger import audio_recording
import crazyflie_interface


def crazyflie_control():
    cflib.crtp.init_drivers()

    with SyncCrazyflie(crazyflie_interface.URI, cf=Crazyflie(rw_cache='./cache')) as scf:

        scf.cf.param.add_update_callback(group='deck', name='bcFlow2',
                                         cb=crazyflie_interface.param_deck_flow)
        time.sleep(1)

        logconf = LogConfig(name='Position', period_in_ms=10)
        logconf.add_variable('stateEstimate.x', 'float')
        logconf.add_variable('stateEstimate.y', 'float')
        scf.cf.log.add_config(logconf)
        logconf.data_received_cb.add_callback(crazyflie_interface.log_pos_callback)

    if not crazyflie_interface.deck_attached_event.wait(timeout=5):
        print('No flow deck detected!')
        sys.exit(1)

    logconf.start()
    crazyflie_interface.move_box_limit(scf)
    logconf.stop()


# Create threads for PyAudio and Crazyflie
audio_thread = threading.Thread(target=audio_recording())
crazyflie_thread = threading.Thread(target=crazyflie_control)

# Start both threads
audio_thread.start()
crazyflie_thread.start()

# Wait for both threads to complete (optional)
audio_thread.join()
crazyflie_thread.join()
