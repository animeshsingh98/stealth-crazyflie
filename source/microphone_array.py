import time

import usb.core
import usb.util
import pyaudio
import wave
from tuning import Tuning
import multiprocessing as mp

RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 1 # change base on firmwares, 1_channel_firmware.bin as 1 or 6_channels_firmware.bin as 6
RESPEAKER_WIDTH = 2
CHUNK = 1024
CHUNKSIZE = 15


def ang_shift(angle):
    shifted_angle = angle + 360
    return shifted_angle


def find_device(index):
    devices = list(usb.core.find(find_all=True, idVendor=0x2886, idProduct=0x0018))
    if not devices:
        raise Exception("No USB devices found.")
    if index >= len(devices):
        raise Exception(f"Device index {index} out of range. Only {len(devices)} devices found.")
    return devices[index]


def open_audio_stream(p, RESPEAKER_INDEX):
    stream = p.open(
        rate=RESPEAKER_RATE,
        format=p.get_format_from_width(RESPEAKER_WIDTH),
        channels=RESPEAKER_CHANNELS,
        input=True,
        input_device_index=RESPEAKER_INDEX,
    )
    return stream


def record_audio(stream, p, dev, audio_file, doa_file, doa_val):

    wf = wave.open(audio_file, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
    wf.setframerate(RESPEAKER_RATE)
    for i in range(0, int(RESPEAKER_RATE / CHUNK * CHUNKSIZE)):
        data = stream.read(CHUNK, exception_on_overflow = False)
        wf.writeframes(data)

        Mic_tuning = Tuning(dev)
        doa = Mic_tuning.direction
        print(doa)
        doa_val.value = doa
        time.sleep(1)


def close_audio_stream(stream, p):
    stream.stop_stream()
    stream.close()
    p.terminate()


def speaker_array(index, data):
    RESPEAKER_INDEX = index
    while True:
        dev = find_device(RESPEAKER_INDEX - 2)
        p = pyaudio.PyAudio()
        stream = open_audio_stream(p, RESPEAKER_INDEX)
        audio_file = f'chunk_{RESPEAKER_INDEX}.wav'
        doa_file   = f'DOA_{RESPEAKER_INDEX}.json'

        record_audio(stream, p, dev, audio_file, doa_file, data)

        close_audio_stream(stream, p)


if __name__ == '__main__':
    doaR = mp.Value('d', 5)
    speaker_array(1, doaR)

