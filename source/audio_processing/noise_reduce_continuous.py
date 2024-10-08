import pyaudio
import numpy as np
import noisereduce as nr
import soundfile as sf

# Audio stream configuration
CHUNK = 1024  # Number of samples per chunk
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1  # Mono channel
RATE = 44100  # Sample rate (44.1 kHz)
NOISE_PROFILE_DURATION = 1  # Duration to capture noise profile in seconds

# Initialize PyAudio
p = pyaudio.PyAudio()

# Initialize recording stream
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
try:
    while True:
        # Read audio chunk
        data = stream.read(CHUNK)
        audio_chunk = np.frombuffer(data, dtype=np.int16)

        # Apply noise reduction to the current chunk
        reduced_chunk = nr.reduce_noise(y=audio_chunk, sr=RATE, y_noise=noise_profile)

        # Store the filtered audio
        filtered_audio.extend(reduced_chunk)

        # Optionally: Play back the filtered audio in real-time (not implemented here)

except KeyboardInterrupt:
    print("Recording stopped.")

# Stop and close the stream
stream.stop_stream()
stream.close()
p.terminate()

# Convert the filtered audio into a numpy array
filtered_audio = np.array(filtered_audio, dtype=np.int16)

# Save the filtered audio to a .wav file
output_file = 'real_time_filtered_output.wav'
sf.write(output_file, filtered_audio, RATE)

print(f"Filtered audio saved to {output_file}")
