import pyaudio
import wave

# Set the parameters for audio recording
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1              # Number of audio channels (1 for mono)
RATE = 44100              # Sample rate (samples per second)
CHUNK = 1024              # Number of frames per buffer
RECORD_SECONDS = 5        # Length of recording in seconds
WAVE_OUTPUT_FILENAME = "output.wav"  # Output file name

audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Recording...")

frames = []

for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)  # Read audio data from the input stream
    frames.append(data)  # Add the data to the frames list

print("Fniished recording.")

stream.stop_stream()
stream.close()
audio.terminate()

with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)  # Set number of channels
    wf.setsampwidth(audio.get_sample_size(FORMAT))  # Set sample width
    wf.setframerate(RATE)  # Set frame rate
    wf.writeframes(b''.join(frames))  # Write the audio frames

print(f"Saved to {WAVE_OUTPUT_FILENAME}")