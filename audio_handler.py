import pyaudio
import wave
import numpy as np
import speech_recognition as sr
from google.cloud import speech

class AudioHandler:
    def __init__(self, device_index=None):
        self.audio = pyaudio.PyAudio()
        self.device_index = device_index if device_index is not None else self.find_external_microphone()
        self.recognizer = sr.Recognizer()
        
        # Standard audio parameters
        self.sample_rate = 16000
        self.channels = 1
        self.chunk = 1024
        self.format = pyaudio.paInt16

    def find_external_microphone(self):
        """Find external microphone if available"""
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                if 'USB' in device_info['name'] or 'External' in device_info['name']:
                    return i
        return None

    def record_audio(self, duration=5):
        """Record audio for specified duration"""
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk
        )

        print(f"Recording for {duration} seconds...")
        frames = []
        for _ in range(0, int(self.sample_rate / self.chunk * duration)):
            data = stream.read(self.chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        
        return b''.join(frames)

    def save_audio(self, audio_data, filename):
        """Save audio data to WAV file"""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data)

    def transcribe_audio(self, audio_data=None, language_code="bn-IN"):
        """Transcribe audio to text using Google Speech-to-Text"""
        if audio_data is None:
            audio_data = self.record_audio()

        # Save temporary WAV file
        temp_filename = "temp_audio.wav"
        self.save_audio(audio_data, temp_filename)

        # Use speech recognition
        with sr.AudioFile(temp_filename) as source:
            audio = self.recognizer.record(source)
            try:
                text = self.recognizer.recognize_google(audio, language=language_code)
                return text
            except sr.UnknownValueError:
                print("Could not understand audio")
                return ""
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return ""

    def cleanup(self):
        """Cleanup resources"""
        self.audio.terminate()