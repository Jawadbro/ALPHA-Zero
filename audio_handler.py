import pyaudio
import wave
import numpy as np
import speech_recognition as sr
import os


class AudioHandler:
    def __init__(self, device_index=None):
        self.audio = pyaudio.PyAudio()
        self.device_index = device_index if device_index is not None else self.find_external_microphone()
        self.recognizer = sr.Recognizer()

        # Updated audio parameters for better speech recognition
        self.sample_rate = 44100  # Standard sample rate for better quality
        self.channels = 1
        self.chunk = 1024  # Increased chunk size for better processing
        self.format = pyaudio.paInt16

        # Initialize microphone using speech_recognition
        try:
            self.microphone = sr.Microphone(
                device_index=self.device_index,
                sample_rate=self.sample_rate
            )
            # Adjust for ambient noise
            with self.microphone as source:
                print("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
        except Exception as e:
            print(f"Error initializing microphone: {e}")
            self.microphone = None

        # Print device info for debugging
        self.print_device_info()

    def find_external_microphone(self):
        """Find external microphone if available."""
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                if 'USB' in device_info['name'] or 'External' in device_info['name']:
                    return i
        return None

    def print_device_info(self):
        """Print audio device info."""
        print("\nAvailable audio devices:")
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            print(f"Device {i}: {device_info['name']}")
            print(f"  Max Input Channels: {device_info['maxInputChannels']}")
            print(f"  Default Sample Rate: {device_info['defaultSampleRate']}")

    def listen_for_wake_word(self, timeout=1, phrase_time_limit=3):
        """Listen for audio and return the audio data."""
        if not self.microphone:
            print("Error: Microphone not initialized")
            return None

        try:
            with self.microphone as source:
                print("Listening...")
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
                return audio
        except sr.WaitTimeoutError:
            return None
        except Exception as e:
            print(f"Error while listening: {e}")
            return None

    def record_audio(self, duration=5):
        """Record audio for a specified duration."""
        print(f"Recording for {duration} seconds...")
        
        frames = []
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk
        )

        try:
            for _ in range(0, int(self.sample_rate / self.chunk * duration)):
                data = stream.read(self.chunk, exception_on_overflow=False)
                frames.append(data)

            audio_data = b''.join(frames)
            print(f"Recording completed: {len(audio_data)} bytes")
            return audio_data

        finally:
            stream.stop_stream()
            stream.close()

    def save_audio(self, audio_data, filename="recorded_audio.wav"):
        """Save audio data to WAV file."""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data)
        print(f"Audio saved to {filename}")

    def transcribe_audio(self, audio_data=None, language_code="bn-IN"):
        """Transcribe audio to text using Google Speech-to-Text."""
        try:
            if isinstance(audio_data, sr.AudioData):
                # If audio_data is already an AudioData object
                return self.recognizer.recognize_google(audio_data, language=language_code)
            
            if audio_data is None:
                audio_data = self.record_audio()

            # Save temporary WAV file
            temp_filename = "temp_audio.wav"
            self.save_audio(audio_data, temp_filename)

            # Use speech recognition
            with sr.AudioFile(temp_filename) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio, language=language_code)
                return text

        except sr.UnknownValueError:
            print("Could not understand audio")
            return ""
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return ""
        finally:
            # Clean up temporary file
            if os.path.exists("temp_audio.wav"):
                os.remove("temp_audio.wav")

    def cleanup(self):
        """Cleanup resources."""
        self.audio.terminate()