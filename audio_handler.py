import pyaudio
import wave
import numpy as np
import speech_recognition as sr
from typing import Optional, Union
import logging

class AudioHandler:
    def __init__(self, device_index: Optional[int] = None, 
                 sample_rate: int = 44100, 
                 channels: int = 1, 
                 chunk_size: int = 1024):
        """
        Initialize the AudioHandler.
        
        Args:
            device_index: Specific microphone device index. If None, will try to find external mic
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1 for mono, 2 for stereo)
            chunk_size: Size of audio chunks to process at a time
        """
        # Set up logging first
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.device_index = device_index if device_index is not None else self.find_external_microphone()
        
        # Audio parameters
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk_size
        self.format = pyaudio.paInt16
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        
        # Initialize microphone
        try:
            self.microphone = sr.Microphone(
                device_index=self.device_index,
                sample_rate=self.sample_rate
            )
            # Adjust for ambient noise
            with self.microphone as source:
                self.logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                self.logger.info("Ambient noise adjustment complete")
        except Exception as e:
            self.logger.error(f"Error initializing microphone: {e}")
            self.microphone = None

    def find_external_microphone(self) -> Optional[int]:
        """
        Find external microphone if available.
        
        Returns:
            Optional[int]: Device index of external microphone if found, None otherwise
        """
        try:
            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    if 'USB' in device_info['name'] or 'External' in device_info['name']:
                        self.logger.info(f"Found external microphone: {device_info['name']}")
                        return i
            
            # If no external mic found, use default
            self.logger.warning("No external microphone found, using default device")
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding external microphone: {e}")
            return None

    def listen_for_wake_word(self, timeout: int = 5) -> Optional[sr.AudioData]:
        """
        Listen for audio and return the audio data.
        
        Args:
            timeout: Maximum number of seconds to wait for audio
        
        Returns:
            Optional[sr.AudioData]: The captured audio data if successful, None otherwise
        """
        if not self.microphone:
            self.logger.error("Error: Microphone not initialized")
            return None

        try:
            with self.microphone as source:
                self.logger.info("Listening for wake word...")
                audio = self.recognizer.listen(source, timeout=timeout)
                return audio
        except sr.WaitTimeoutError:
            self.logger.info("No audio detected within the timeout period")
            return None
        except Exception as e:
            self.logger.error(f"Error while listening for audio: {e}")
            return None

    def listen_for_speech(self, timeout: int = 5, 
                         phrase_time_limit: Optional[int] = None) -> Optional[sr.AudioData]:
        """
        Listen for speech input and return the audio data.
        
        Args:
            timeout: Maximum number of seconds to wait for speech
            phrase_time_limit: Maximum number of seconds for a phrase
        
        Returns:
            Optional[sr.AudioData]: The captured audio data if successful, None otherwise
        """
        if not self.microphone:
            self.logger.error("Error: Microphone not initialized")
            return None

        try:
            with self.microphone as source:
                self.logger.info("Listening for speech...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                return audio
        except sr.WaitTimeoutError:
            self.logger.info("No speech detected within the timeout period")
            return None
        except Exception as e:
            self.logger.error(f"Error while listening for speech: {e}")
            return None

    def transcribe_audio(self, audio_data: sr.AudioData, 
                        language_code: str = "en-US") -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio_data: The audio data to transcribe
            language_code: The language code for transcription
        
        Returns:
            str: The transcribed text, or empty string if transcription fails
        """
        try:
            text = self.recognizer.recognize_google(audio_data, language=language_code)
            self.logger.info(f"Transcribed text: {text}")
            return text
        except sr.UnknownValueError:
            self.logger.warning("Speech recognition could not understand the audio")
            return ""
        except sr.RequestError as e:
            self.logger.error(f"Could not request results from speech recognition service: {e}")
            return ""
        except Exception as e:
            self.logger.error(f"Unexpected error during transcription: {e}")
            return ""

    def record_audio(self, duration: int = 5, 
                    output_filename: str = "recorded_audio.wav") -> bool:
        """
        Record audio for a specified duration and save to file.
        
        Args:
            duration: Recording duration in seconds
            output_filename: Output wav file name
        
        Returns:
            bool: True if recording successful, False otherwise
        """
        try:
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk
            )

            self.logger.info(f"Recording {duration} seconds of audio...")
            frames = []
            
            for _ in range(0, int(self.sample_rate / self.chunk * duration)):
                data = stream.read(self.chunk)
                frames.append(data)

            self.logger.info("Recording complete")
            
            stream.stop_stream()
            stream.close()

            # Save the recorded audio to wav file
            with wave.open(output_filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(frames))
            
            self.logger.info(f"Audio saved to {output_filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during audio recording: {e}")
            return False

    def play_audio(self, filename: str) -> bool:
        """
        Play audio from a wav file.
        
        Args:
            filename: Path to the wav file to play
        
        Returns:
            bool: True if playback successful, False otherwise
        """
        try:
            # Open the wave file
            with wave.open(filename, 'rb') as wf:
                # Open a stream
                stream = self.audio.open(
                    format=self.audio.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True
                )

                # Read data in chunks and play
                data = wf.readframes(self.chunk)
                while data:
                    stream.write(data)
                    data = wf.readframes(self.chunk)

                # Cleanup
                stream.stop_stream()
                stream.close()
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error playing audio file: {e}")
            return False

    def get_audio_devices(self) -> list:
        """
        Get list of available audio devices.
        
        Returns:
            list: List of dictionaries containing device information
        """
        devices = []
        try:
            for i in range(self.audio.get_device_count()):
                devices.append(self.audio.get_device_info_by_index(i))
            return devices
        except Exception as e:
            self.logger.error(f"Error getting audio devices: {e}")
            return []

    def cleanup(self):
        """Clean up resources."""
        try:
            self.audio.terminate()
            self.logger.info("Audio resources cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")