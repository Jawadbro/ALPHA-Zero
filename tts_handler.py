# File: tts_handler.py
######################

import pyttsx3
import tempfile
import os

class TTSHandler:
    def __init__(self):
        """Initialize TTS handler with pyttsx3"""
        self.engine = pyttsx3.init()
        
        # Configure default properties
        self.engine.setProperty('rate', 150)    # Speaking rate
        self.engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
        
        # Get available voices
        voices = self.engine.getProperty('voices')
        # Try to find a suitable voice for Bangla/Bengali
        # Note: pyttsx3 might not have direct Bangla support, so we'll use the default voice
        self.engine.setProperty('voice', voices[1].id)

    def set_voice_properties(self, rate=None, volume=None, voice_id=None):
        """
        Configure voice properties
        
        Args:
            rate (int, optional): Speech rate (words per minute)
            volume (float, optional): Volume level (0.0 to 1.0)
            voice_id (str, optional): Voice identifier
        """
        if rate is not None:
            self.engine.setProperty('rate', rate)
        if volume is not None:
            self.engine.setProperty('volume', volume)
        if voice_id is not None:
            self.engine.setProperty('voice', voice_id)

    def text_to_speech(self, text):
        """
        Convert text to speech and play it
        
        Args:
            text (str): Text to convert to speech
            
        Returns:
            bool: Success status
        """
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except Exception as e:
            print(f"Error in text-to-speech conversion: {e}")
            return False

    def save_to_file(self, text, output_file):
        """
        Save speech to an audio file
        
        Args:
            text (str): Text to convert to speech
            output_file (str): Output file path (must end with .mp3)
            
        Returns:
            bool: Success status
        """
        try:
            self.engine.save_to_file(text, output_file)
            self.engine.runAndWait()
            return True
        except Exception as e:
            print(f"Error saving speech to file: {e}")
            return False

    def list_available_voices(self):
        """
        List all available voices
        
        Returns:
            list: List of available voice information
        """
        voices = []
        for voice in self.engine.getProperty('voices'):
            voices.append({
                'id': voice.id,
                'name': voice.name,
                'languages': voice.languages,
                'gender': voice.gender,
                'age': voice.age
            })
        return voices

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.engine.stop()
        except:
            pass