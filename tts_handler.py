import pyttsx3

class TTSHandler:
    def __init__(self):
        """Initialize TTS handler with pyttsx3."""
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)    # Speaking rate
        self.engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
        self.set_default_voice()

    def set_default_voice(self):
        """Set a suitable default voice."""
        voices = self.engine.getProperty('voices')
        if voices:
            self.engine.setProperty('voice', voices[0].id)

    def set_voice_properties(self, rate=None, volume=None, voice_id=None):
        """Configure voice properties."""
        if rate is not None:
            self.engine.setProperty('rate', rate)
        if volume is not None:
            self.engine.setProperty('volume', volume)
        if voice_id is not None:
            self.engine.setProperty('voice', voice_id)

    def text_to_speech(self, text):
        """Convert text to speech and play it."""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except Exception as e:
            print(f"Error in text-to-speech conversion: {e}")
            return False

    def save_to_file(self, text, output_file):
        """Save speech to an audio file."""
        try:
            self.engine.save_to_file(text, output_file)
            self.engine.runAndWait()
            return True
        except Exception as e:
            print(f"Error saving speech to file: {e}")
            return False

    def list_available_voices(self):
        """List all available voices."""
        return [{'id': voice.id, 'name': voice.name} for voice in self.engine.getProperty('voices')]

    def cleanup(self):
        """Cleanup resources."""
        self.engine.stop()
