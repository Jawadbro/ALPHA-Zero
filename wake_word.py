import pvporcupine

class WakeWordDetector:
    def __init__(self, access_key, keyword_path):
        self.porcupine = pvporcupine.create(access_key=access_key, keyword_paths=[keyword_path])

    def detect_wake_word(self, pcm):
        """Detect wake word in PCM audio data."""
        return self.porcupine.process(pcm) >= 0

    def cleanup(self):
        """Cleanup resources."""
        self.porcupine.delete()
