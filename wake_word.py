import pvporcupine
import struct

class WakeWordDetector:
    def __init__(self, access_key, keyword_path):
        self.porcupine = pvporcupine.create(access_key=access_key, keyword_paths=[keyword_path])
        self.frame_length = self.porcupine.frame_length

    def detect_wake_word(self, pcm):
        """Detect wake word in PCM audio data."""
        try:
            # Check if the input PCM data matches the expected frame length
            if len(pcm) != self.frame_length * 2:  # *2 because each sample is 2 bytes
                print(f"Warning: Expected {self.frame_length * 2} bytes, but got {len(pcm)} bytes")
                # Adjust PCM data to match expected length
                if len(pcm) > self.frame_length * 2:
                    pcm = pcm[:self.frame_length * 2]
                else:
                    pcm = pcm.ljust(self.frame_length * 2, b'\x00')

            # Convert bytes to int16 array
            pcm_array = struct.unpack_from(f"{self.frame_length}h", pcm)

            # Process the audio frame
            result = self.porcupine.process(pcm_array)
            return result >= 0
        except Exception as e:
            print(f"Error in wake word detection: {e}")
            return False

    def cleanup(self):
        """Cleanup resources."""
        self.porcupine.delete()