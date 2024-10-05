import os
import time
import speech_recognition as sr
import numpy as np
from tts_handler import TTSHandler
from audio_handler import AudioHandler
from image_handler import ImageHandler
from vision_handler import VisionHandler
from nlp_handler import NLPHandler
from database_handler import DatabaseHandler
from config import (FIREBASE_CREDENTIALS, GEMINI_API_KEY,
                   EXTERNAL_WEBCAM_INDEX, EXTERNAL_MIC_INDEX,
                   SYSTEM_PROMPT)

class SimplifiedWakeWordDetector:
    def __init__(self, wake_word="alpha"):
        self.recognizer = sr.Recognizer()
        self.wake_word = wake_word.lower()
    
    def detect_wake_word(self, audio_data):
        try:
            text = self.recognizer.recognize_google(audio_data).lower()
            print(f"Detected speech: {text}")
            return self.wake_word in text
        except sr.UnknownValueError:
            return False
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return False
    
    def cleanup(self):
        pass

def print_separator():
    print("\n" + "="*50 + "\n")

def main():
    print_separator()
    print("Initializing ALPHA Zero System...")
    print_separator()
    
    try:
        # Initialize all handlers
        print("Initializing handlers...")
        tts_handler = TTSHandler()
        print("✓ TTS Handler initialized")
        
        audio_handler = AudioHandler(device_index=EXTERNAL_MIC_INDEX)
        print("✓ Audio Handler initialized")
        
        image_handler = ImageHandler(device_index=EXTERNAL_WEBCAM_INDEX)
        print("✓ Image Handler initialized")
        
        vision_handler = VisionHandler()
        print("✓ Vision Handler initialized")
        
        nlp_handler = NLPHandler(api_key=GEMINI_API_KEY, system_prompt=SYSTEM_PROMPT)
        print("✓ NLP Handler initialized")
        
        db_handler = DatabaseHandler(credentials_path=FIREBASE_CREDENTIALS)
        print("✓ Database Handler initialized")
        
        wake_detector = SimplifiedWakeWordDetector()
        print("✓ Wake Word Detector initialized")
        
        print_separator()
        print("System initialization complete!")
        print("Listening for wake word 'alpha'...")
        print_separator()

        while True:
            # Listen for wake word
            audio_data = audio_handler.listen_for_wake_word()
            
            if audio_data:
                # Check for wake word
                if wake_detector.detect_wake_word(audio_data):
                    print_separator()
                    print("Wake word detected! Starting processing pipeline...")
                    
                    # 1. Image Capture
                    print("\n1. Capturing image...")
                    image = image_handler.capture_image(return_numpy=True)
                    
                    if image is not None:
                        try:
                            print(f"✓ Image captured successfully. Shape: {image.shape}")
                            
                            # 2. Text Detection
                            print("\n2. Detecting text from image...")
                            detected_text = vision_handler.detect_text(image)
                            
                            if detected_text:
                                print(f"✓ Detected text: {detected_text}")
                                
                                # 3. Text Processing
                                print("\n3. Processing detected text...")
                                translated_text, summary_text = nlp_handler.translate_and_summarize(detected_text)
                                print(f"✓ Translated text: {translated_text}")
                                print(f"✓ Summary: {summary_text}")
                                
                                # 4. Database Storage
                                print("\n4. Saving to database...")
                                db_handler.save_interaction(detected_text, summary_text)
                                print("✓ Saved to database")
                                
                                # 5. Text-to-Speech
                                print("\n5. Converting to speech...")
                                tts_handler.text_to_speech(summary_text)
                                print("✓ Speech conversion complete")
                            else:
                                print("⚠ No text detected in the image")
                        except Exception as e:
                            print(f"\n❌ Error during processing: {e}")
                            import traceback
                            print(traceback.format_exc())
                    else:
                        print("\n❌ Failed to capture image")
                    
                    print_separator()
                    print("Resuming wake word detection...")
                    print_separator()
            
            time.sleep(0.1)  # Small delay to prevent high CPU usage

    except KeyboardInterrupt:
        print_separator()
        print("Shutdown initiated by user...")
    except Exception as e:
        print_separator()
        print(f"❌ Unexpected error: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        print("\nCleaning up resources...")
        try:
            tts_handler.cleanup()
            print("✓ TTS Handler cleaned up")
            audio_handler.cleanup()
            print("✓ Audio Handler cleaned up")
            image_handler.cleanup()
            print("✓ Image Handler cleaned up")
            vision_handler.cleanup()
            print("✓ Vision Handler cleaned up")
            db_handler.cleanup()
            print("✓ Database Handler cleaned up")
            wake_detector.cleanup()
            print("✓ Wake Word Detector cleaned up")
            print_separator()
            print("System shutdown complete!")
            print_separator()
        except Exception as e:
            print(f"\n❌ Error during cleanup: {e}")

if __name__ == "__main__":
    main()