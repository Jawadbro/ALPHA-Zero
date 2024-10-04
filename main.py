import os
import time
from tts_handler import TTSHandler
from audio_handler import AudioHandler
from image_handler import ImageHandler
from vision_handler import VisionHandler
from nlp_handler import NLPHandler
from wake_word import WakeWordDetector
from database_handler import DatabaseHandler
from config import (FIREBASE_CREDENTIALS, GEMINI_API_KEY, 
                    PORCUPINE_ACCESS_KEY, PORCUPINE_KEYWORD_PATH,
                    EXTERNAL_WEBCAM_INDEX, EXTERNAL_MIC_INDEX, 
                    SYSTEM_PROMPT)

# Initialize the handlers
def main():
    # Initialize handlers
    tts_handler = TTSHandler()
    audio_handler = AudioHandler(device_index=EXTERNAL_MIC_INDEX)
    image_handler = ImageHandler(device_index=EXTERNAL_WEBCAM_INDEX)
    vision_handler = VisionHandler()
    nlp_handler = NLPHandler(api_key=GEMINI_API_KEY, system_prompt=SYSTEM_PROMPT)
    db_handler = DatabaseHandler(credentials_path=FIREBASE_CREDENTIALS)
    wake_word_detector = WakeWordDetector(access_key=PORCUPINE_ACCESS_KEY, keyword_path=PORCUPINE_KEYWORD_PATH)

    # Example main loop
    try:
        while True:
            # Wake word detection
            pcm_data = audio_handler.record_audio(duration=5)  # Adjust duration as needed
            if wake_word_detector.detect_wake_word(pcm_data):
                print("Wake word detected!")

                # Capture image
                image = image_handler.capture_image()
                if image:
                    # Detect text from image
                    detected_text = vision_handler.detect_text(image)
                    print(f"Detected text: {detected_text}")

                    # Translate and summarize the detected text
                    translated_text, summary_text = nlp_handler.translate_and_summarize(detected_text)
                    print(f"Translated text: {translated_text}")
                    print(f"Summary: {summary_text}")

                    # Save interaction to the database
                    db_handler.save_interaction(detected_text, summary_text)

                    # Convert summary to speech
                    tts_handler.text_to_speech(summary_text)

            time.sleep(1)  # Adjust the sleep duration as needed

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # Cleanup all resources
        tts_handler.cleanup()
        audio_handler.cleanup()
        image_handler.cleanup()
        vision_handler.cleanup()
        db_handler.cleanup()
        wake_word_detector.cleanup()

if __name__ == "__main__":
    main()
