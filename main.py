import time
import logging
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

def initialize_logging():
    """Initialize logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('alpha_zero.log'), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

class SimplifiedWakeWordDetector:
    def __init__(self, wake_word="alpha"):
        self.recognizer = sr.Recognizer()
        self.wake_word = wake_word.lower()
        self.logger = logging.getLogger(__name__)
    
    def detect_wake_word(self, audio_data):
        if audio_data is None:
            return False
            
        try:
            text = self.recognizer.recognize_google(audio_data).lower()
            self.logger.info(f"Detected speech: {text}")
            return self.wake_word in text
        except sr.UnknownValueError:
            self.logger.info("Could not understand audio")
            return False
        except sr.RequestError as e:
            self.logger.error(f"Speech recognition error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error in wake word detection: {e}")
            return False
    
    def cleanup(self):
        pass

def print_separator():
    print("\n" + "="*50 + "\n")

def handle_questions(audio_handler, nlp_handler, tts_handler, context):
    """Handle interactive Q&A session about detected text/objects."""
    print("You can now ask questions about the detected text or objects. Say 'exit' to stop.")
    
    while True:
        question_audio = audio_handler.listen_for_speech(timeout=5)
        if question_audio:
            question = audio_handler.transcribe_audio(question_audio)
            if not question:  # If transcription failed
                tts_handler.text_to_speech("I couldn't understand that. Could you please repeat?")
                continue
                
            if question.lower() == "exit":
                break
                
            answer = nlp_handler.answer_question(question, context=context)
            print(f"Q: {question}")
            print(f"A: {answer}")
            tts_handler.text_to_speech(answer)

def main():
    logger = initialize_logging()
    print_separator()
    logger.info("Initializing ALPHA Zero System...")
    print_separator()
    
    handlers = {}  # Initialize handlers dictionary
    
    try:
        # Initialize all handlers
        logger.info("Initializing handlers...")
        
        handlers["TTS Handler"] = TTSHandler()
        handlers["Audio Handler"] = AudioHandler(device_index=EXTERNAL_MIC_INDEX)
        handlers["Image Handler"] = ImageHandler(device_index=EXTERNAL_WEBCAM_INDEX)
        handlers["Vision Handler"] = VisionHandler()
        handlers["NLP Handler"] = NLPHandler(api_key=GEMINI_API_KEY, system_prompt=SYSTEM_PROMPT)
        handlers["Database Handler"] = DatabaseHandler(credentials_path=FIREBASE_CREDENTIALS)
        handlers["Wake Word Detector"] = SimplifiedWakeWordDetector()
        
        # Verify all handlers initialized successfully
        for name, handler in handlers.items():
            if handler is None:
                raise RuntimeError(f"Failed to initialize {name}")
        
        logger.info("✓ All handlers initialized successfully")
        
        print_separator()
        logger.info("System initialization complete!")
        logger.info("Listening for wake word 'alpha'...")
        print_separator()

        while True:
            # Listen for wake word
            audio_data = handlers["Audio Handler"].listen_for_wake_word(timeout=5)
            
            if audio_data and handlers["Wake Word Detector"].detect_wake_word(audio_data):
                print_separator()
                logger.info("Wake word detected! Starting processing pipeline...")
                
                # Image Capture
                logger.info("\n1. Capturing image...")
                image = handlers["Image Handler"].capture_image(return_numpy=True)
                
                if image is not None:
                    try:
                        logger.info(f"✓ Image captured successfully. Shape: {image.shape}")
                        
                        # Text Detection and Processing
                        logger.info("\n2. Detecting and processing text...")
                        detected_text = handlers["Vision Handler"].detect_text(image)
                        
                        if detected_text:
                            logger.info(f"✓ Detected text: {detected_text}")
                            translated_text, summary_text = handlers["NLP Handler"].translate_and_summarize(detected_text)
                            logger.info(f"✓ Translated text: {translated_text}")
                            logger.info(f"✓ Summary: {summary_text}")
                            
                            # Save to database and provide audio feedback
                            handlers["Database Handler"].save_interaction("text_detection", detected_text, summary_text)
                            handlers["TTS Handler"].text_to_speech(summary_text)
                        else:
                            logger.warning("⚠ No text detected in the image")
                        
                        # Object Detection
                        logger.info("\n3. Detecting objects...")
                        detected_objects = handlers["Vision Handler"].detect_objects(image)
                        
                        if detected_objects:
                            object_description = handlers["NLP Handler"].describe_objects(detected_objects)
                            logger.info(f"✓ Detected objects: {object_description}")
                            
                            # Save to database and provide audio feedback
                            handlers["Database Handler"].save_interaction("object_detection", 
                                                      str(detected_objects), 
                                                      object_description)
                            handlers["TTS Handler"].text_to_speech(object_description)
                        else:
                            logger.warning("⚠ No objects detected in the image")
                        
                        # Question Answering Session
                        if detected_text or detected_objects:
                            context = f"Text: {detected_text}\nObjects: {str(detected_objects)}"
                            handle_questions(handlers["Audio Handler"], handlers["NLP Handler"], handlers["TTS Handler"], context)
                        
                    except Exception as e:
                        logger.error(f"\n❌ Error during processing: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        handlers["TTS Handler"].text_to_speech("Sorry, I encountered an error while processing. Please try again.")
                else:
                    logger.error("\n❌ Failed to capture image")
                    handlers["TTS Handler"].text_to_speech("Sorry, I couldn't capture an image. Please check the camera connection.")
                
                print_separator()
                logger.info("Resuming wake word detection...")
                print_separator()
        
            time.sleep(0.1)  # Small delay to prevent high CPU usage

    except KeyboardInterrupt:
        print_separator()
        logger.info("Shutdown initiated by user...")
    except Exception as e:
        print_separator()
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("\nCleaning up resources...")
        for name, handler in handlers.items():
            if handler is not None:
                try:
                    handler.cleanup()
                    logger.info(f"✓ {name} cleaned up")
                except Exception as e:
                    logger.error(f"❌ Error cleaning up {name}: {e}")
        print_separator()
        logger.info("System shutdown complete!")
        print_separator()

if __name__ == "__main__":
    main()
