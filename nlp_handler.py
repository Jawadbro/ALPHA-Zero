import logging
from typing import Dict, List, Tuple, Optional
import google.generativeai as genai
from langdetect import detect, LangDetectException

class NLPHandler:
    def __init__(self, api_key: str, system_prompt: str):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing NLPHandler...")
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-pro-latest')
            self.system_prompt = system_prompt
            self.logger.info("Gemini model initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing Gemini model: {e}")
            raise

    def translate_and_summarize(self, text: str) -> Tuple[str, str]:
        """
        Translate Bangla text to English if necessary and generate a summary.
        
        Args:
            text (str): Input text in Bangla or English
            
        Returns:
            Tuple[str, str]: (translated_text, summary_text)
        """
        if not text.strip():
            return "", "No text detected."
            
        try:
            # Detect language with error handling
            try:
                lang = detect(text)
                self.logger.info(f"Detected language: {lang}")
            except LangDetectException as e:
                self.logger.warning(f"Language detection failed: {e}. Defaulting to English.")
                lang = 'en'
            
            # Translate if text is in Bangla
            if lang == 'bn':
                translation_prompt = (
                    f"{self.system_prompt}\n"
                    f"Translate the following Bengali text to English accurately:\n"
                    f"Text: {text}\n"
                    f"Translation:"
                )
                
                try:
                    translation_response = self.model.generate_content(translation_prompt)
                    translated_text = translation_response.text.strip()
                    self.logger.info("Translation successful")
                except Exception as e:
                    self.logger.error(f"Translation failed: {e}")
                    return text, "Translation failed. Processing original text."
            else:
                translated_text = text
            
            # Generate summary
            summary_prompt = (
                f"{self.system_prompt}\n"
                f"Provide a clear and concise summary of the following text in 2-3 sentences:\n"
                f"Text: {translated_text}\n"
                f"Summary:"
            )
            
            try:
                summary_response = self.model.generate_content(summary_prompt)
                summary_text = summary_response.text.strip()
                self.logger.info("Summary generation successful")
            except Exception as e:
                self.logger.error(f"Summary generation failed: {e}")
                return translated_text, "Summary generation failed."
            
            return translated_text, summary_text
            
        except Exception as e:
            self.logger.error(f"Error in translate_and_summarize: {e}")
            return text, "Error processing text."

    def describe_objects(self, objects: List[Dict]) -> str:
        """
        Generate a natural language description of detected objects.
        
        Args:
            objects (List[Dict]): List of detected objects with their properties
            
        Returns:
            str: Natural language description of the objects
        """
        if not objects:
            return "No objects detected in the image."
            
        try:
            # Create a detailed object description
            object_details = []
            for obj in objects:
                confidence_str = f" (confidence: {obj['confidence']:.2%})" if obj['confidence'] is not None else ""
                location = f"at coordinates ({obj['bbox'][0]}, {obj['bbox'][1]})"
                object_details.append(f"A {obj['label']}{confidence_str} {location}")
            
            object_description = "\n".join(object_details)
            
            description_prompt = (
                f"{self.system_prompt}\n"
                f"I've detected the following objects in an image:\n{object_description}\n\n"
                f"Please provide a natural, conversational description of what's in the image, "
                f"including the number and types of objects detected. Focus on the most important "
                f"or interesting elements."
            )
            
            try:
                description_response = self.model.generate_content(description_prompt)
                return description_response.text.strip()
            except Exception as e:
                self.logger.error(f"Object description generation failed: {e}")
                return f"Detected objects: {', '.join(obj['label'] for obj in objects)}"
                
        except Exception as e:
            self.logger.error(f"Error in describe_objects: {e}")
            return "Error processing object descriptions."

    def answer_question(self, question: str, context: str) -> str:
        """
        Answer questions about detected text and objects using the provided context.
        
        Args:
            question (str): User's question
            context (str): Context including detected text and objects
            
        Returns:
            str: Answer to the question
        """
        if not question.strip():
            return "I couldn't understand the question. Please try again."
            
        try:
            # Create a more focused prompt for question answering
            answer_prompt = (
                f"{self.system_prompt}\n"
                f"You are answering questions about an image based on the following information:\n"
                f"{context}\n\n"
                f"Please answer this question: {question}\n"
                f"If you cannot answer based on the provided information, please say so clearly.\n"
                f"Answer:"
            )
            
            try:
                answer_response = self.model.generate_content(answer_prompt)
                return answer_response.text.strip()
            except Exception as e:
                self.logger.error(f"Question answering failed: {e}")
                return "I encountered an error while trying to answer your question. Please try again."
                
        except Exception as e:
            self.logger.error(f"Error in answer_question: {e}")
            return "Sorry, I'm having trouble processing your question."

    def cleanup(self):
        """Cleanup any resources"""
        self.logger.info("Cleaning up NLPHandler...")
        # Add any cleanup code if needed
        pass