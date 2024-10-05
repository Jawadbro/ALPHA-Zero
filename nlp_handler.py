import google.generativeai as genai
from langdetect import detect

class NLPHandler:
    def __init__(self, api_key, system_prompt):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')
        self.system_prompt = system_prompt

    # UPDATED METHOD
    def translate_and_summarize(self, text):
        """
        Translate Bangla text to English if necessary, and summarize the text.
        
        Args:
            text (str): Input text in Bangla or English.
        
        Returns:
            tuple: (translated_text, summary_text)
        """
        try:
            lang = detect(text)
        except:
            lang = 'en'  # Default to English if detection fails
        
        if lang == 'bn':
            # Translate Bangla to English
            translation_prompt = f"{self.system_prompt}\nTranslate the following text from Bangla to English: {text}"
            translation_response = self.model.generate_content(translation_prompt)
            translated_text = translation_response.text.strip()
        else:
            translated_text = text
        
        # Summarize the text
        summary_prompt = f"{self.system_prompt}\nSummarize the following text: {translated_text}"
        summary_response = self.model.generate_content(summary_prompt)
        summary_text = summary_response.text.strip()
        
        return translated_text, summary_text

    # NEW METHOD
    def describe_objects(self, objects):
        """
        Generate a description of detected objects.
        
        Args:
            objects (list): List of detected objects with their properties.
        
        Returns:
            str: Description of the detected objects.
        """
        object_description = f"I detected {len(objects)} object(s) in the image:\n"
        for i, obj in enumerate(objects, 1):
            object_description += f"{i}. A {obj['label']} at position (x: {obj['bbox'][0]}, y: {obj['bbox'][1]})\n"
        
        description_prompt = f"{self.system_prompt}\nGiven the following object detections, provide a brief, natural language description:\n{object_description}"
        description_response = self.model.generate_content(description_prompt)
        return description_response.text.strip()

    # UPDATED METHOD
    def answer_question(self, question, context):
        """
        Answer a question based on the provided context using the Gemini model.
        
        Args:
            question (str): The question to be answered.
            context (str): The context information (detected text and objects).
        
        Returns:
            str: The answer to the question.
        """
        answer_prompt = f"{self.system_prompt}\nContext: {context}\n\nQuestion: {question}\nAnswer:"
        answer_response = self.model.generate_content(answer_prompt)
        return answer_response.text.strip()