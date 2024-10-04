# File: nlp_handler.py
import google.generativeai as genai

class NLPHandler:
    def __init__(self, api_key, system_prompt):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')
        self.system_prompt = system_prompt  # Storing the system prompt

    def translate_and_summarize(self, bangla_text):
        # Incorporating the system prompt for translation
        translation_prompt = f"{self.system_prompt}\nTranslate the following text from Bangla to English: {bangla_text}"
        translation_response = self.model.generate_content(translation_prompt)
        translated_text = translation_response.text

        # Incorporating the system prompt for summarization
        summary_prompt = f"{self.system_prompt}\nSummarize the following text: {translated_text}"
        summary_response = self.model.generate_content(summary_prompt)
        summary_text = summary_response.text

        return translated_text, summary_text
