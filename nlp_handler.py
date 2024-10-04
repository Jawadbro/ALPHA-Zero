import google.generativeai as genai

class NLPHandler:
    def __init__(self, api_key, system_prompt):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')
        self.system_prompt = system_prompt  # Storing the system prompt

    def translate_and_summarize(self, bangla_text):
        """Translate Bangla text to English and summarize it."""
        translation_prompt = f"{self.system_prompt}\nTranslate the following text from Bangla to English: {bangla_text}"
        translation_response = self.model.generate_content(translation_prompt)
        translated_text = translation_response.text.strip()

        summary_prompt = f"{self.system_prompt}\nSummarize the following text: {translated_text}"
        summary_response = self.model.generate_content(summary_prompt)
        summary_text = summary_response.text.strip()

        return translated_text, summary_text
