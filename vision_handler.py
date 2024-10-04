import pytesseract
from PIL import Image

class VisionHandler:
    def __init__(self):
        # No credentials needed for Tesseract OCR
        pass

    def detect_text(self, image_path):
        # Open image using PIL
        img = Image.open(image_path)
        # Use pytesseract to do OCR on the image
        text = pytesseract.image_to_string(img, lang='ben')  # 'ben' for Bangla
        return text
