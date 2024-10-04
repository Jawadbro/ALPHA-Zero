from PIL import Image
import pytesseract
import numpy as np

class VisionHandler:
    def detect_text(self, image):
        """
        Detect text from an image using Tesseract OCR.

        Args:
            image (numpy.ndarray): Image in RGB format.

        Returns:
            str: Detected text.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array.")

        pil_image = Image.fromarray(image)  # Convert the numpy array to a PIL image
        text = pytesseract.image_to_string(pil_image, lang='ben')  # Use Bangla language
        return text
    def cleanup(self):
        """Perform cleanup if necessary."""
        pass  # Add any cleanup code if needed
   