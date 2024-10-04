import cv2
from PIL import Image
import numpy as np

class ImageHandler:
    def __init__(self, device_index=None):
        self.camera = cv2.VideoCapture(device_index or self.find_external_webcam())
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    def find_external_webcam(self):
        """Find an external webcam if available."""
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                return i if i > 0 else 0
        return 0

    def capture_image(self):
        """Captures an image and returns it in PIL format."""
        for _ in range(3):  # Multiple attempts to get a good frame
            ret, frame = self.camera.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                return Image.fromarray(rgb_frame)
        return None

    def save_image(self, image, path):
        """Saves a PIL Image to the specified path."""
        if isinstance(image, Image.Image):
            image.save(path)
            return True
        return False

    def load_image(self, path):
        """Loads an image from the specified path and returns it as a PIL Image."""
        try:
            return Image.open(path)
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def cleanup(self):
        """Release the camera resource."""
        if self.camera.isOpened():
            self.camera.release()
