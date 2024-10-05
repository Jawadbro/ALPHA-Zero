from PIL import Image
import pytesseract
import numpy as np
import cv2
from ultralytics import YOLO

class VisionHandler:
    def __init__(self):
        # Initialize the YOLO model (you can specify different versions or custom weights)
        self.model = YOLO('yolov8s.pt')  # You can use 'yolov8s.pt' for a larger model

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
        
        pil_image = Image.fromarray(image)
        text = pytesseract.image_to_string(pil_image, lang='ben+eng')  # Use Bangla and English languages
        return text

    def detect_objects(self, image):
        """
        Detect objects in an image using YOLO or fallback on contour-based detection (placeholder).
        
        Args:
            image (numpy.ndarray): Image in RGB format.
        
        Returns:
            list: Detected objects with their bounding boxes and labels.
        """
        # YOLO object detection
        try:
            results = self.model(image)  # Run YOLO object detection
            detected_objects = []
            for result in results.xyxy[0]:  # Extract bounding boxes, confidence, and class
                x1, y1, x2, y2, conf, cls = result
                detected_objects.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'label': self.model.names[int(cls)],  # Get class label
                    'confidence': float(conf)
                })
            return detected_objects

        except Exception as e:
            print(f"YOLO detection failed: {e}")
            # Fallback to basic contour-based detection as a placeholder
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            objects = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    objects.append({
                        'bbox': (x, y, w, h),
                        'label': 'Unknown Object'  # Placeholder classification
                    })
            return objects

    def cleanup(self):
        """Perform cleanup if necessary."""
        pass  # Add any cleanup code if needed
