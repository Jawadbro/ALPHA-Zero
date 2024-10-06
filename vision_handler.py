from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import cv2
from PIL import Image
import pytesseract
from ultralytics import YOLO
from googletrans import Translator

class VisionHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing VisionHandler...")
        
        # Initialize YOLO
        try:
            self.logger.info("Loading YOLO model...")
            self.model = YOLO('yolov8s.pt')
            self.logger.info("YOLO model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading YOLO model: {e}")
            self.model = None
        
        # Initialize translator
        try:
            self.translator = Translator()
            self.logger.info("Translator initialized successfully.")
        except Exception as e:
            self.logger.error(f"Error initializing translator: {e}")
            self.translator = None

    def detect_and_process_text(self, image: np.ndarray) -> Dict[str, str]:
        """
        Detects text in both English and Bangla and processes it accordingly.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Dict[str, str]: Dictionary containing original text, translated text (if applicable),
                           and a flag indicating if the text is in Bangla
        """
        self.logger.info("Starting text detection and processing...")
        
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array.")
        
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Detect text in both English and Bangla
            text = pytesseract.image_to_string(pil_image, lang='ben+eng')
            
            if not text.strip():
                return {
                    "original_text": "",
                    "translated_text": "",
                    "is_bangla": False
                }
            
            # Check if text contains Bangla characters
            is_bangla = any(ord(char) >= 0x0980 and ord(char) <= 0x09FF for char in text)
            
            if is_bangla and self.translator:
                try:
                    translated = self.translator.translate(text, src='bn', dest='en')
                    translated_text = translated.text
                    self.logger.info(f"Translation successful: {translated_text}")
                    return {
                        "original_text": text,
                        "translated_text": translated_text,
                        "is_bangla": True
                    }
                except Exception as e:
                    self.logger.error(f"Translation error: {e}")
                    return {
                        "original_text": text,
                        "translated_text": "",
                        "is_bangla": True
                    }
            else:
                return {
                    "original_text": text,
                    "translated_text": "",
                    "is_bangla": False
                }
                
        except Exception as e:
            self.logger.error(f"Error in text detection: {e}")
            return {
                "original_text": "",
                "translated_text": "",
                "is_bangla": False
            }

    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Detects objects in the image using YOLO or falls back to contour detection.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            List[Dict]: List of detected objects with their properties
        """
        self.logger.info("Starting object detection...")
        
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array.")
        
        if self.model is None:
            self.logger.warning("YOLO model not available. Falling back to contour-based detection.")
            return self._contour_based_detection(image)
        
        try:
            # Run YOLO detection with a lower confidence threshold
            results = self.model(image, conf=0.25)
            
            detected_objects = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    
                    obj = {
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'label': self.model.names[cls],
                        'confidence': float(conf)
                    }
                    detected_objects.append(obj)
                    self.logger.info(f"Detected object: {obj}")
            
            return detected_objects
            
        except Exception as e:
            self.logger.error(f"YOLO detection failed: {e}")
            return self._contour_based_detection(image)

    def _contour_based_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Fallback method using contour detection.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            List[Dict]: List of detected objects with their properties
        """
        self.logger.info("Performing contour-based detection...")
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            objects = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    obj = {
                        'bbox': (x, y, x + w, y + h),
                        'label': 'Unknown Object',
                        'confidence': None
                    }
                    objects.append(obj)
                    
            return objects
            
        except Exception as e:
            self.logger.error(f"Error in contour-based detection: {e}")
            return []

    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up VisionHandler...")
        if self.translator:
            del self.translator
        if self.model:
            del self.model