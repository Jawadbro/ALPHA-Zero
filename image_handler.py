import cv2
from PIL import Image
import numpy as np

class ImageHandler:
    def __init__(self, device_index=None):
        self.camera = cv2.VideoCapture(device_index or self.find_external_webcam())
        if not self.camera.isOpened():
            raise RuntimeError("Failed to open camera")
            
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Verify camera settings
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera initialized with resolution: {actual_width}x{actual_height}")

    def find_external_webcam(self):
        """Find an external webcam if available."""
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get device name if possible
                if hasattr(cap, 'getBackendName'):
                    print(f"Found camera {i}: {cap.getBackendName()}")
                cap.release()
                return i if i > 0 else 0
        return 0

    def capture_image(self, return_numpy=True):
        """
        Captures an image and returns it in the specified format.
        
        Args:
            return_numpy (bool): If True, returns numpy array. If False, returns PIL Image.
        
        Returns:
            numpy.ndarray or PIL.Image or None: The captured image in the specified format.
        """
        for attempt in range(3):  # Multiple attempts to get a good frame
            ret, frame = self.camera.read()
            if ret and frame is not None:
                try:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    if return_numpy:
                        return rgb_frame  # Already a numpy array
                    else:
                        return Image.fromarray(rgb_frame)
                        
                except Exception as e:
                    print(f"Error processing frame on attempt {attempt + 1}: {e}")
                    continue
                    
            print(f"Failed capture attempt {attempt + 1}")
            
        print("All capture attempts failed")
        return None

    def save_image(self, image, path):
        """
        Saves an image to the specified path.
        
        Args:
            image: Can be either PIL Image or numpy array
            path (str): Path to save the image
        """
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            if isinstance(image, Image.Image):
                image.save(path)
                print(f"Image saved successfully to {path}")
                return True
                
            print(f"Unsupported image type: {type(image)}")
            return False
            
        except Exception as e:
            print(f"Error saving image: {e}")
            return False

    def load_image(self, path, return_numpy=True):
        """
        Loads an image from the specified path.
        
        Args:
            path (str): Path to the image file
            return_numpy (bool): If True, returns numpy array. If False, returns PIL Image.
            
        Returns:
            numpy.ndarray or PIL.Image or None: The loaded image in the specified format.
        """
        try:
            image = Image.open(path)
            if return_numpy:
                return np.array(image)
            return image
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def cleanup(self):
        """Release the camera resource."""
        if self.camera.isOpened():
            self.camera.release()
            print("Camera released successfully")