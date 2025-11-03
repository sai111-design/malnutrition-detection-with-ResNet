import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

class FacePreprocessor:
    """Face detection and preprocessing using OpenCV"""

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def detect_and_crop_face(self, image_np):
        """Detect face and return cropped region"""
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            return image_np

        x, y, w, h = max(faces, key=lambda rect: rect[2]*rect[3])
        face = image_np[y:y+h, x:x+w]
        return face

    def preprocess(self, image_input):
        """Preprocess image: detect face, crop, resize, normalize"""
        if isinstance(image_input, Image.Image):
            image_np = np.array(image_input)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_np = image_input

        face = self.detect_and_crop_face(image_np)
        face_resized = cv2.resize(face, (224, 224))
        face_pil = Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))
        tensor = self.transform(face_pil)
        return tensor.unsqueeze(0)
