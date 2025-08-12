import cv2
import numpy as np

def read_image(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def encode_image(image: np.ndarray):
    _, buffer = cv2.imencode('.jpg', image)
    return buffer.tobytes()
