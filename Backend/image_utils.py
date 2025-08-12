import cv2

def preprocess_image(path: str):
    img = cv2.imread(path)
    return cv2.resize(img, (640, 640))
