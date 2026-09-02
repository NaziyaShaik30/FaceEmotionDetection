import cv2
import numpy as np
import torch

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def preprocess_image(image_path, target_size=(48, 48)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    if len(faces) == 0:
        return None

    (x, y, w, h) = faces[0]
    face = img[y:y + h, x:x + w]
    face = cv2.resize(face, target_size)
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)  # channel
    face = np.expand_dims(face, axis=0)  # batch
    return torch.tensor(face, dtype=torch.float32)
