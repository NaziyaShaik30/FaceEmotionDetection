from flask import Flask, render_template, request
import os, base64, re
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
from Classification_model import EmotionCNN
# ------------------------------------------------
# Flask App Initialization
# ------------------------------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------------------------------
# Load Trained Emotion Model
# ------------------------------------------------
checkpoint = torch.load("emotion_model.pth", map_location="cpu")

model = EmotionCNN(num_classes=len(checkpoint["classes"]))
model.load_state_dict(checkpoint["model_state"])
model.eval()

EMOTIONS = checkpoint["classes"]

# ------------------------------------------------
# Image Preprocessing (same as training)
# ------------------------------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ------------------------------------------------
# Face Detectors
# ------------------------------------------------

# Haar Cascade (fast but weak)
haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# OpenCV DNN (very reliable)
proto = "deploy.prototxt.txt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"

face_net = None
if os.path.exists(proto) and os.path.exists(model_path):
    face_net = cv2.dnn.readNetFromCaffe(proto, model_path)


def detect_face_dnn(image):
    """
    Detect face using OpenCV deep learning model.
    Works well for side faces, lighting changes, webcam images.
    """
    h, w = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    best_conf = 0
    best_box = None

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > 0.5 and conf > best_conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            best_box = box.astype("int")
            best_conf = conf

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    return max(0, x1), max(0, y1), x2 - x1, y2 - y1


# ------------------------------------------------
# Routes
# ------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # -------------------------
    # Load Image
    # -------------------------
    if "file" in request.files and request.files["file"].filename != "":
        file = request.files["file"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        img = Image.open(filepath).convert("RGB")

    elif "camera_image" in request.form:
        data = request.form["camera_image"]
        data = re.sub("^data:image/.+;base64,", "", data)
        img = Image.open(BytesIO(base64.b64decode(data))).convert("RGB")
        filepath = os.path.join(UPLOAD_FOLDER, "camera.png")
        img.save(filepath)

    else:
        return "No image provided"

    # -------------------------
    # Convert to OpenCV
    # -------------------------
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # -------------------------
    # Face Detection
    # -------------------------
    faces = haar.detectMultiScale(gray, 1.1, 4)

    # If Haar fails → DNN
    if len(faces) == 0 and face_net is not None:
        dnn_face = detect_face_dnn(img_cv)
        if dnn_face:
            faces = [dnn_face]

    if len(faces) == 0:
        return render_template(
            "index.html",
            prediction="❌ No face detected",
            img_path=filepath
        )

    # Select largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    face = img_cv[y:y+h, x:x+w]

    # -------------------------
    # Emotion Prediction
    # -------------------------
    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    face_tensor = transform(face_pil).unsqueeze(0)

    with torch.no_grad():
        outputs = model(face_tensor)
        emotion = EMOTIONS[torch.argmax(outputs).item()]

    return render_template(
        "index.html",
        prediction=emotion,
        img_path=filepath
    )


if __name__ == "__main__":
    app.run(debug=True)
