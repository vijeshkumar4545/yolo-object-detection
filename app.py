from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import numpy as np
import os
import uuid

app = Flask(__name__)
model = YOLO("yolov8n.pt")   # ya best.pt

UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]

        # Read image
        img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        # YOLO detection
        results = model(img, conf=0.3)
        annotated = results[0].plot()

        # Unique filename
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        # Save processed image
        cv2.imwrite(filepath, annotated)

        return render_template("index.html", image=filename)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)