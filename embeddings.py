# store_embeddings.py
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

from db import SessionLocal
from models import FaceEmbedding

# -------------------------------
# Face model
# -------------------------------
app = FaceAnalysis(
    name="buffalo_l",
    root="./.insightface",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

# -------------------------------
# DB session
# -------------------------------
db = SessionLocal()

image_dir = "./image"

for folder_name in os.listdir(image_dir):
    folder_path = os.path.join(image_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    # -------------------------------
    # Parse folder name: Name-Roll
    # -------------------------------
    try:
        student_name, roll_number = folder_name.rsplit("-", 1)
    except ValueError:
        print(f"Skipping invalid folder name: {folder_name}")
        continue

    embeddings = []

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = app.get(img)
        if faces:
            embeddings.append(faces[0].embedding)

    if not embeddings:
        print(f"No face found for {folder_name}")
        continue

    # -------------------------------
    # EXACT CSV LOGIC
    # -------------------------------
    avg_embedding = np.mean(embeddings, axis=0)
    avg_embedding = avg_embedding.tolist()

    # -------------------------------
    # Store / Update embedding
    # -------------------------------
    record = (
        db.query(FaceEmbedding)
        .filter(FaceEmbedding.roll_number == roll_number)
        .first()
    )

    if record:
        record.embedding = avg_embedding
    else:
        db.add(
            FaceEmbedding(
                roll_number=roll_number,
                embedding=avg_embedding
            )
        )

    db.commit()
    print(f"Stored embedding for {student_name} ({roll_number})")

db.close()
