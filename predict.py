import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from sqlalchemy.exc import IntegrityError

from db import SessionLocal
from models import FaceEmbedding, Attendance, Person

# --------------------------------
# Load embeddings FROM DB (CSV style)
# --------------------------------
db = SessionLocal()

embeddings = {}

records = (
    db.query(FaceEmbedding, Person)
    .join(Person, FaceEmbedding.roll_number == Person.roll_number)
    .all()
)

for emb, person in records:
    key = f"{person.name}-{person.roll_number}"
    embeddings[key] = np.array(emb.embedding).reshape(1, -1)

print(f"Loaded embeddings for {len(embeddings)} students from DB")

# --------------------------------
# Face model
# --------------------------------
app = FaceAnalysis(
    name="buffalo_l",
    root="./.insightface",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

# --------------------------------
# Attendance state (SAME AS CSV)
# --------------------------------
attendance_marked = set()
THRESHOLD = 0.6
USE_WEBCAM = 1

# --------------------------------
# IMAGE MODE (testing)
# --------------------------------
if not USE_WEBCAM:
    # img_path = r"D:\cutie\car\lw\IMG20240809192124.jpg"
    # frame = cv2.imread(img_path)
    # if frame is None:
    #     print("Image not found")
    #     exit()

    # faces = app.get(frame)

    # for face in faces:
    #     embedding = face.embedding.reshape(1, -1)

    #     best_match = None
    #     best_score = 0.0

    #     for student, ref_embedding in embeddings.items():
    #         score = cosine_similarity(embedding, ref_embedding)[0][0]
    #         if score > best_score:
    #             best_match = student
    #             best_score = score

    #     if best_match and best_score > THRESHOLD:
    #         name, roll = best_match.split("-")

    #         if best_match not in attendance_marked:
    #             try:
    #                 db.add(Attendance(roll_number=roll))
    #                 db.commit()
    #             except IntegrityError:
    #                 db.rollback()

    #             attendance_marked.add(best_match)
    #             print(f"Attendance marked: {name} ({roll})")

    #         box = face.bbox.astype(int)
    #         cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    #         cv2.putText(frame, name, (box[0], box[1] - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # cv2.imshow("Image Result", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    pass

# --------------------------------
# WEBCAM MODE (SAME CSV LOGIC)
# --------------------------------
else:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)

        for face in faces:
            embedding = face.embedding.reshape(1, -1)

            best_match = None
            best_score = 0.0

            for student, ref_embedding in embeddings.items():
                score = cosine_similarity(embedding, ref_embedding)[0][0]
                if score > best_score:
                    best_match = student
                    best_score = score

            if best_match and best_score > THRESHOLD:
                name, roll = best_match.split("-")

                if best_match not in attendance_marked:
                    try:
                        db.add(Attendance(roll_number=roll))
                        db.commit()
                        print(f"Attendance marked: {name} ({roll})")
                    except IntegrityError:
                        db.rollback()

                    attendance_marked.add(best_match)

                box = face.bbox.astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame, name, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

db.close()
