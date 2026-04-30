import cv2
import numpy as np
import time
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sqlalchemy.exc import IntegrityError

from db import SessionLocal
from models import FaceEmbedding, Attendance, Person
from antispoof import AntiSpoof

db = SessionLocal()

embeddings = {}

records = (
    db.query(FaceEmbedding, Person)
    .join(Person, FaceEmbedding.roll_number == Person.roll_number)
    .all()
)

for emb, person in records:
    key = f"{person.name}-{person.roll_number}"
    embeddings[key] = normalize(np.array(emb.embedding).reshape(1, -1))

print(f"Loaded embeddings for {len(embeddings)} students")

attendance_marked = set()

existing_attendance = db.query(Attendance.roll_number).all()

for (roll,) in existing_attendance:
    for key in embeddings.keys():
        if key.endswith(f"-{roll}"):
            attendance_marked.add(key)

# ------------------------
# MODELS
# ------------------------
app = FaceAnalysis(
    name="buffalo_l",
    root="./.insightface",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

antispoof = AntiSpoof()

THRESHOLD = 0.6

frame_count = 0
process_interval = 2
last_faces = []
last_detection_time = 0
cooldown = 0.3

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    frame_count += 1

    if frame_count % process_interval == 0:
        if time.time() - last_detection_time > cooldown:
            last_faces = app.get(frame)
            last_detection_time = time.time()

    faces = last_faces

    for face in faces:

        box = face.bbox.astype(int)
        x1, y1, x2, y2 = box

        is_real, _ = antispoof.check(frame, (x1, y1, x2, y2))

        if not is_real:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Spoof", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            continue

        embedding = normalize(face.embedding.reshape(1, -1))

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
                    attendance_marked.add(best_match)
                    print(f"✅ Attendance marked: {name} ({roll})")
                except IntegrityError:
                    db.rollback()

            # draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({best_score:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

    try:
        db.commit()
    except:
        db.rollback()

    fps = 1 / (time.time() - start_time)

    cv2.putText(frame, f"FPS: {int(fps)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 255), 1)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
db.close()