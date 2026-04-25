import cv2
import numpy as np
import time
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sqlalchemy.exc import IntegrityError

from db import SessionLocal
from models import FaceEmbedding, Attendance, Person

# ------------------------
# Load embeddings FROM DB
# ------------------------
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

print(f"Loaded embeddings for {len(embeddings)} students from DB")

# ------------------------
# Load already marked attendance from DB
# ------------------------
attendance_marked = set()

existing_attendance = db.query(Attendance.roll_number).all()

for (roll,) in existing_attendance:
    for key in embeddings.keys():
        if key.endswith(f"-{roll}"):
            attendance_marked.add(key)

# ------------------------
# Initialize model (GPU)
# ------------------------
app = FaceAnalysis(
    name="buffalo_l",
    root="./.insightface",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

# ------------------------
# Config
# ------------------------
THRESHOLD = 0.6
USE_WEBCAM = 1

# Performance controls
frame_count = 0
process_interval = 2
last_faces = []
last_detection_time = 0
cooldown = 0.3

# ------------------------
# WEBCAM MODE
# ------------------------
if USE_WEBCAM:

    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # ---- Frame skipping + cooldown ----
        frame_count += 1

        if frame_count % process_interval == 0:
            if time.time() - last_detection_time > cooldown:
                last_faces = app.get(frame)
                last_detection_time = time.time()

        faces = last_faces

        # ---- Process faces ----
        for face in faces:
            embedding = normalize(face.embedding.reshape(1, -1))

            best_match = None
            best_score = 0.0

            # ---- Compare with DB embeddings ----
            for student, ref_embedding in embeddings.items():
                score = cosine_similarity(embedding, ref_embedding)[0][0]
                if score > best_score:
                    best_match = student
                    best_score = score

            if best_match and best_score > THRESHOLD:

                name, roll = best_match.split("-")

                # ---- CHECK DB (REAL SOURCE OF TRUTH) ----
                already_in_db = db.query(Attendance).filter_by(roll_number=roll).first()

                if not already_in_db:
                    try:
                        db.add(Attendance(roll_number=roll))
                        print(f"✅ Attendance marked: {name} ({roll})")
                    except IntegrityError:
                        db.rollback()

                # ---- Update RAM state ----
                attendance_marked.add(best_match)

                # ---- ALWAYS draw box ----
                box = face.bbox.astype(int)
                cv2.rectangle(frame, (box[0], box[1]),
                              (box[2], box[3]), (0, 255, 0), 2)

                label = f"{name} ({best_score:.2f})"
                cv2.putText(frame, label,
                            (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

        # ---- Commit once per frame ----
        try:
            db.commit()
        except:
            db.rollback()

        # ---- FPS display ----
        end_time = time.time()
        fps = 1 / (end_time - start_time)

        cv2.putText(frame, f"FPS: {int(fps)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1)

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------------
# CLEANUP
# ------------------------
db.close()