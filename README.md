# Face Recognition Based Smart Attendance System

This project implements a **Face Recognition Based Smart Attendance System** capable of detecting and recognizing **multiple faces (50+ students)** in a single classroom image or live camera feed using **InsightFace (buffalo_l model)**.

The system generates face embeddings for registered students and matches them during attendance marking.  

---

## 🎥 Prequisites
- Docker Desktop
- python v3+
- download [Git lfs](https://github.com/git-lfs/git-lfs/releases/download/v3.7.1/git-lfs-windows-v3.7.1.exe)

---

## 👤 Student Registration

Each student must have a separate folder inside `image/` named using their roll number.

Folder format:

Create folder in this format `name-roll_nummber` then put face images for that specific student.
Minimum 4 clear frontal images per student are recommended.

---

## 🧠Steps to run
- Run one by one
```bash
git lfs install
git clone https://github.com/Swapnil-070904/major-proj.git
cd major-proj
pip install -r req.txt
docker compose up -d
```
- python embeddings.py (only when registering faces for students)
- python predict.py

