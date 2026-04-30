import cv2
import numpy as np
import time
import onnxruntime as ort


class AntiSpoof:
    def __init__(self, model_path="./.insightface/models/MiniFASNetV2.onnx", interval=1.0):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.face_state = {}       
        self.face_last_seen = {}   
        self.expire_time = 6

        self.cache = {}
        self.interval = interval

    def preprocess(self, face_img):
        face = cv2.resize(face_img, (80, 80))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32)
        face = (face - 127.5) / 128.0
        face = np.transpose(face, (2, 0, 1))  # HWC → CHW
        face = np.expand_dims(face, axis=0)
        return face

    def _predict_once(self, face_img):

        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        brightness = np.mean(gray)

        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = np.sum(edges > 0) / edges.size

        glare_ratio = np.sum(gray > 240) / gray.size

        brightness_flag = brightness > 200
        edge_flag = edge_ratio > 0.25
        glare_flag = glare_ratio > 0.1

        filter_fail = brightness_flag or edge_flag or glare_flag

        input_tensor = self.preprocess(face_img)
        output = self.session.run(None, {self.input_name: input_tensor})[0]

        output = np.array(output)

        exp = np.exp(output)
        probs = exp / np.sum(exp)

        real_score = float(probs[0, 2])
        spoof_score = float(max(probs[0, 0], probs[0, 1]))

        is_real = (
            not filter_fail and
            real_score > 0.7 and
            real_score > spoof_score + 0.2
        )

        # Debug (optional)
        # print(f"real={real_score:.2f}, filter_fail={filter_fail}")

        return is_real, real_score

    def check(self, frame, box):

        x1, y1, x2, y2 = box

        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            return False, 0.0

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        face_id = (
            int(cx / 50) * 50,
            int(cy / 50) * 50
        )

        current_time = time.time()

        self.face_last_seen[face_id] = current_time

        for fid in list(self.face_last_seen.keys()):
            if current_time - self.face_last_seen[fid] > self.expire_time:
                self.face_last_seen.pop(fid)
                self.face_state.pop(fid, None)

        if face_id in self.face_state:
            if self.face_state[face_id] == "spoof":
                return False, 0.0
            elif self.face_state[face_id] == "real":
                return True, 1.0

        is_real, score = self._predict_once(face_crop)

        if is_real:
            self.face_state[face_id] = "real"
            return True, score
        else:
            self.face_state[face_id] = "spoof"
            return False, score