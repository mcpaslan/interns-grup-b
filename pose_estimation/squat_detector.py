import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import csv
import json

class SquatDetector:
    def __init__(self,
                 down_thresh: float = 70,
                 up_thresh: float   = 160,
                 min_det_conf: float = 0.5,
                 min_track_conf: float = 0.5):
        # Thresholds
        self.DOWN_THRESH = down_thresh
        self.UP_THRESH   = up_thresh
        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf)
        # State machine
        self.state = 0              # 0=standing,1=down
        self.squat_count = 0
        # Logging
        self.log = []               # kayıt listesi
        self.current_pose = "Standing"
        self.state_start_time = datetime.now().isoformat()

    @staticmethod
    def angle(a, b, c):
        a, b, c = map(lambda x: np.array(x, dtype=np.float32), (a, b, c))
        ba = a - b; bc = c - b
        cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-8)
        return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

    def _record_log(self, old_pose, start_ts, end_ts):
        """Eski poz için bir log girdisi ekle."""
        # süre saniye cinsinden
        start = datetime.fromisoformat(start_ts)
        end   = datetime.fromisoformat(end_ts)
        duration = (end - start).total_seconds()
        self.log.append({
            "pose": old_pose,
            "start": start_ts,
            "end": end_ts,
            "duration_s": duration
        })

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        now_ts = datetime.now().isoformat()
        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            coord = lambda pt: (int(pt.x*w), int(pt.y*h))

            # gerekli noktalar
            lh = coord(lm[self.mp_pose.PoseLandmark.LEFT_HIP])
            lk = coord(lm[self.mp_pose.PoseLandmark.LEFT_KNEE])
            la = coord(lm[self.mp_pose.PoseLandmark.LEFT_ANKLE])
            rk = coord(lm[self.mp_pose.PoseLandmark.RIGHT_KNEE])
            ra = coord(lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE])

            # diz açıları
            left_ang  = self.angle(lh, lk, la)
            right_ang = self.angle(lh, rk, ra)
            avg_ang   = (left_ang + right_ang) / 2.0

            # hangi pozdayız?
            new_pose = self.current_pose
            if self.state == 0 and avg_ang < self.DOWN_THRESH:
                # standing -> down
                new_pose = "Squatting"
                self.state = 1
            elif self.state == 1 and avg_ang > self.UP_THRESH:
                # down -> standing
                new_pose = "Standing"
                self.state = 0
                self.squat_count += 1

            # poz değiştiyse log kaydı yap
            if new_pose != self.current_pose:
                self._record_log(self.current_pose,
                                 self.state_start_time,
                                 now_ts)
                # yeni poz başlangıç zamanı
                self.current_pose = new_pose
                self.state_start_time = now_ts

            # görselleştirme
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )
            cv2.putText(frame, f"Squats: {self.squat_count}", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0,255,0), 3)
            cv2.putText(frame, f"Pose: {self.current_pose}", (10,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (200,200,0), 3)

        return frame

    def finish(self):
        """Video bittiğinde son poz kaydını tamamla."""
        end_ts = datetime.now().isoformat()
        self._record_log(self.current_pose,
                         self.state_start_time,
                         end_ts)

    def export_csv(self, path="behavior_log.csv"):
        fieldnames = ["pose","start","end","duration_s"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.log)

    def export_json(self, path="behavior_log.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.log, f, ensure_ascii=False, indent=2)
