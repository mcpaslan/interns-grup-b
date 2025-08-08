import cv2
from squat_detector import SquatDetector

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    detector = SquatDetector(down_thresh=70, up_thresh=160)

    while True:
        ret, frame = cap.read()
        if not ret: break
        out = detector.process_frame(frame)
        cv2.imshow("Squat Tracker", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Bitiş ve export
    detector.finish()
    detector.export_csv("squat_behavior.csv")
    detector.export_json("squat_behavior.json")
    cap.release()
    cv2.destroyAllWindows()
    print("Loglar kaydedildi: squat_behavior.csv, squat_behavior.json")

if __name__ == "__main__":
    main("squat.mp4")
