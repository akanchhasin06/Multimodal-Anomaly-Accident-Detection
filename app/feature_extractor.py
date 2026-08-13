import cv2
import csv
import math
from ultralytics import YOLO

MODEL_PATH="yolov8n.pt"
model=YOLO(MODEL_PATH)

def extract_features(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Unable to open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps==0:
        fps=30

    frame_number=0
    previous_positions={}
    previous_speeds={}

    with open(output_csv, 'w', newline="") as file:
        writer=csv.writer(file)

        writer .writerow([
            "frame",
            "track_id",
            "class_id",
            "class_name",
            "x",
            "y",
            "dx",
            "dy",
            "speed",
            "acceleration"
        ])

        while True:
            success, frame=cap.read()

            if not success:
                break

            frame_number+=1
            
            results=model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

            boxes=results[0].boxes

            for box in boxes:

                if box.id is None:
                    continue

                track_id=int(box.id[0])

                x1,x2,y1,y2=map(int, box.xyxy[0])

                center_x=(x1+x2)//2
                center_y=(y1+y2)//2

                previous_position=previous_positions.get(track_id)

                if previous_position is None:
                    dx=0
                    dy=0
                    speed=0
                    acceleration=0

                else:
                    previous_x, previous_y = previous_position

                    dx = center_x - previous_x
                    dy = center_y - previous_y

                    speed = math.hypot(dx, dy)

                    previous_speed = previous_speeds.get(
                        track_id,
                        speed
                    )

                    acceleration = speed - previous_speed

                previous_positions[track_id] = (
                    center_x,
                    center_y
                )

                previous_speeds[track_id] = speed

                class_id = int(box.cls[0])

                class_name = results[0].names[class_id]

                writer.writerow([
                    frame_number,
                    track_id,
                    class_id,
                    class_name,
                    center_x,
                    center_y,
                    dx,
                    dy,
                    speed,
                    acceleration
                ])

    cap.release()

    print(f"Feature extraction complete: {output_csv}")
