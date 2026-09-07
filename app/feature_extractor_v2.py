import cv2
import csv
import math
from ultralytics import YOLO


MODEL_PATH = "yolov8n.pt"

model = YOLO(MODEL_PATH)


def extract_features(video_path, output_csv):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Unable to open video")
        return

    frame_number = 0

    previous_positions = {}
    previous_speeds = {}
    previous_accelerations = {}
    previous_directions = {}

    with open(
        output_csv,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "frame",
            "track_id",
            "class_id",
            "class_name",
            "x",
            "y",
            "dx",
            "dy",
            "speed",
            "acceleration",
            "direction_change",
            "jerk"
        ])

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                device=0,
                verbose=False
            )

            boxes = results[0].boxes

            for box in boxes:

                if box.id is None:
                    continue

                track_id = int(box.id[0])

                # Preserve the original extractor's
                # coordinate behavior exactly.
                x1, x2, y1, y2 = map(
                    int,
                    box.xyxy[0]
                )

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                previous_position = (
                    previous_positions.get(track_id)
                )

                previous_speed = (
                    previous_speeds.get(
                        track_id,
                        0
                    )
                )

                previous_acceleration = (
                    previous_accelerations.get(
                        track_id,
                        0
                    )
                )

                previous_direction = (
                    previous_directions.get(
                        track_id
                    )
                )

                # ==========================================
                # FIRST OBSERVATION
                # ==========================================

                if previous_position is None:

                    dx = 0
                    dy = 0
                    speed = 0
                    acceleration = 0
                    direction_change = 0
                    jerk = 0

                    current_direction = None

                # ==========================================
                # SUBSEQUENT OBSERVATION
                # ==========================================

                else:

                    previous_x, previous_y = (
                        previous_position
                    )

                    dx = (
                        center_x -
                        previous_x
                    )

                    dy = (
                        center_y -
                        previous_y
                    )

                    speed = math.hypot(
                        dx,
                        dy
                    )

                    acceleration = (
                        speed -
                        previous_speed
                    )

                    # -------------------------------
                    # Direction
                    # -------------------------------

                    if dx == 0 and dy == 0:

                        current_direction = (
                            previous_direction
                        )

                    else:

                        current_direction = math.atan2(
                            dy,
                            dx
                        )

                    if (
                        current_direction is None
                        or previous_direction is None
                    ):

                        direction_change = 0

                    else:

                        angle_difference = (
                            current_direction
                            - previous_direction
                        )

                        # Wrap to [-pi, pi]
                        angle_difference = (
                            angle_difference + math.pi
                        ) % (
                            2 * math.pi
                        ) - math.pi

                        direction_change = abs(
                            angle_difference
                        )

                    # -------------------------------
                    # Jerk
                    # -------------------------------

                    jerk = (
                        acceleration
                        - previous_acceleration
                    )

                # ==========================================
                # SAVE STATE
                # ==========================================

                previous_positions[track_id] = (
                    center_x,
                    center_y
                )

                previous_speeds[track_id] = (
                    speed
                )

                previous_accelerations[track_id] = (
                    acceleration
                )

                if current_direction is not None:

                    previous_directions[track_id] = (
                        current_direction
                    )

                # ==========================================
                # OBJECT INFORMATION
                # ==========================================

                class_id = int(
                    box.cls[0]
                )

                class_name = (
                    results[0].names[class_id]
                )

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
                    acceleration,
                    direction_change,
                    jerk
                ])

    cap.release()

    print(
        f"Feature extraction complete: {output_csv}"
    )