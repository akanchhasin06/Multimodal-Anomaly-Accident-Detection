import cv2
import math
from ultralytics import YOLO
from config import VIDEO_PATH

# -----------------------------
# Constants
# -----------------------------

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX

LINE_Y = 250
MAX_TRAJECTORY_LENGTH = 30

# -----------------------------
# Load YOLO Model
# -----------------------------

model = YOLO("yolov8n.pt")


def run():

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Unable to open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 30

    delay = int(1000 / fps)

    frame_number = 0

    # -----------------------------
    # Tracking Variables
    # -----------------------------

    previous_positions = {}
    trajectory_history = {}

    counted_ids = set()
    vehicle_count = 0

    # -----------------------------
    # Main Loop
    # -----------------------------

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # -----------------------------
        # YOLO + ByteTrack
        # -----------------------------

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        boxes = results[0].boxes

        # -----------------------------
        # FPS & Frame Number
        # -----------------------------

        cv2.putText(
            frame,
            f"FPS : {fps:.2f}",
            (10, 30),
            FONT,
            1,
            (0, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Frame : {frame_number}",
            (10, 70),
            FONT,
            1,
            (0, 0, 0),
            2
        )

        # -----------------------------
        # Process Each Detection
        # -----------------------------

        for box in boxes:

            # -----------------------------
            # Track ID
            # -----------------------------

            track_id = -1

            if box.id is not None:
                track_id = int(box.id[0])

            # -----------------------------
            # Bounding Box
            # -----------------------------

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # -----------------------------
            # Centroid
            # -----------------------------

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # -----------------------------
            # Draw Centroid
            # -----------------------------

            cv2.circle(
                frame,
                (center_x, center_y),
                4,
                (0, 0, 255),
                -1
            )

            # -----------------------------
            # Tracking + Motion
            # -----------------------------

            if track_id != -1:

                # -----------------------------
                # Initialize Trajectory
                # -----------------------------

                if track_id not in trajectory_history:
                    trajectory_history[track_id] = []

                # -----------------------------
                # Add Current Position
                # -----------------------------

                trajectory_history[track_id].append(
                    (center_x, center_y)
                )

                # -----------------------------
                # Keep Last 30 Positions
                # -----------------------------

                if len(trajectory_history[track_id]) > MAX_TRAJECTORY_LENGTH:
                    trajectory_history[track_id].pop(0)

                # -----------------------------
                # Get Previous Position
                # -----------------------------

                previous_position = previous_positions.get(track_id)

                if previous_position is not None:

                    previous_x, previous_y = previous_position

                    # -----------------------------
                    # Movement
                    # -----------------------------

                    dx = center_x - previous_x
                    dy = center_y - previous_y

                    # -----------------------------
                    # Speed in Pixels / Frame
                    # -----------------------------

                    speed = math.hypot(dx, dy)

                    print(
                        f"ID: {track_id} | "
                        f"dx: {dx} | "
                        f"dy: {dy} | "
                        f"Speed: {speed:.2f}"
                    )

                    # -----------------------------
                    # Vehicle Counting
                    # -----------------------------

                    if previous_y > LINE_Y and center_y <= LINE_Y:

                        if track_id not in counted_ids:

                            counted_ids.add(track_id)
                            vehicle_count += 1

                            print(
                                "Vehicle Count:",
                                vehicle_count
                            )

                # -----------------------------
                # Update Previous Position
                # -----------------------------

                previous_positions[track_id] = (
                    center_x,
                    center_y
                )

                # -----------------------------
                # Draw Trajectory
                # -----------------------------

                points = trajectory_history[track_id]

                for i in range(1, len(points)):

                    cv2.line(
                        frame,
                        points[i - 1],
                        points[i],
                        (255, 255, 0),
                        2
                    )

            # -----------------------------
            # Class Name
            # -----------------------------

            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]

            # -----------------------------
            # Confidence
            # -----------------------------

            confidence = box.conf.item()

            # -----------------------------
            # Label
            # -----------------------------

            label = (
                f"{class_name} "
                f"#{track_id} "
                f"{confidence * 100:.1f}%"
            )

            # -----------------------------
            # Bounding Box
            # -----------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                BOX_COLOR,
                BOX_THICKNESS
            )

            # -----------------------------
            # Label
            # -----------------------------

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                FONT,
                0.7,
                TEXT_COLOR,
                2
            )

        # -----------------------------
        # Vehicle Count
        # -----------------------------

        cv2.putText(
            frame,
            f"Vehicles : {vehicle_count}",
            (10, 110),
            FONT,
            1,
            (255, 0, 0),
            2
        )

        # -----------------------------
        # Counting Line
        # -----------------------------

        cv2.line(
            frame,
            (0, LINE_Y),
            (frame.shape[1], LINE_Y),
            (255, 0, 255),
            2
        )

        # -----------------------------
        # Display Frame
        # -----------------------------

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(delay)

        if key == ord("q"):
            break

    # -----------------------------
    # Cleanup
    # -----------------------------

    cap.release()
    cv2.destroyAllWindows()