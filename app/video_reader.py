import cv2
from ultralytics import YOLO
from config import VIDEO_PATH

# -----------------------------
# Constants
# -----------------------------
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX

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
    counted_ids = set()
    vehicle_count = 0
    LINE_Y = 250

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
        # Display FPS & Frame Number
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
        # Process Every Detection
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

            cv2.circle(
                frame,
                (center_x, center_y),
                4,
                (0, 0, 255),
                -1
            )

            # -----------------------------
            # Store Previous Position
            # -----------------------------

            if track_id != -1:

                previous_y = previous_positions.get(track_id)
                if previous_y is not None:
                    if previous_y > LINE_Y and center_y <= LINE_Y:
                        if track_id not in counted_ids:

                            counted_ids.add(track_id)
                            vehicle_count += 1
                            print("Vehicle Count:", vehicle_count)

                previous_positions[track_id] = center_y

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
            label = f"{class_name} #{track_id} {confidence * 100:.1f}%"

            # -----------------------------
            # Draw Bounding Box
            # -----------------------------
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                BOX_COLOR,
                BOX_THICKNESS
            )

            # -----------------------------
            # Draw Label
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
        # Show Vehicle Count
        # (Still 0 for now)
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

        cv2.line(
            frame,
            (0, LINE_Y),
            (frame.shape[1], LINE_Y),
            (255, 0, 255),
            2
        )

        # -----------------------------
        # Display Video
        # -----------------------------
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(delay)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()