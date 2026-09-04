import cv2
import math
import numpy as np
import torch
from collections import defaultdict, deque
from ultralytics import YOLO


# =========================
# PATHS
# =========================

YOLO_MODEL_PATH = "yolov8n.pt"
GRU_MODEL_PATH = "models/best_anomaly_gru.pt"

MEAN_PATH = "models/feature_mean.npy"
STD_PATH = "models/feature_std.npy"

INPUT_VIDEO = "videos/sample2.mp4"
OUTPUT_VIDEO = "outputs/anomaly_result.mp4"


# =========================
# SETTINGS
# =========================

SEQUENCE_LENGTH = 30

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# GRU MODEL
# =========================

class AnomalyGRU(torch.nn.Module):

    def __init__(
        self,
        input_size=4,
        hidden_size=128,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        self.gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):

        output, _ = self.gru(x)

        last_output = output[:, -1, :]

        return self.classifier(last_output).squeeze(1)


# =========================
# LOAD MODELS
# =========================

print("Loading YOLO...")

yolo_model = YOLO(YOLO_MODEL_PATH)

print("Loading GRU...")

gru_model = AnomalyGRU()

checkpoint = torch.load(
    GRU_MODEL_PATH,
    map_location=DEVICE
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    gru_model.load_state_dict(checkpoint["model_state_dict"])
else:
    gru_model.load_state_dict(checkpoint)

gru_model.to(DEVICE)
gru_model.eval()


# =========================
# LOAD NORMALIZATION
# =========================

feature_mean = np.load(MEAN_PATH)
feature_std = np.load(STD_PATH)

feature_std[feature_std == 0] = 1.0


print("Device:", DEVICE)
print("Models loaded successfully.")


# =========================
# VIDEO
# =========================

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError(
        f"Unable to open video: {INPUT_VIDEO}"
    )


fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# =========================
# OUTPUT VIDEO
# =========================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


# =========================
# TRACK HISTORY
# =========================

previous_positions = {}
previous_speeds = {}

feature_sequences = defaultdict(
    lambda: deque(maxlen=SEQUENCE_LENGTH)
)


# =========================
# PROCESS VIDEO
# =========================

frame_number = 0

print("\nStarting anomaly detection...\n")


while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    # -------------------------
    # YOLO + BYTE TRACK
    # -------------------------

    results = yolo_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        device=0,
        verbose=False
    )

    boxes = results[0].boxes

    frame_anomaly = False
    frame_probability = 0.0


    # -------------------------
    # PROCESS EACH TRACK
    # -------------------------

    for box in boxes:

        if box.id is None:
            continue


        track_id = int(box.id[0])


        # Bounding box
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )


        # Center
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2


        # -------------------------
        # MOTION FEATURES
        # -------------------------

        previous_position = previous_positions.get(
            track_id
        )


        if previous_position is None:

            dx = 0
            dy = 0
            speed = 0
            acceleration = 0

        else:

            previous_x, previous_y = previous_position

            dx = center_x - previous_x
            dy = center_y - previous_y

            speed = math.hypot(
                dx,
                dy
            )

            previous_speed = previous_speeds.get(
                track_id,
                speed
            )

            acceleration = (
                speed - previous_speed
            )


        previous_positions[track_id] = (
            center_x,
            center_y
        )

        previous_speeds[track_id] = speed


        # -------------------------
        # FEATURE VECTOR
        # -------------------------

        features = np.array(
            [
                dx,
                dy,
                speed,
                acceleration
            ],
            dtype=np.float32
        )


        # -------------------------
        # NORMALIZATION
        # -------------------------

        features = (
            features - feature_mean
        ) / feature_std


        # -------------------------
        # ADD TO SEQUENCE
        # -------------------------

        feature_sequences[
            track_id
        ].append(features)


        probability = 0.0


        # -------------------------
        # GRU PREDICTION
        # -------------------------

        if len(
            feature_sequences[track_id]
        ) == SEQUENCE_LENGTH:

            sequence = np.array(
                feature_sequences[track_id],
                dtype=np.float32
            )

            # Make sure the sequence has exactly:
            # (sequence_length, number_of_features)
            sequence = sequence.reshape(
                SEQUENCE_LENGTH,
                4
            )

            sequence_tensor = torch.from_numpy(
                sequence
            ).unsqueeze(0).to(DEVICE)

            # Final shape must be (1, 30, 4)


            with torch.no_grad():

                logits = gru_model(
                    sequence_tensor
                )

                probability = torch.sigmoid(
                    logits
                ).item()


            frame_probability = max(
                frame_probability,
                probability
            )


            if probability >= 0.5:
                frame_anomaly = True


        # -------------------------
        # DRAW BOX
        # -------------------------

        label = (
            f"ID {track_id}"
            f" | {probability:.2f}"
        )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # =========================
    # FRAME STATUS
    # =========================

    if frame_anomaly:

        status = (
            f"ANOMALY DETECTED"
            f" | Confidence: "
            f"{frame_probability:.2f}"
        )

        status_color = (0, 0, 255)

    else:

        status = (
            f"NORMAL"
            f" | Confidence: "
            f"{1 - frame_probability:.2f}"
        )

        status_color = (0, 255, 0)


    # -------------------------
    # STATUS PANEL
    # -------------------------

    cv2.rectangle(
        frame,
        (10, 10),
        (520, 60),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        status,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2
    )


    # -------------------------
    # FRAME NUMBER
    # -------------------------

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # -------------------------
    # WRITE FRAME
    # -------------------------

    out.write(frame)


    # -------------------------
    # DISPLAY
    # -------------------------

    cv2.imshow(
        "Multimodal Anomaly Detection",
        frame
    )


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# =========================
# CLEANUP
# =========================

cap.release()
out.release()

cv2.destroyAllWindows()

print("\nDetection complete.")

print(
    f"Output saved to: {OUTPUT_VIDEO}"
)