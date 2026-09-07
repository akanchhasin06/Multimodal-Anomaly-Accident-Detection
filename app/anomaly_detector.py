import cv2
import math
import json
import os
import numpy as np
import torch

from collections import defaultdict, deque
from ultralytics import YOLO
from dotenv import load_dotenv
from google import genai
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_MODEL = "gemini-3.7-flash"

YOLO_MODEL_PATH = "yolov8n.pt"

GRU_MODEL_PATH = "models/best_anomaly_gru_v2.pt"

MEAN_PATH = "models/feature_mean_v2.npy"
STD_PATH = "models/feature_std_v2.npy"

SEQUENCE_LENGTH = 30
ANOMALY_THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# GRU MODEL
# ============================================================

class AnomalyGRU(torch.nn.Module):

    def __init__(
        self,
        input_size=6,
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

        return self.classifier(
            last_output
        ).squeeze(1)


# ============================================================
# LOAD MODELS ONCE
# ============================================================

print("Loading YOLO...")

yolo_model = YOLO(YOLO_MODEL_PATH)

print("Loading V2 GRU...")

gru_model = AnomalyGRU()

checkpoint = torch.load(
    GRU_MODEL_PATH,
    map_location=DEVICE
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    gru_model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    gru_model.load_state_dict(checkpoint)

gru_model.to(DEVICE)
gru_model.eval()


# ============================================================
# NORMALIZATION
# ============================================================

feature_mean = np.load(MEAN_PATH)

feature_std = np.load(STD_PATH)

feature_std[
    feature_std == 0
] = 1.0


print("Device:", DEVICE)
print("V2 features: 6")
print("Gemini:", GEMINI_MODEL)
print("Models loaded successfully.")


# ============================================================
# GEMINI INCIDENT ANALYSIS
# ============================================================

def analyze_incident(
    image_path,
    anomaly_confidence,
    tracked_objects
):

    print("\nSending incident frame to Gemini...")

    image = Image.open(image_path)

    prompt = f"""
You are an incident analysis system for a
multimodal anomaly and accident detection platform.

Analyze the provided surveillance frame.

The temporal anomaly detector produced:

Anomaly confidence:
{anomaly_confidence:.3f}

Objects detected by the computer vision system:
{tracked_objects}

Provide:

1. Incident type
2. Severity: Low, Medium, High, or Critical
3. Involved objects
4. Visible location/environment
5. Short explanation

Do not invent details that cannot reasonably
be inferred from the image.
"""

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            image,
            prompt
        ]
    )

    return response.text


# ============================================================
# MAIN DETECTION FUNCTION
# ============================================================

def detect_video(
    input_video,
    output_video="outputs/anomaly_result_v2.mp4",
    report_path="outputs/incident_report.json",
    display=True
):

    os.makedirs("outputs", exist_ok=True)

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():

        raise RuntimeError(
            f"Unable to open video: {input_video}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 30

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    # --------------------------------------------------------
    # OUTPUT VIDEO
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        output_video,
        fourcc,
        fps,
        (width, height)
    )

    # --------------------------------------------------------
    # TRACK HISTORY
    # --------------------------------------------------------

    previous_positions = {}

    previous_speeds = {}

    previous_accelerations = {}

    previous_directions = {}

    feature_sequences = defaultdict(
        lambda: deque(
            maxlen=SEQUENCE_LENGTH
        )
    )

    # --------------------------------------------------------
    # INCIDENT STATE
    # --------------------------------------------------------

    incident_detected = False

    incident_report = None

    incident_frame_path = None

    incident_frame_number = None

    incident_confidence = 0.0

    incident_objects = []

    # --------------------------------------------------------
    # PROCESS VIDEO
    # --------------------------------------------------------

    frame_number = 0

    print(
        f"\nStarting detection on: {input_video}\n"
    )

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # ====================================================
        # YOLO + BYTE TRACK
        # ====================================================

        results = yolo_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            device=0 if torch.cuda.is_available() else "cpu",
            verbose=False
        )

        boxes = results[0].boxes

        frame_anomaly = False

        frame_probability = 0.0

        tracked_objects = []

        # ====================================================
        # PROCESS TRACKS
        # ====================================================

        for box in boxes:

            if box.id is None:
                continue

            track_id = int(
                box.id[0]
            )

            # ------------------------------------------------
            # Object class
            # ------------------------------------------------

            class_id = int(
                box.cls[0]
            )

            class_name = results[0].names[
                class_id
            ]

            tracked_objects.append(
                class_name
            )

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # ------------------------------------------------
            # Center
            # ------------------------------------------------

            center_x = (
                x1 + x2
            ) // 2

            center_y = (
                y1 + y2
            ) // 2

            # =================================================
            # MOTION FEATURES
            # =================================================

            previous_position = (
                previous_positions.get(
                    track_id
                )
            )

            previous_speed = (
                previous_speeds.get(
                    track_id,
                    0.0
                )
            )

            previous_acceleration = (
                previous_accelerations.get(
                    track_id,
                    0.0
                )
            )

            previous_direction = (
                previous_directions.get(
                    track_id
                )
            )

            if previous_position is None:

                dx = 0.0
                dy = 0.0
                speed = 0.0
                acceleration = 0.0
                direction_change = 0.0
                jerk = 0.0

                current_direction = None

            else:

                previous_x, previous_y = (
                    previous_position
                )

                dx = (
                    center_x - previous_x
                )

                dy = (
                    center_y - previous_y
                )

                speed = math.hypot(
                    dx,
                    dy
                )

                acceleration = (
                    speed - previous_speed
                )

                # Direction

                if dx == 0 and dy == 0:

                    current_direction = (
                        previous_direction
                    )

                else:

                    current_direction = math.atan2(
                        dy,
                        dx
                    )

                # Direction change

                if (
                    current_direction is None
                    or previous_direction is None
                ):

                    direction_change = 0.0

                else:

                    angle_difference = (
                        current_direction
                        - previous_direction
                    )

                    angle_difference = (
                        angle_difference + math.pi
                    ) % (
                        2 * math.pi
                    ) - math.pi

                    direction_change = abs(
                        angle_difference
                    )

                # Jerk

                jerk = (
                    acceleration
                    - previous_acceleration
                )

            # =================================================
            # UPDATE HISTORY
            # =================================================

            previous_positions[
                track_id
            ] = (
                center_x,
                center_y
            )

            previous_speeds[
                track_id
            ] = speed

            previous_accelerations[
                track_id
            ] = acceleration

            if current_direction is not None:

                previous_directions[
                    track_id
                ] = current_direction

            # =================================================
            # FEATURE VECTOR
            # =================================================

            features = np.array(
                [
                    dx,
                    dy,
                    speed,
                    acceleration,
                    direction_change,
                    jerk
                ],
                dtype=np.float32
            )

            # =================================================
            # NORMALIZATION
            # =================================================

            features = (
                features - feature_mean
            ) / feature_std

            # =================================================
            # SEQUENCE
            # =================================================

            feature_sequences[
                track_id
            ].append(features)

            probability = 0.0

            # =================================================
            # GRU
            # =================================================

            if len(
                feature_sequences[
                    track_id
                ]
            ) == SEQUENCE_LENGTH:

                sequence = np.array(
                    feature_sequences[
                        track_id
                    ],
                    dtype=np.float32
                )

                sequence = sequence.reshape(
                    SEQUENCE_LENGTH,
                    6
                )

                sequence_tensor = (
                    torch.from_numpy(
                        sequence
                    )
                    .unsqueeze(0)
                    .to(DEVICE)
                )

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

                if (
                    probability
                    >= ANOMALY_THRESHOLD
                ):

                    frame_anomaly = True

            # =================================================
            # DRAW BOX
            # =================================================

            label = (
                f"ID {track_id}"
                f" | {class_name}"
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
                (
                    x1,
                    max(y1 - 10, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        # ====================================================
        # GEMINI TRIGGER
        # ====================================================

        if (
            frame_anomaly
            and not incident_detected
        ):

            incident_detected = True

            incident_frame_number = frame_number

            incident_confidence = (
                frame_probability
            )

            incident_frame_path = (
                "outputs/incident_frame.jpg"
            )

            cv2.imwrite(
                incident_frame_path,
                frame
            )

            incident_objects = list(
                dict.fromkeys(
                    tracked_objects
                )
            )

            try:

                incident_report = (
                    analyze_incident(
                        incident_frame_path,
                        frame_probability,
                        incident_objects
                    )
                )

                with open(
                    report_path,
                    "w",
                    encoding="utf-8"
                ) as file:

                    json.dump(
                        {
                            "frame": frame_number,
                            "anomaly_confidence":
                                frame_probability,
                            "tracked_objects":
                                incident_objects,
                            "llm_analysis":
                                incident_report
                        },
                        file,
                        indent=4
                    )

                print(
                    "\n===== INCIDENT REPORT ====="
                )

                print(
                    incident_report
                )

                print(
                    f"\nReport saved to: {report_path}"
                )

            except Exception as error:

                print(
                    "\nGemini analysis failed:"
                )

                print(error)

        # ====================================================
        # STATUS
        # ====================================================

        if frame_anomaly:

            status = (
                "ANOMALY DETECTED"
                f" | Confidence: "
                f"{frame_probability:.2f}"
            )

            status_color = (
                0,
                0,
                255
            )

        else:

            status = (
                "NORMAL"
                f" | Confidence: "
                f"{1 - frame_probability:.2f}"
            )

            status_color = (
                0,
                255,
                0
            )

        # ====================================================
        # STATUS PANEL
        # ====================================================

        cv2.rectangle(
            frame,
            (10, 10),
            (600, 60),
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

        # ====================================================
        # FRAME NUMBER
        # ====================================================

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (
                10,
                height - 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ====================================================
        # WRITE OUTPUT
        # ====================================================

        out.write(frame)

        # ====================================================
        # DISPLAY
        # ====================================================

        if display:

            cv2.imshow(
                "Multimodal Anomaly Detection",
                frame
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    out.release()
    if display:
        cv2.destroyAllWindows()

    print("\nDetection complete.")
    print(f"Output video: {output_video}")

    # ========================================================
    # RETURN API-FRIENDLY RESULT
    # ========================================================

    return {
        "status": "completed",
        "output_video": output_video,
        "incident_detected": incident_detected,
        "incident_frame": incident_frame_number,
        "anomaly_confidence": incident_confidence,
        "tracked_objects": incident_objects,
        "incident_report": incident_report,
        "report_path": (
            report_path
            if incident_report
            else None
        )
    }


# ============================================================
# STANDALONE EXECUTION
# ============================================================

if __name__ == "__main__":

    result = detect_video(
        input_video="videos/sample2.mp4"
    )

    print("\nFinal result:")
    print(
        json.dumps(
            result,
            indent=4
        )
    )