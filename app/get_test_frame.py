import cv2
import os

INPUT_VIDEO = "videos/sample2.mp4"
OUTPUT_IMAGE = "outputs/test_frame.jpg"

os.makedirs("outputs", exist_ok=True)

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError("Could not open video")

# Go to frame 300
cap.set(cv2.CAP_PROP_POS_FRAMES, 300)

success, frame = cap.read()

if not success:
    raise RuntimeError("Could not read frame")

cv2.imwrite(OUTPUT_IMAGE, frame)

cap.release()

print("Test frame saved to:")
print(OUTPUT_IMAGE)