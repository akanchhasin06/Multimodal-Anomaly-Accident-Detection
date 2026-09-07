from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
import shutil
import subprocess

from app.anomaly_detector import detect_video


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Multimodal Anomaly & Accident Detection",
    description="API for video anomaly detection and incident analysis",
    version="1.0.0"
)


# ============================================================
# BASIC ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Multimodal Anomaly & Accident Detection API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# DETECTION ENDPOINT
# ============================================================

@app.post("/detect")
def detect(file: UploadFile = File(...)):

    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # --------------------------------------------------------
    # Save uploaded video
    # --------------------------------------------------------

    input_path = os.path.join(
        "uploads",
        file.filename
    )

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    output_video = os.path.join(
        "outputs",
        "anomaly_result_api.mp4"
    )

    report_path = os.path.join(
        "outputs",
        "incident_report_api.json"
    )

    browser_video = os.path.join(
        "outputs",
        "anomaly_result_browser.mp4"
    )

    # --------------------------------------------------------
    # Run ML pipeline
    # --------------------------------------------------------

    result = detect_video(
        input_video=input_path,
        output_video=output_video,
        report_path=report_path,
        display=False
    )

    # --------------------------------------------------------
    # Convert video to browser-compatible H.264
    # --------------------------------------------------------

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            output_video,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            browser_video
        ],
        check=True
    )

    # --------------------------------------------------------
    # Extract severity from Gemini report
    # --------------------------------------------------------

    incident_report = result.get(
        "incident_report"
    )

    severity = None

    if incident_report:

        if "**Severity:** Critical" in incident_report:
            severity = "Critical"

        elif "**Severity:** High" in incident_report:
            severity = "High"

        elif "**Severity:** Medium" in incident_report:
            severity = "Medium"

        elif "**Severity:** Low" in incident_report:
            severity = "Low"

    # --------------------------------------------------------
    # Clean API response
    # --------------------------------------------------------

    return {
        "status": result["status"],
        "filename": file.filename,

        "incident_detected": result[
            "incident_detected"
        ],

        "incident": {
            "frame": result["incident_frame"],
            "anomaly_confidence": result[
                "anomaly_confidence"
            ],
            "severity": severity,
            "objects": result[
                "tracked_objects"
            ]
        },

        "incident_analysis": incident_report,

        "video_url": "/video",
        "report_url": "/report"
    }


# ============================================================
# PROCESSED VIDEO
# ============================================================

@app.get("/video")
def get_video():

    video_path = (
        "outputs/anomaly_result_browser.mp4"
    )

    if not os.path.exists(video_path):

        return {
            "error": "Output video not found"
        }

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename="anomaly_result_browser.mp4"
    )


# ============================================================
# INCIDENT REPORT
# ============================================================

@app.get("/report")
def get_report():

    report_path = (
        "outputs/incident_report_api.json"
    )

    if not os.path.exists(report_path):

        return {
            "error": "Incident report not found"
        }

    return FileResponse(
        report_path,
        media_type="application/json",
        filename="incident_report_api.json"
    )