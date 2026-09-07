# 🚨 Multimodal Anomaly & Accident Detection System

An AI-powered surveillance system that combines **computer vision, object tracking, temporal anomaly detection, and Vision-Language Models** to detect abnormal events and generate contextual incident reports from video footage.

---

## 📌 Overview

Traditional surveillance systems mainly rely on object detection and predefined rules. This project combines spatial, temporal, and semantic analysis to identify potentially abnormal events.

The system processes surveillance video through the following pipeline:

**Video → YOLOv8 → ByteTrack → Motion Features → GRU → Gemini Vision → Incident Report**

The system can detect abnormal motion patterns and, when an anomaly is detected, analyze the corresponding frame using a Vision-Language Model to provide contextual information such as incident type, severity, involved objects, and environment.

---

## 🎯 Objectives

- Process surveillance/CCTV video footage
- Detect objects using YOLOv8
- Track objects across frames using ByteTrack
- Extract temporal motion features from tracked objects
- Detect anomalous motion using a GRU-based deep learning model
- Capture the frame associated with the first detected anomaly
- Analyze the incident using a Vision-Language Model
- Generate structured incident reports
- Expose the complete detection pipeline through FastAPI
- Produce an annotated output video

---

## 🧠 System Architecture

```text
                    CCTV / Video
                         │
                         ▼
                 ┌──────────────┐
                 │   FastAPI    │
                 │ Video Upload │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │   YOLOv8     │
                 │Object Detect │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │  ByteTrack   │
                 │Object Tracking│
                 └──────┬───────┘
                        │
                        ▼
             ┌──────────────────────┐
             │   Motion Features    │
             │                      │
             │ • dx                 │
             │ • dy                 │
             │ • speed              │
             │ • acceleration       │
             │ • direction change   │
             │ • jerk               │
             └──────────┬───────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │     GRU      │
                 │ Temporal     │
                 │ Anomaly      │
                 │ Detection    │
                 └──────┬───────┘
                        │
                 Anomaly Detected
                        │
                        ▼
                 ┌──────────────┐
                 │ Incident     │
                 │ Frame        │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │ Gemini Vision│
                 │ VLM Analysis │
                 └──────┬───────┘
                        │
                        ▼
              ┌────────────────────┐
              │ Incident Report    │
              │                    │
              │ • Incident Type    │
              │ • Severity         │
              │ • Objects          │
              │ • Environment      │
              │ • Explanation      │
              └────────────────────┘
