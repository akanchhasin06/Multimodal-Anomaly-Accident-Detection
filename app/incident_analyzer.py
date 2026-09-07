import os
from dotenv import load_dotenv
from google import genai
from PIL import Image

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-3.7-flash"


def analyze_incident(image_path, anomaly_confidence, tracked_objects):

    image = Image.open(image_path)

    prompt = f"""
Analyze this surveillance image for an anomaly or accident.

Anomaly detector confidence: {anomaly_confidence:.2f}

Objects detected by the computer vision system:
{tracked_objects}

Tell me:

1. Incident type
2. Severity: Low, Medium, High, or Critical
3. Involved objects
4. Visible location/environment
5. Short explanation

Do not invent details that cannot be seen.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            image,
            prompt
        ]
    )

    return response.text


if __name__ == "__main__":

    print("Analyzing incident with Gemini...")

    result = analyze_incident(
        "outputs/test_frame.jpg",
        0.87,
        ["car", "person"]
    )

    print("\n===== INCIDENT ANALYSIS =====\n")
    print(result)