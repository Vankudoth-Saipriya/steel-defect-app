# ============================================================
#  Steel Surface Defect Detection + Gemini 2.5 Reasoning App
# Streamlit interface for YOLOv8 + Gemini
# ============================================================

import streamlit as st
import os, json, re, requests, gdown
from ultralytics import YOLO
from PIL import Image, ImageDraw

# ----------------------------
# CONFIGURATION
# ----------------------------
st.set_page_config(
    page_title="Steel Defect Detection + Reasoning",
    page_icon=" ",
    layout="wide"
)

# Paths
MODEL_DIR = "train_results_yolov8m/weights"
MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Replace this with your Drive file ID ---
DRIVE_FILE_ID = "YOUR_FILE_ID_HERE"

# Gemini API
GEMINI_KEY = "AIzaSyBdMxoq2Ma8R2ZXas1ANZrOYj2Bj4xvhTk"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"

# ----------------------------
# UTILITY FUNCTIONS
# ----------------------------
def download_model_from_drive():
    """Download model file from Google Drive if not present"""
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading YOLOv8 model from Google Drive... (first run only)")
        gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully ")

def ask_gemini(defect):
    """Query Gemini reasoning API for metallurgical defect analysis"""
    prompt = f"""
You are a metallurgical defect analysis expert.
For defect '{defect}' in cold-rolled steel, write concise one-sentence summaries and detailed reasoning.

Format strictly as:
Root Summary: ...
Root Details: ...
Severity Summary: ...
Severity Details: ...
Corrective Summary: ...
Corrective Details: ...
Preventive Summary: ...
Preventive Details: ...
"""
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    r = requests.post(GEMINI_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
    if r.status_code == 200:
        try:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except:
            return "⚠️ Gemini response incomplete."
    else:
        return f" API Error {r.status_code}"

def extract_sections(reasoning):
    """Split Gemini response into structured sections"""
    sections = {}
    for key in ["Root", "Severity", "Corrective", "Preventive"]:
        s_match = re.search(fr"{key} Summary:(.*)", reasoning)
        d_match = re.search(fr"{key} Details:(.*?)(?=\w+ Summary:|$)", reasoning, re.S)
        summary = s_match.group(1).strip() if s_match else "—"
        details = d_match.group(1).strip() if d_match else ""
        sections[key] = {"summary": summary, "details": details}
    return sections

def draw_boxes(image, boxes, names):
    """Draw YOLO detections on image"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cls = int(box.cls[0])
        name = names[cls]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), name, fill="red")
    return img

# ----------------------------
# APP LAYOUT
# ----------------------------
st.title(" Steel Surface Defect Detection + Gemini Reasoning")
st.markdown("Upload an image to detect defects using **YOLOv8** and get metallurgical reasoning from **Gemini 2.5 Flash**.")
st.markdown("---")

uploaded_image = st.file_uploader(" Upload a steel surface image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button(" Detect and Analyze"):
        with st.spinner("Running detection and reasoning... please wait"):
            download_model_from_drive()
            model = YOLO(MODEL_PATH)

            # Run YOLO detection
            results = model.predict(image, conf=0.25)
            boxes = results[0].boxes
            names = results[0].names
            detected_classes = list(set([names[int(c)] for c in boxes.cls]))

            st.subheader(" Detection Results")
            st.write("Detected defects:", ", ".join(detected_classes) if detected_classes else "None")

            # Show image with bounding boxes
            boxed_img = draw_boxes(image, boxes, names)
            st.image(boxed_img, caption="Detected Defects", use_column_width=True)

            # Reasoning for each detected defect
            for defect in detected_classes:
                st.markdown(f"###  {defect.upper()}")
                reasoning = ask_gemini(defect)
                sections = extract_sections(reasoning)
                for sec, val in sections.items():
                    with st.expander(f" {sec} Analysis"):
                        st.markdown(f"**Summary:** {val['summary']}")
                        st.markdown(f"**Details:** {val['details']}")

            st.success(" Analysis Complete!")

else:
    st.info("Please upload an image to start detection.")
