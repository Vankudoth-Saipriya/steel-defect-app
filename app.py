

import os
import io
import re
import json
import time
import base64
import requests
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from dotenv import load_dotenv
from collections import defaultdict

# --- Load Environment ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_PATH = os.getenv("MODEL_PATH", "train_results_yolov8m/weights/best.pt")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY missing in your .env file")
    st.stop()

# --- Load YOLO model ---
@st.cache_resource(show_spinner=False)
def load_yolo_model():
    from ultralytics import YOLO
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return YOLO(MODEL_PATH)

# --- Gemini API ---
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

def ask_gemini(defect):
    prompt = f"""
You are a metallurgical defect analysis expert.
For defect '{defect}' in cold-rolled steel, write concise one-sentence summaries and detailed reasoning.

Format strictly as:
Root Cause Summary: ...
Root Cause Details: ...
Severity Analysis Summary: ...
Severity Analysis Details: ...
Corrective Action Summary: ...
Corrective Action Details: ...
Preventive Measure Summary: ...
Preventive Measure Details: ...
"""
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(GEMINI_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=30)
        if r.status_code == 200:
            data = r.json()
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except:
                return "⚠️ Incomplete Gemini response."
        else:
            return f"❌ API Error {r.status_code}: {r.text}"
    except Exception as e:
        return f"❌ Request Error: {e}"

# --- Helper to extract sections ---
SECTIONS = [
    ("Root Cause", "color-root"),
    ("Severity Analysis", "color-severity"),
    ("Corrective Action", "color-corrective"),
    ("Preventive Measure", "color-preventive"),
]

def extract_sections(reasoning):
    sections = {}
    for name, _ in SECTIONS:
        s_match = re.search(fr"{name} Summary:(.*)", reasoning)
        d_match = re.search(fr"{name} Details:(.*?)(?=\w+ Summary:|$)", reasoning, re.S)
        summary = s_match.group(1).strip() if s_match else "—"
        details = d_match.group(1).strip() if d_match else ""
        sections[name] = {"summary": summary, "details": details}
    return sections

# --- Draw bounding boxes ---
def draw_boxes(image, names, boxes_xyxy, scores, classes):
    im = image.convert("RGB").copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
    for xyxy, score, cls in zip(boxes_xyxy, scores, classes):
        x1, y1, x2, y2 = map(int, xyxy)
        label = f"{names[int(cls)]} {float(score):.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.text((x1 + 3, y1 - 15), label, fill=(255, 255, 255), font=font)
    return im

# --- Build HTML report ---
def build_html_report(all_results):
    html = """
<html>
<head>
<meta charset='utf-8'>
<title>Steel Surface Defect Report</title>
<style>
body { font-family: 'Segoe UI', Arial; margin: 40px; background: #f8f9fa; color: #222; }
h1 { text-align: center; color: #003366; }
h2 { color: #003366; margin-top: 20px; }
.card { background: white; border-radius: 10px; padding: 20px; margin-bottom: 25px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
.section { margin-top: 10px; }
.summary { font-weight: 500; }
.more { color: #0066cc; cursor: pointer; text-decoration: underline; margin-left: 8px; }
.details { display: none; margin-top: 5px; margin-left: 25px; color: #444; font-size: 0.95em; line-height: 1.4em; }
button { background-color: #003366; color: white; padding: 10px 18px; border: none; border-radius: 6px; cursor: pointer; font-size: 15px; margin: 0 5px; }
button:hover { background-color: #0055aa; }
img { max-width: 460px; margin: 10px 0; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
#topbar { text-align: center; margin-bottom: 25px; }
.color-root { color: #0d6efd; }
.color-severity { color: #dc3545; }
.color-corrective { color: #fd7e14; }
.color-preventive { color: #198754; }
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.9.2/html2pdf.bundle.min.js"></script>
<script>
function toggle(id, linkId) {
  var el = document.getElementById(id);
  var link = document.getElementById(linkId);
  if (el.style.display === 'none') { el.style.display = 'block'; link.innerText = 'Less...'; }
  else { el.style.display = 'none'; link.innerText = 'More...'; }
}
function generatePDF(full) {
  let details = document.querySelectorAll('.details');
  let toggles = document.querySelectorAll('.more');
  if (full) { details.forEach(d=>d.style.display='block'); toggles.forEach(t=>t.innerText='Less...'); }
  else { details.forEach(d=>d.style.display='none'); toggles.forEach(t=>t.innerText='More...'); }
  const opt = {
    margin: 0.5,
    filename: full ? 'Steel_Defect_Full_Report.pdf' : 'Steel_Defect_Summary_Report.pdf',
    image: { type: 'jpeg', quality: 0.98 },
    html2canvas: { scale: 2 },
    jsPDF: { unit: 'in', format: 'a4', orientation: 'portrait' }
  };
  html2pdf().set(opt).from(document.body).save();
}
</script>
</head><body>
<div id='topbar'>
  <h1>Steel Surface Defect Detection and Reasoning Report</h1>
  <p>Click “More…” to expand inline, or export using buttons below.</p>
  <button onclick="generatePDF(false)">📄 Summary PDF</button>
  <button onclick="generatePDF(true)">📘 Full PDF</button>
</div>
"""
    sec_id = 0
    for img_name, img_b64, defects in all_results:
        html += f"<div class='card'><h2>Image: {img_name}</h2>"
        html += f"<img src='data:image/png;base64,{img_b64}' />"
        for defect, sections in defects.items():
            html += f"<h3>Defect: {defect}</h3>"
            for name, color in SECTIONS:
                sec = sections[name]
                sec_id += 1
                html += f"""
                <div class='section'>
                  <div class='summary {color}'>&bull; <b>{name}:</b> {sec['summary']}
                    <span id='link{sec_id}' class='more' onclick="toggle('sec{sec_id}', 'link{sec_id}')">More...</span>
                  </div>
                  <div id='sec{sec_id}' class='details'>{sec['details']}</div>
                </div>
                """
        html += "</div>"
    html += "</body></html>"
    return html

# --- Streamlit UI ---
st.set_page_config(page_title="Steel Defect Detector + Reasoning", layout="wide")
st.title(" Steel Surface Defect Detection + Gemini Reasoning")

uploaded_images = st.file_uploader("Upload one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if st.button(" Run Detection and Reasoning"):
    if not uploaded_images:
        st.warning("Please upload at least one image.")
        st.stop()

    model = load_yolo_model()
    all_results = []

    for uploaded in uploaded_images:
        pil = Image.open(uploaded).convert("RGB")
        st.image(pil, caption=f"Uploaded: {uploaded.name}", use_container_width=True)

        res = model.predict(pil, conf=0.25, imgsz=1024, verbose=False)[0]
        if len(res.boxes) == 0:
            st.warning(f"No defects detected in {uploaded.name}")
            continue

        xyxy = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        boxed = draw_boxes(pil, res.names, xyxy, scores, classes)

        st.image(boxed, caption=f"Detections: {uploaded.name}", use_container_width=True)

        # Reasoning for each unique defect
        defects = sorted(set(res.names[int(c)] for c in classes))
        per_defect_sections = {}
        for defect in defects:
            with st.spinner(f"Reasoning for {defect}..."):
                text = ask_gemini(defect)
            st.subheader(defect)
            if text.startswith("❌"):
                st.error(text)
                continue
            sections = extract_sections(text)
            per_defect_sections[defect] = sections
            for name, _ in SECTIONS:
                s = sections[name]
                with st.expander(f"{name}: {s['summary']}"):
                    st.write(s['details'])

        buf = io.BytesIO()
        boxed.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        all_results.append((uploaded.name, img_b64, per_defect_sections))

    # Combined report
    if all_results:
        html_report = build_html_report(all_results)
        st.download_button(
            "⬇️ Download Interactive Report (HTML + PDF Options)",
            data=html_report.encode("utf-8"),
            file_name=f"Steel_Defect_Report_{int(time.time())}.html",
            mime="text/html",
        )
        st.success("✅ Report generated successfully!")
