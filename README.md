# Steel Defect Detection + Reasoning App

This Streamlit web app detects defects in steel surface images using YOLOv8 and provides metallurgical reasoning with Gemini 2.5 Flash.

### Features
- Upload image
- YOLOv8 defect detection with bounding boxes
- Gemini reasoning for each defect
- Interactive report view
- Optional PDF export (future extension)

### Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
