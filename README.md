# YOLOv5 Object Detection Web App
### Deploy on Render.com — Free Tier

---

## Project Structure

```
yolov5-webapp/
├── app.py                  ← Flask backend
├── requirements.txt        ← Python dependencies
├── Procfile                ← Gunicorn start command
├── render.yaml             ← Render.com config
├── runtime.txt             ← Python version
├── .gitignore
├── templates/
│   └── index.html          ← Upload UI
└── static/
    └── results/
        └── .gitkeep
```

---
 — Test Locally

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
# Open: http://localhost:5000
```
