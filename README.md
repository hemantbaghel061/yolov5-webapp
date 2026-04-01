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

## STEP 1 — Test Locally (optional but recommended)

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

---

## STEP 2 — Upload to GitHub

1. Go to https://github.com and sign in (or create a free account)
2. Click the green **"New"** button → create a repository
   - Name it: `yolov5-webapp`
   - Set to **Public**
   - Click **Create repository**

3. Open a terminal inside your project folder and run:

```bash
git init
git add .
git commit -m "Initial commit - YOLOv5 web app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/yolov5-webapp.git
git push -u origin main
```

> Replace `YOUR_USERNAME` with your actual GitHub username.

---

## STEP 3 — Deploy on Render.com

1. Go to https://render.com → Sign up with GitHub (free)

2. Click **"New +"** → **"Web Service"**

3. Click **"Connect a repository"** → select `yolov5-webapp`

4. Fill in the settings:
   - **Name**: `yolov5-detector` (or any name)
   - **Region**: Oregon (US West)
   - **Branch**: `main`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --preload`
   - **Plan**: Free

5. Click **"Create Web Service"**

6. Wait 5–10 minutes for the first build to complete.

7. Your public URL will look like:
   ```
   https://yolov5-detector.onrender.com
   ```

---

## Public URL Structure

```
https://your-app-name.onrender.com/          ← Upload page
https://your-app-name.onrender.com/detect    ← Detection API (POST)
```

---

## Common Errors & Fixes

### ❌ "Build failed – pip install error"
**Fix:** Make sure `requirements.txt` is in the root of your repo, not inside a subfolder.

### ❌ "Application failed to respond"
**Fix:** The free tier sleeps after inactivity. First request takes 30–60 seconds to wake up. This is normal.

### ❌ "out of memory" / app crashes
**Fix:** The free tier has 512MB RAM. The app is already optimized with `--workers 1` and model loaded once. If it still crashes, go to Render dashboard → Settings → increase timeout to 180s.

### ❌ "No module named 'cv2'"
**Fix:** Make sure you're using `opencv-python-headless` (not `opencv-python`) in requirements.txt. Headless version works on servers without display.

### ❌ "git push" fails with authentication error
**Fix:** Use a GitHub Personal Access Token instead of your password:
- GitHub → Settings → Developer settings → Personal access tokens → Generate new token
- Use that token as your password when prompted.

### ❌ Slow detection (20+ seconds)
**Cause:** Free tier CPU is slow. yolov5s is already the smallest model. This is expected on free tier.
**Fix:** Nothing needed — it will still work, just slower than on your laptop.

---

## Notes

- The model (`yolov5s`) downloads automatically on first startup from PyTorch Hub
- It is cached after the first download
- Results are stored temporarily in `static/results/` — they are NOT permanent on Render's free tier (ephemeral filesystem)
- The app uses base64 inline images so results still show correctly in the browser even without persistent storage
