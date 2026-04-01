import os
import io
import uuid
import threading
import base64
import pathlib

import torch
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from PIL import Image

pathlib.WindowsPath = pathlib.PosixPath

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

print("Loading YOLOv5s model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
model.eval()
print("Model ready.")

UPLOAD_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

rtsp_streams = {}
rtsp_lock = threading.Lock()


def run_detection_on_pil(img):
    results = model(img, size=640)
    results.render()
    result_img = Image.fromarray(results.ims[0])
    buf = io.BytesIO()
    result_img.save(buf, format='JPEG', quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    detections = []
    df = results.pandas().xyxy[0]
    for _, row in df.iterrows():
        detections.append({
            'label': row['name'],
            'confidence': round(float(row['confidence']) * 100, 1)
        })
    return f"data:image/jpeg;base64,{b64}", detections, len(detections)


@app.route('/')
def index():
    # html = open("templates/index.html").read()
    html = open("templates/index.html", encoding="utf-8").read()
    return render_template_string(html)


@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        b64, detections, count = run_detection_on_pil(img)
        return jsonify({'success': True, 'image_b64': b64,
                        'detections': detections, 'count': count})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    data = request.get_json(force=True, silent=True)
    if not data or 'frame' not in data:
        return jsonify({'error': 'No frame data'}), 400
    try:
        raw = data['frame']
        if ',' in raw:
            raw = raw.split(',', 1)[1]
        img_bytes = base64.b64decode(raw)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        b64, detections, count = run_detection_on_pil(img)
        return jsonify({'success': True, 'image_b64': b64,
                        'detections': detections, 'count': count})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/rtsp/open', methods=['POST'])
def rtsp_open():
    try:
        import cv2
    except ImportError:
        return jsonify({'error': 'opencv-python-headless not installed'}), 500
    data = request.get_json(force=True, silent=True) or {}
    url = (data.get('url') or '').strip()
    if not url:
        return jsonify({'error': 'No RTSP URL provided'}), 400
    stream_id = uuid.uuid4().hex
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return jsonify({'error': f'Cannot open stream: {url}'}), 400
    with rtsp_lock:
        for s in rtsp_streams.values():
            s['cap'].release()
        rtsp_streams.clear()
        rtsp_streams[stream_id] = {'cap': cap, 'active': True, 'url': url}
    return jsonify({'success': True, 'stream_id': stream_id})


@app.route('/rtsp/frame/<stream_id>', methods=['GET'])
def rtsp_frame(stream_id):
    try:
        import cv2
    except ImportError:
        return jsonify({'error': 'opencv not available'}), 500
    with rtsp_lock:
        stream = rtsp_streams.get(stream_id)
    if not stream or not stream['active']:
        return jsonify({'error': 'Stream not found or closed'}), 404
    ret, frame = stream['cap'].read()
    if not ret:
        return jsonify({'error': 'Failed to read frame'}), 500
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        b64, detections, count = run_detection_on_pil(img)
        return jsonify({'success': True, 'image_b64': b64,
                        'detections': detections, 'count': count})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/rtsp/close/<stream_id>', methods=['POST'])
def rtsp_close(stream_id):
    with rtsp_lock:
        stream = rtsp_streams.pop(stream_id, None)
    if stream:
        stream['cap'].release()
    return jsonify({'success': True})


@app.route('/static/results/<path:filename>')
def result_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
