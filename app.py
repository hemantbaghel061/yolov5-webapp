import os
import io
import uuid
import threading
import base64
import pathlib

from flask import Flask, request, jsonify, send_from_directory, render_template_string
from PIL import Image

pathlib.WindowsPath = pathlib.PosixPath

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

rtsp_streams = {}
rtsp_lock = threading.Lock()

print("Loading YOLOv5s model...")
from ultralytics import YOLO
model = YOLO('yolov5s.pt')
print("Model loaded OK")


def run_detection_on_pil(img):
    img_rgb = img.convert('RGB')
    results = model.predict(source=img_rgb, conf=0.15, iou=0.45, imgsz=640, verbose=False)
    result_img = Image.fromarray(results[0].plot())
    buf = io.BytesIO()
    result_img.save(buf, format='JPEG', quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    detections = []
    for box in results[0].boxes:
        label = model.names[int(box.cls)]
        conf = round(float(box.conf) * 100, 1)
        detections.append({'label': label, 'confidence': conf})
    return f"data:image/jpeg;base64,{b64}", detections, len(detections)


@app.route('/')
def index():
    html = open("templates/index.html").read()
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
        return jsonify({'error': 'opencv not installed'}), 500
    data = request.get_json(force=True, silent=True) or {}
    url = (data.get('url') or '').strip()
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    stream_id = uuid.uuid4().hex
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return jsonify({'error': f'Cannot open stream: {url}'}), 400
    with rtsp_lock:
        for s in rtsp_streams.values():
            s['cap'].release()
        rtsp_streams.clear()
        rtsp_streams[stream_id] = {'cap': cap, 'active': True}
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
        return jsonify({'error': 'Stream not found'}), 404
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


@app.route('/test_detect', methods=['GET'])
def test_detect():
    try:
        import numpy as np
        # Create a simple test image (640x640 white image)
        test_img = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8) + 200)
        results = model.predict(source=test_img, conf=0.15, iou=0.45, imgsz=640, verbose=True)
        boxes = results[0].boxes
        return jsonify({
            'success': True,
            'num_detections': len(boxes),
            'model_names': model.names,
            'results_type': str(type(results[0])),
            'boxes_type': str(type(boxes)),
            'raw_boxes': str(boxes)
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})


@app.route('/static/results/<path:filename>')
def result_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
