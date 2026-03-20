import os
import time
import base64
from io import BytesIO
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─── Detection Configuration ───────────────────────────────────────────────────
CONF_THRESHOLD = 0.35   # Minimum confidence to keep a detection
IOU_THRESHOLD  = 0.45   # IoU threshold for Non-Maximum Suppression
MODEL_SIZE     = 'yolov8n.pt'  # nano (fast). Options: yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# Path to custom trained model (created by train.py)
CUSTOM_MODEL_PATH = os.path.join('runs', 'detect', 'vehicle_model', 'weights', 'best.pt')

# ─── COCO class IDs that correspond to vehicles ────────────────────────────────
# COCO IDs: 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
COCO_VEHICLE_IDS = {1, 2, 3, 5, 7}

# Map COCO class names → our custom vehicle types
COCO_TO_VEHICLE = {
    'bicycle':    'bicycle',
    'car':        'car',
    'motorcycle': 'motorcycle',
    'bus':        'bus',
    'truck':      'truck',
}

VEHICLE_CLASSES = {
    'car':        {'color': '#00D4FF', 'icon': '🚗'},
    'truck':      {'color': '#FF6B35', 'icon': '🚛'},
    'bus':        {'color': '#FFE135', 'icon': '🚌'},
    'motorcycle': {'color': '#A8FF3E', 'icon': '🏍️'},
    'bicycle':    {'color': '#FF3E9A', 'icon': '🚲'},
    'van':        {'color': '#B44FFF', 'icon': '🚐'},
    'suv':        {'color': '#3EFFDC', 'icon': '🚙'},
    'ambulance':  {'color': '#FF3333', 'icon': '🚑'},
}

# ─── Load YOLOv8 Model (runs once at startup) ─────────────────────────────────
# Auto-detect custom trained model; fall back to pretrained
is_custom_model = False
if os.path.exists(CUSTOM_MODEL_PATH):
    model_path = CUSTOM_MODEL_PATH
    is_custom_model = True
    print(f"[VehicleAI] Custom trained model found at {CUSTOM_MODEL_PATH}")
else:
    model_path = MODEL_SIZE
    print(f"[VehicleAI] No custom model found — using pretrained {MODEL_SIZE}")
    print(f"[VehicleAI] Tip: Run 'python train.py' to train a custom model for better accuracy!")

print(f"[VehicleAI] Loading model: {model_path}...")
model = YOLO(model_path)
num_classes = len(model.names)
print(f"[VehicleAI] Model loaded — {num_classes} classes | {'CUSTOM' if is_custom_model else 'PRETRAINED'}")


def classify_vehicle_subtype(coco_name, bbox):
    """
    Refine COCO classes into finer vehicle subtypes using bounding box heuristics.
    - 'car' boxes that are tall & wide → 'suv', narrow & tall → 'van'
    - 'truck' boxes that are small → 'ambulance' (rough heuristic)
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    aspect = w / max(h, 1)
    area = w * h

    if coco_name == 'car':
        if aspect > 1.6 and h > 100:
            return 'suv'
        if aspect < 0.85:
            return 'van'
        return 'car'

    if coco_name == 'truck':
        if area < 15000:
            return 'ambulance'
        return 'truck'

    return COCO_TO_VEHICLE.get(coco_name, 'car')


def detect_vehicles(img):
    """
    Run YOLOv8 inference on a PIL image.
    Returns a list of detection dicts matching the frontend format.
    Handles both pretrained (COCO) and custom-trained models.
    """
    # Run inference
    results = model(img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

    detections = []
    if not results or len(results) == 0:
        return detections

    result = results[0]  # single image → single result

    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id].lower()
        confidence = round(float(box.conf[0]), 2)
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

        if is_custom_model:
            # Custom model: all classes are vehicles, use name directly
            vehicle_type = class_name if class_name in VEHICLE_CLASSES else 'car'
        else:
            # Pretrained COCO model: filter to vehicle classes only
            if cls_id not in COCO_VEHICLE_IDS:
                continue
            # Refine into vehicle subtype using heuristics
            vehicle_type = classify_vehicle_subtype(class_name, [x1, y1, x2, y2])

        meta = VEHICLE_CLASSES.get(vehicle_type, VEHICLE_CLASSES['car'])

        detections.append({
            'class': vehicle_type,
            'confidence': confidence,
            'bbox': [x1, y1, x2, y2],
            'color': meta['color'],
            'icon': meta['icon'],
        })

    # Sort by confidence descending
    detections.sort(key=lambda d: d['confidence'], reverse=True)
    return detections


def draw_detections_on_image(img, detections):
    """Draw bounding boxes and labels on the image with improved styling."""
    draw = ImageDraw.Draw(img, 'RGBA')

    # Try loading a decent font; fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_small = font

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        color = det['color']
        conf_pct = int(det['confidence'] * 100)
        label = f"{det['class'].upper()} {conf_pct}%"

        # Parse hex color for RGBA operations
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)

        # Semi-transparent fill inside the box
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, 30))
        img = Image.alpha_composite(img.convert('RGBA'), overlay)
        draw = ImageDraw.Draw(img, 'RGBA')

        # Draw bounding box (4px thick)
        for t in range(4):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)

        # Corner accents (short lines at each corner for a modern look)
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        for t in range(2):
            # Top-left
            draw.line([(x1 - t, y1), (x1 + corner_len, y1)], fill=color, width=3)
            draw.line([(x1, y1 - t), (x1, y1 + corner_len)], fill=color, width=3)
            # Top-right
            draw.line([(x2 + t, y1), (x2 - corner_len, y1)], fill=color, width=3)
            draw.line([(x2, y1 - t), (x2, y1 + corner_len)], fill=color, width=3)
            # Bottom-left
            draw.line([(x1 - t, y2), (x1 + corner_len, y2)], fill=color, width=3)
            draw.line([(x1, y2 + t), (x1, y2 - corner_len)], fill=color, width=3)
            # Bottom-right
            draw.line([(x2 + t, y2), (x2 - corner_len, y2)], fill=color, width=3)
            draw.line([(x2, y2 + t), (x2, y2 - corner_len)], fill=color, width=3)

        # Label background
        bbox_label = draw.textbbox((0, 0), label, font=font)
        label_w = bbox_label[2] - bbox_label[0] + 14
        label_h = bbox_label[3] - bbox_label[1] + 10
        label_y = max(y1 - label_h - 2, 0)
        draw.rectangle([x1, label_y, x1 + label_w, label_y + label_h], fill=(r, g, b, 220))
        draw.text((x1 + 7, label_y + 4), label, fill='black', font=font)

    return img.convert('RGB')


def image_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=92)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def compute_stats(detections):
    """Compute summary statistics."""
    counts = {}
    total_conf = 0
    for d in detections:
        counts[d['class']] = counts.get(d['class'], 0) + 1
        total_conf += d['confidence']

    avg_conf = round(total_conf / len(detections) * 100, 1) if detections else 0
    dominant = max(counts, key=counts.get) if counts else None

    return {
        'total': len(detections),
        'counts': counts,
        'avg_confidence': avg_conf,
        'dominant_type': dominant,
        'unique_types': len(counts),
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    allowed = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
        orig_w, orig_h = img.size

        # Resize for display (max 900px wide)
        max_w = 900
        if orig_w > max_w:
            scale = max_w / orig_w
            img = img.resize((max_w, int(orig_h * scale)), Image.LANCZOS)

        w, h = img.size

        # ─── Real YOLOv8 Inference ──────────────────────────────────────
        start = time.perf_counter()
        detections = detect_vehicles(img)
        elapsed = round((time.perf_counter() - start) * 1000, 1)

        # Draw on copy
        annotated = img.copy()
        annotated = draw_detections_on_image(annotated, detections)

        # Encode both images
        orig_b64 = image_to_base64(img)
        ann_b64 = image_to_base64(annotated)

        stats = compute_stats(detections)

        return jsonify({
            'success': True,
            'original_image': orig_b64,
            'annotated_image': ann_b64,
            'detections': detections,
            'stats': stats,
            'inference_time': elapsed,
            'image_size': {'width': w, 'height': h, 'original_width': orig_w, 'original_height': orig_h},
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model-info')
def model_info():
    """Return info about the currently loaded model."""
    return jsonify({
        'model_type': 'custom' if is_custom_model else 'pretrained',
        'model_path': model_path,
        'num_classes': num_classes,
        'class_names': list(model.names.values()),
        'conf_threshold': CONF_THRESHOLD,
        'iou_threshold': IOU_THRESHOLD,
    })


@app.route('/static/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
