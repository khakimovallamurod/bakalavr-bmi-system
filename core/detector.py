"""
ONNX Runtime based hazardous object detector.
Works without Ultralytics — pure ONNX Runtime + OpenCV.
"""
import os
import cv2
import numpy as np
from pathlib import Path


CLASS_NAMES = ['bolga','pichoq','otvyortka','kalit','qisqich','bolta','tanga','sanchqi','qaychi','stepler']
CLASS_NAMES_EN = ['Hammer','Knife','Screwdriver','Key','Pliers','Axe','Coin','Fork','Scissors','Stapler']

DANGER_LEVELS = {
    'bolga':'high','pichoq':'critical','otvyortka':'high','kalit':'low',
    'qisqich':'medium','bolta':'high','tanga':'low','sanchqi':'critical',
    'qaychi':'critical','stepler':'medium',
}

DANGER_COLORS = {
    'critical': (0, 0, 220),    # Red BGR
    'high':     (0, 100, 255),  # Orange
    'medium':   (0, 200, 255),  # Yellow
    'low':      (0, 200, 50),   # Green
}

DANGER_LABELS = {
    'critical': '🔴 KRITIK',
    'high':     '🟠 YUQORI',
    'medium':   '🟡 O\'RTA',
    'low':      '🟢 PAST',
}

DANGER_LABELS_EN = {
    'critical': 'CRITICAL',
    'high':     'HIGH',
    'medium':   'MEDIUM',
    'low':      'LOW',
}


def letterbox(img, new_shape=640):
    """Resize image with unchanged aspect ratio using padding."""
    shape = img.shape[:2]
    r = min(new_shape / shape[0], new_shape / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape - new_unpad[0]
    dh = new_shape - new_unpad[1]
    dw /= 2; dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right  = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    return img, r, (dw, dh)


def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def nms(boxes, scores, iou_threshold=0.45):
    """Simple NMS."""
    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


class ONNXDetector:
    def __init__(self, model_path, input_size=640, conf_threshold=0.35, iou_threshold=0.45):
        self.model_path = str(model_path)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.session = None
        self.available = os.path.exists(self.model_path)
        if self.available:
            self._load_model()

    def _load_model(self):
        try:
            import onnxruntime as ort
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            print(f"[ONNX] Model loaded: {self.model_path}")
        except Exception as e:
            print(f"[ONNX] Load error: {e}")
            self.available = False

    def preprocess(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_lb, ratio, pad = letterbox(img_rgb, self.input_size)
        img_arr = img_lb.astype(np.float32) / 255.0
        img_arr = np.transpose(img_arr, (2, 0, 1))
        img_arr = np.expand_dims(img_arr, 0)
        return img_arr, ratio, pad

    def postprocess(self, output, orig_shape, ratio, pad):
        """Parse YOLO output [1, 14, 8400] or [1, 8400, 14]."""
        pred = output[0]
        if pred.ndim == 3:
            pred = pred[0]  # [14, 8400] or [8400, 14]
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T   # now [8400, 14]

        boxes_xywh = pred[:, :4]
        scores_all = pred[:, 4:]

        class_ids = np.argmax(scores_all, axis=1)
        confidences = scores_all[np.arange(len(class_ids)), class_ids]

        mask = confidences >= self.conf_threshold
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        boxes_xyxy = xywh2xyxy(boxes_xywh)

        # Scale back to original image
        dw, dh = pad
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - dw) / ratio
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - dh) / ratio

        h, w = orig_shape[:2]
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h)

        keep = nms(boxes_xyxy, confidences, self.iou_threshold)
        results = []
        for i in keep:
            cid = int(class_ids[i])
            name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f'class_{cid}'
            name_en = CLASS_NAMES_EN[cid] if cid < len(CLASS_NAMES_EN) else f'class_{cid}'
            danger = DANGER_LEVELS.get(name, 'low')
            x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
            results.append({
                'class_id': cid,
                'class_name': name,
                'class_name_en': name_en,
                'confidence': float(confidences[i]),
                'confidence_pct': round(float(confidences[i]) * 100, 1),
                'bbox': [x1, y1, x2, y2],
                'danger_level': danger,
                'danger_label': DANGER_LABELS.get(danger, ''),
                'danger_label_en': DANGER_LABELS_EN.get(danger, ''),
            })
        return results

    def detect(self, img_bgr):
        """Run detection. Returns (detections, annotated_img)."""
        if not self.available or self.session is None:
            return self._demo_detect(img_bgr)

        orig_shape = img_bgr.shape
        inp, ratio, pad = self.preprocess(img_bgr)
        outputs = self.session.run(None, {self.input_name: inp})
        detections = self.postprocess(outputs, orig_shape, ratio, pad)
        annotated = self.draw_boxes(img_bgr.copy(), detections)
        return detections, annotated

    def _demo_detect(self, img_bgr):
        """Demo mode — simulate detections when model not available."""
        h, w = img_bgr.shape[:2]
        # Simulate 1-2 detections
        demo_dets = [
            {
                'class_id': 1, 'class_name': 'pichoq', 'class_name_en': 'Knife',
                'confidence': 0.87, 'confidence_pct': 87.0,
                'bbox': [int(w*0.2), int(h*0.2), int(w*0.5), int(h*0.7)],
                'danger_level': 'critical',
                'danger_label': '🔴 KRITIK', 'danger_label_en': 'CRITICAL',
            },
            {
                'class_id': 8, 'class_name': 'qaychi', 'class_name_en': 'Scissors',
                'confidence': 0.73, 'confidence_pct': 73.0,
                'bbox': [int(w*0.55), int(h*0.3), int(w*0.85), int(h*0.75)],
                'danger_level': 'critical',
                'danger_label': '🔴 KRITIK', 'danger_label_en': 'CRITICAL',
            },
        ]
        annotated = self.draw_boxes(img_bgr.copy(), demo_dets)
        return demo_dets, annotated

    def draw_boxes(self, img, detections):
        """Draw bounding boxes with labels."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            danger = det['danger_level']
            color = DANGER_COLORS.get(danger, (0, 200, 50))
            # Box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            # Label background
            label = f"{det['class_name_en']} {det['confidence_pct']}%"
            (lw, lh), baseline = cv2.getTextSize(label, font, 0.65, 2)
            cv2.rectangle(img, (x1, y1 - lh - baseline - 8), (x1 + lw + 6, y1), color, -1)
            # Label text
            cv2.putText(img, label, (x1 + 3, y1 - baseline - 3), font, 0.65, (255,255,255), 2)
            # Danger badge
            badge = DANGER_LABELS_EN.get(danger, '')
            cv2.putText(img, badge, (x1 + 3, y1 + 22), font, 0.55, color, 2)
        return img


def get_risk_summary(detections):
    """Analyze detections and return risk summary."""
    if not detections:
        return {
            'has_danger': False,
            'risk_level': 'safe',
            'risk_label': '✅ XAVF YO\'Q',
            'risk_label_en': 'SAFE',
            'risk_color': 'success',
            'summary_uz': 'Hech qanday xavfli obyekt aniqlanmadi.',
            'summary_en': 'No hazardous objects detected.',
            'detected_count': 0,
            'critical_count': 0,
            'objects': [],
        }

    levels = [d['danger_level'] for d in detections]
    if 'critical' in levels:
        risk = 'critical'
        label = '🔴 KRITIK XAVF'
        label_en = 'CRITICAL DANGER'
        color = 'danger'
    elif 'high' in levels:
        risk = 'high'
        label = '🟠 YUQORI XAVF'
        label_en = 'HIGH DANGER'
        color = 'warning'
    elif 'medium' in levels:
        risk = 'medium'
        label = '🟡 O\'RTA XAVF'
        label_en = 'MODERATE RISK'
        color = 'secondary'
    else:
        risk = 'low'
        label = '🟢 PAST XAVF'
        label_en = 'LOW RISK'
        color = 'info'

    critical_count = sum(1 for l in levels if l == 'critical')
    names = [f"{d['class_name_en']} ({d['confidence_pct']}%)" for d in detections]

    return {
        'has_danger': True,
        'risk_level': risk,
        'risk_label': label,
        'risk_label_en': label_en,
        'risk_color': color,
        'summary_uz': f"{len(detections)} ta xavfli obyekt aniqlandi: {', '.join(names)}",
        'summary_en': f"{len(detections)} hazardous object(s) detected: {', '.join(names)}",
        'detected_count': len(detections),
        'critical_count': critical_count,
        'objects': detections,
    }
