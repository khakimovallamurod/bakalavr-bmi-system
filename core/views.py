import os, json, uuid, time
from pathlib import Path
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np

from .detector import ONNXDetector, get_risk_summary

# Lazy-load detector singleton
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = ONNXDetector(
            model_path=settings.ONNX_MODEL_PATH,
            input_size=settings.MODEL_INPUT_SIZE,
            conf_threshold=settings.CONFIDENCE_THRESHOLD,
            iou_threshold=settings.IOU_THRESHOLD,
        )
    return _detector


def home(request):
    classes = [
        {'name': 'pichoq',    'en': 'Knife',        'danger': 'critical', 'icon': '🔪'},
        {'name': 'qaychi',    'en': 'Scissors',     'danger': 'critical', 'icon': '✂️'},
        {'name': 'sanchqi',   'en': 'Fork',         'danger': 'critical', 'icon': '🍴'},
        {'name': 'bolta',     'en': 'Axe',          'danger': 'high',     'icon': '🪓'},
        {'name': 'bolga',     'en': 'Hammer',       'danger': 'high',     'icon': '🔨'},
        {'name': 'otvyortka', 'en': 'Screwdriver',  'danger': 'high',     'icon': '🔧'},
        {'name': 'qisqich',   'en': 'Pliers',       'danger': 'medium',   'icon': '🔩'},
        {'name': 'stepler',   'en': 'Stapler',      'danger': 'medium',   'icon': '📎'},
        {'name': 'kalit',     'en': 'Key',          'danger': 'low',      'icon': '🔑'},
        {'name': 'tanga',     'en': 'Coin',         'danger': 'low',      'icon': '🪙'},
    ]
    return render(request, 'core/home.html', {'classes': classes})


def about(request):
    classes_info = [
        {'name': 'pichoq', 'en': 'Knife', 'danger': 'critical', 'icon': '🔪', 'count': 1500},
        {'name': 'qaychi', 'en': 'Scissors', 'danger': 'critical', 'icon': '✂️', 'count': 3081},
        {'name': 'sanchqi', 'en': 'Fork', 'danger': 'critical', 'icon': '🍴', 'count': 1967},
        {'name': 'bolta', 'en': 'Axe', 'danger': 'high', 'icon': '🪓', 'count': 2699},
        {'name': 'bolga', 'en': 'Hammer', 'danger': 'high', 'icon': '🔨', 'count': 1868},
        {'name': 'otvyortka', 'en': 'Screwdriver', 'danger': 'high', 'icon': '🔧', 'count': 2218},
        {'name': 'qisqich', 'en': 'Pliers', 'danger': 'medium', 'icon': '🔩', 'count': 1921},
        {'name': 'stepler', 'en': 'Stapler', 'danger': 'medium', 'icon': '📎', 'count': 1435},
        {'name': 'kalit', 'en': 'Key', 'danger': 'low', 'icon': '🔑', 'count': 2108},
        {'name': 'tanga', 'en': 'Coin', 'danger': 'low', 'icon': '🪙', 'count': 3147},
    ]
    models_data = [
        {'name': 'YOLOv8n', 'layers': 73, 'params': '3.01M', 'gflops': 8.1,
         'precision': 0.866, 'recall': 0.763, 'f1': 0.812, 'map50': 0.834, 'map9595': 0.631, 'speed': 4.5, 'color': 'primary'},
        {'name': 'YOLO11s', 'layers': 101, 'params': '9.42M', 'gflops': 21.3,
         'precision': 0.897, 'recall': 0.766, 'f1': 0.826, 'map50': 0.863, 'map9595': 0.673, 'speed': 6.9, 'color': 'warning'},
        {'name': 'YOLO26s', 'layers': 122, 'params': '9.47M', 'gflops': 20.5,
         'precision': 0.884, 'recall': 0.804, 'f1': 0.842, 'map50': 0.876, 'map9595': 0.701, 'speed': 6.6, 'color': 'success'},
    ]
    return render(request, 'core/about.html', {
        'classes_info': classes_info,
        'models_data': models_data,
    })


def examples(request):
    # Generate synthetic demo images with annotations
    demo_data = generate_demo_examples()
    return render(request, 'core/examples.html', {'examples': demo_data})


def results(request):
    per_class = [
        {'name':'Hammer','uz':'bolga','danger':'high',
         'v8p':0.791,'v8r':0.792,'v8m':0.828,'v8m95':0.631,
         'v11p':0.806,'v11r':0.810,'v11m':0.857,'v11m95':0.693,
         'v26p':0.785,'v26r':0.847,'v26m':0.857,'v26m95':0.717},
        {'name':'Knife','uz':'pichoq','danger':'critical',
         'v8p':0.843,'v8r':0.520,'v8m':0.656,'v8m95':0.516,
         'v11p':0.857,'v11r':0.553,'v11m':0.727,'v11m95':0.583,
         'v26p':0.903,'v26r':0.685,'v26m':0.785,'v26m95':0.647},
        {'name':'Screwdriver','uz':'otvyortka','danger':'high',
         'v8p':0.869,'v8r':0.779,'v8m':0.820,'v8m95':0.608,
         'v11p':0.933,'v11r':0.811,'v11m':0.862,'v11m95':0.667,
         'v26p':0.830,'v26r':0.820,'v26m':0.876,'v26m95':0.680},
        {'name':'Key','uz':'kalit','danger':'low',
         'v8p':0.869,'v8r':0.804,'v8m':0.873,'v8m95':0.495,
         'v11p':0.928,'v11r':0.798,'v11m':0.898,'v11m95':0.515,
         'v26p':0.902,'v26r':0.765,'v26m':0.880,'v26m95':0.540},
        {'name':'Pliers','uz':'qisqich','danger':'medium',
         'v8p':0.896,'v8r':0.726,'v8m':0.848,'v8m95':0.664,
         'v11p':0.948,'v11r':0.769,'v11m':0.905,'v11m95':0.737,
         'v26p':0.919,'v26r':0.795,'v26m':0.899,'v26m95':0.738},
        {'name':'Axe','uz':'bolta','danger':'high',
         'v8p':0.956,'v8r':0.806,'v8m':0.898,'v8m95':0.648,
         'v11p':0.954,'v11r':0.772,'v11m':0.930,'v11m95':0.677,
         'v26p':0.958,'v26r':0.829,'v26m':0.934,'v26m95':0.677},
        {'name':'Coin','uz':'tanga','danger':'low',
         'v8p':0.721,'v8r':0.826,'v8m':0.834,'v8m95':0.720,
         'v11p':0.794,'v11r':0.768,'v11m':0.832,'v11m95':0.735,
         'v26p':0.826,'v26r':0.831,'v26m':0.867,'v26m95':0.773},
        {'name':'Fork','uz':'sanchqi','danger':'critical',
         'v8p':0.872,'v8r':0.515,'v8m':0.648,'v8m95':0.370,
         'v11p':0.842,'v11r':0.507,'v11m':0.664,'v11m95':0.410,
         'v26p':0.807,'v26r':0.572,'v26m':0.706,'v26m95':0.465},
        {'name':'Scissors','uz':'qaychi','danger':'critical',
         'v8p':0.913,'v8r':0.926,'v8m':0.961,'v8m95':0.785,
         'v11p':0.934,'v11r':0.907,'v11m':0.964,'v11m95':0.809,
         'v26p':0.942,'v26r':0.923,'v26m':0.966,'v26m95':0.852},
        {'name':'Stapler','uz':'stepler','danger':'medium',
         'v8p':0.924,'v8r':0.939,'v8m':0.973,'v8m95':0.873,
         'v11p':0.977,'v11r':0.960,'v11m':0.993,'v11m95':0.906,
         'v26p':0.965,'v26r':0.977,'v26m':0.992,'v26m95':0.919},
    ]
    return render(request, 'core/results.html', {'per_class': per_class})


def demo(request):
    model_available = get_detector().available
    return render(request, 'core/demo.html', {
        'model_available': model_available,
        'model_status': 'Faol' if model_available else 'Demo rejimi (model yuklanmagan)',
    })


@csrf_exempt
def predict_image(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    if 'image' not in request.FILES:
        return JsonResponse({'error': 'Rasm yuklanmadi'}, status=400)

    file = request.FILES['image']
    if file.size > settings.MAX_UPLOAD_SIZE:
        return JsonResponse({'error': 'Fayl hajmi 50MB dan oshmasligi kerak'}, status=400)

    try:
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return JsonResponse({'error': 'Rasm o\'qilmadi'}, status=400)

        start = time.time()
        detector = get_detector()
        detections, annotated = detector.detect(img)
        elapsed = round((time.time() - start) * 1000, 1)

        # Save results
        uid = str(uuid.uuid4())[:8]
        orig_path = f'results/orig_{uid}.jpg'
        pred_path = f'results/pred_{uid}.jpg'

        cv2.imwrite(str(settings.MEDIA_ROOT / orig_path), img)
        cv2.imwrite(str(settings.MEDIA_ROOT / pred_path), annotated)

        risk = get_risk_summary(detections)

        return JsonResponse({
            'success': True,
            'original_url': settings.MEDIA_URL + orig_path,
            'predicted_url': settings.MEDIA_URL + pred_path,
            'detections': detections,
            'risk': risk,
            'inference_ms': elapsed,
            'image_size': f"{img.shape[1]}x{img.shape[0]}",
            'demo_mode': not detector.available,
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def predict_video(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    if 'video' not in request.FILES:
        return JsonResponse({'error': 'Video yuklanmadi'}, status=400)

    file = request.FILES['video']
    if file.size > settings.MAX_UPLOAD_SIZE:
        return JsonResponse({'error': 'Fayl hajmi 50MB dan oshmasligi kerak'}, status=400)

    try:
        uid = str(uuid.uuid4())[:8]
        input_path = settings.MEDIA_ROOT / f'uploads/video_{uid}{Path(file.name).suffix}'
        output_path = settings.MEDIA_ROOT / f'results/video_{uid}_pred.mp4'

        with open(input_path, 'wb') as f:
            for chunk in file.chunks():
                f.write(chunk)

        # Process video frames
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        process_frames = min(total_frames, 150)  # max 5s at 30fps

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        detector = get_detector()
        all_detections = []
        frame_count = 0

        while frame_count < process_frames:
            ret, frame = cap.read()
            if not ret:
                break
            dets, annotated = detector.detect(frame)
            all_detections.extend(dets)
            out.write(annotated)
            frame_count += 1

        cap.release()
        out.release()

        # Unique objects seen
        seen = {}
        for d in all_detections:
            name = d['class_name']
            if name not in seen or d['confidence'] > seen[name]['confidence']:
                seen[name] = d

        unique_dets = list(seen.values())
        risk = get_risk_summary(unique_dets)

        return JsonResponse({
            'success': True,
            'video_url': settings.MEDIA_URL + f'results/video_{uid}_pred.mp4',
            'frames_processed': frame_count,
            'unique_detections': unique_dets,
            'risk': risk,
            'demo_mode': not detector.available,
        })

    except Exception as e:
        import traceback
        return JsonResponse({'error': str(e), 'trace': traceback.format_exc()}, status=500)


def generate_demo_examples():
    """Generate demo examples with synthetic bounding boxes."""
    import random
    examples = [
        {'id': 1, 'title': 'Pichoq aniqlash', 'en': 'Knife Detection',
         'objects': [{'name': 'Knife', 'conf': 92.3, 'danger': 'critical', 'color': '#dc3545'}],
         'description': 'Oshxonada pichoq aniqlanib, kritik xavf belgisi ko\'rsatildi.',
         'bg_color': '#fff3cd', 'icon': '🔪', 'risk': 'critical'},
        {'id': 2, 'title': 'Qaychi aniqlash', 'en': 'Scissors Detection',
         'objects': [{'name': 'Scissors', 'conf': 88.7, 'danger': 'critical', 'color': '#dc3545'}],
         'description': 'O\'yin xonasida qaychi topildi.',
         'bg_color': '#f8d7da', 'icon': '✂️', 'risk': 'critical'},
        {'id': 3, 'title': 'Ko\'p obyekt', 'en': 'Multiple Objects',
         'objects': [
             {'name': 'Hammer', 'conf': 85.4, 'danger': 'high', 'color': '#fd7e14'},
             {'name': 'Screwdriver', 'conf': 79.2, 'danger': 'high', 'color': '#fd7e14'},
         ],
         'description': 'Ustaxonada bir nechta xavfli asboblar aniqlandi.',
         'bg_color': '#ffe5d0', 'icon': '🔨', 'risk': 'high'},
        {'id': 4, 'title': 'Sanchqi', 'en': 'Fork Detection',
         'objects': [{'name': 'Fork', 'conf': 76.8, 'danger': 'critical', 'color': '#dc3545'}],
         'description': 'Stol ustida sanchqi topildi.',
         'bg_color': '#f8d7da', 'icon': '🍴', 'risk': 'critical'},
        {'id': 5, 'title': 'Kalit (past xavf)', 'en': 'Key — Low Risk',
         'objects': [{'name': 'Key', 'conf': 94.1, 'danger': 'low', 'color': '#28a745'}],
         'description': 'Kalit aniqlandi — past xavf darajasi.',
         'bg_color': '#d4edda', 'icon': '🔑', 'risk': 'low'},
        {'id': 6, 'title': 'Bolta', 'en': 'Axe Detection',
         'objects': [{'name': 'Axe', 'conf': 91.0, 'danger': 'high', 'color': '#fd7e14'}],
         'description': 'Bolalar xonasida bolta aniqlandi — yuqori xavf.',
         'bg_color': '#ffe5d0', 'icon': '🪓', 'risk': 'high'},
    ]
    return examples
