# 🛡️ SafeChild AI — Bolalar Xavfsizligi Platformasi

Django asosida yaratilgan, bolalar uchun xavfli uy buyumlarini real vaqtda
aniqlash uchun mo'ljallangan sun'iy intellekt veb-platformasi.

## 📋 Sahifalar
- `/` — Bosh sahifa (hero, pipeline, sinflar, statistika)
- `/about/` — Loyiha haqida (pipeline detail, modellar jadvali, sinflar)
- `/examples/` — Namunalar (bounding box vizualizatsiyasi)
- `/results/` — Test natijalari (per-class jadval, Misty robot)
- `/demo/` — Demo (rasm/video yuklash, real prediction)

## 🚀 Ishga Tushirish

### 1. O'rnatish
```bash
pip install django pillow onnxruntime opencv-python numpy
```

### 2. ONNX Model (ixtiyoriy)
```
models_dir/best.onnx   ← YOLO26s ONNX modelini shu yerga joylashtiring
```
Model bo'lmasa — Demo rejim avtomatik ishlaydi.

### 3. Ishga tushirish
```bash
python manage.py runserver
# http://127.0.0.1:8000 da oching
```

## ⚙️ Muhim sozlamalar (config/settings.py)
- `CONFIDENCE_THRESHOLD = 0.35` — aniqlash chegarasi
- `IOU_THRESHOLD = 0.45` — NMS chegarasi
- `MODEL_INPUT_SIZE = 640` — model kirish o'lchami
- `MAX_UPLOAD_SIZE = 50MB` — maksimal fayl hajmi

## 🔑 API Endpoints
- `POST /predict/image/` — rasm prediction (multipart: `image`)
- `POST /predict/video/` — video prediction (multipart: `video`)

## 📁 Loyiha Tuzilmasi
```
dengrous_platform/
├── config/          Django sozlamalari
├── core/            Asosiy app
│   ├── detector.py  ONNX Runtime detektor
│   ├── views.py     Sahifalar va API
│   └── urls.py      URL routing
├── templates/core/  HTML shablonlar
├── static/          CSS, JS
│   ├── css/main.css Dark theme dizayn
│   └── js/main.js   Animatsiyalar, interactions
├── media/           Yuklangan fayllar
│   ├── uploads/     Kiruvchi fayllar
│   └── results/     Prediction natijalari
└── models_dir/      ONNX model joyi
```

## 🎯 Texnologiyalar
- **Backend**: Django 6.0, Python 3.12
- **ML Inference**: ONNX Runtime (Ultralytics ishlatilmaydi)
- **Rasm qayta ishlash**: OpenCV, NumPy
- **Frontend**: Vanilla JS, CSS3 (kutubxonasiz)
