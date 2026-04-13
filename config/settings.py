import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = 'django-dengrous-child-safety-2025-secret-key-xyz'
DEBUG = True
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.staticfiles',
    'core',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.template.context_processors.static',
                'django.template.context_processors.media',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'
LANGUAGE_CODE = 'uz'
TIME_ZONE = 'Asia/Tashkent'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

ONNX_MODEL_PATH = BASE_DIR / 'models' / 'yolo26s_model.onnx'
MODEL_INPUT_SIZE = 640
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

CLASS_NAMES = ['bolga','pichoq','otvyortka','kalit','qisqich','bolta','tanga','sanchqi','qaychi','stepler']
CLASS_NAMES_EN = ['Hammer','Knife','Screwdriver','Key','Pliers','Axe','Coin','Fork','Scissors','Stapler']

DANGER_LEVELS = {
    'bolga':'high','pichoq':'critical','otvyortka':'high','kalit':'low',
    'qisqich':'medium','bolta':'high','tanga':'low','sanchqi':'critical',
    'qaychi':'critical','stepler':'medium',
}
MAX_UPLOAD_SIZE = 50 * 1024 * 1024
