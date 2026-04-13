from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('examples/', views.examples, name='examples'),
    path('results/', views.results, name='results'),
    path('demo/', views.demo, name='demo'),
    # AJAX endpoints
    path('predict/image/', views.predict_image, name='predict_image'),
    path('predict/video/', views.predict_video, name='predict_video'),
]
