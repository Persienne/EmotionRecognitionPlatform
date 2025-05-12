from django.urls import path, include
from .views import upload_file

urlpatterns = [
    path('res/', upload_file),
]
