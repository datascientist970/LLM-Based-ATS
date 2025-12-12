from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='home'),
    path('upload/', views.upload_page, name='upload'),
    path('analyze/', views.analyze, name='analyze'),
]
