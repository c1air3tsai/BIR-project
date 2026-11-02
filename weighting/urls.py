# Project 3 的網址設定（在 biomedir/urls.py 以 include('weighting.urls') 掛上）
from django.urls import path

from . import views

app_name = 'weighting'

urlpatterns = [
    path('analysis/term-weighting/', views.page, name='page'),
    path('analysis/term-weighting/export/', views.export_csv, name='export'),
    path('analysis/term-weighting/results/<str:name>', views.result_file, name='result_file'),
]
