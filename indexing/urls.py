# Project 5 的網址設定（在 biomedir/urls.py 以 include('indexing.urls') 掛上）
#   /indexing/                  主選單的「Indexing」頁面
#   /indexing/doc/<docID>/      點 docID 開的彈出視窗內容（JSON：全文，查詢詞標螢光筆）
#   /indexing/results/<名稱>    run_indexing_experiments 產生的 PNG 圖
from django.urls import path

from . import views

app_name = 'indexing'

urlpatterns = [
    path('indexing/', views.page, name='page'),
    path('indexing/doc/<int:doc_id>/', views.document, name='document'),
    path('indexing/results/<str:name>', views.result_file, name='result_file'),
]
