# Project 4 的網址設定（在 biomedir/urls.py 以 include('reuters.urls') 掛上）
#   /reuters/                          主選單的「Reuters」頁面
#   /reuters/results/<名稱>            實驗指令產生的 PNG 圖（三種切分共用的圖）
#   /reuters/results/<切分>/<名稱>     某個切分的 PNG 圖（modapte / modlewis / modhayes）
#   /reuters/news/                     新聞列表（依 TOPICS 主題瀏覽 Reuters-21578）
#   /reuters/news/<NEWID>/             新聞內頁（全文、主題、所屬切分、查詢詞高亮、各方法排名）
#   /analysis/reuters/                 舊網址（之前放在分析子選單），自動轉到 /reuters/
from django.urls import path
from django.views.generic import RedirectView

from . import views

app_name = 'reuters'

urlpatterns = [
    path('reuters/', views.page, name='page'),
    path('reuters/results/<str:name>', views.result_file, name='result_file'),
    path('reuters/results/<str:split>/<str:name>', views.result_file, name='split_result'),
    path('reuters/news/', views.news_list, name='news_list'),
    path('reuters/news/<int:newid>/', views.news_detail, name='news'),
    path('analysis/reuters/', RedirectView.as_view(pattern_name='reuters:page', query_string=True, permanent=False)),
]
