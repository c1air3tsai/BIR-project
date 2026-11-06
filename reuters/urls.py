# Project 4 的網址設定（在 biomedir/urls.py 以 include('reuters.urls') 掛上）
#   /reuters/                  主選單的「Reuters」頁面
#   /reuters/results/<名稱>    實驗指令產生的 PNG 圖
#   /reuters/news/<NEWID>/     新聞內頁（全文、查詢詞高亮、各方法排名）
#   /analysis/reuters/         舊網址（之前放在分析子選單），自動轉到 /reuters/
from django.urls import path
from django.views.generic import RedirectView

from . import views

app_name = 'reuters'

urlpatterns = [
    path('reuters/', views.page, name='page'),
    path('reuters/results/<str:name>', views.result_file, name='result_file'),
    path('reuters/news/<int:newid>/', views.news_detail, name='news'),
    path('analysis/reuters/', RedirectView.as_view(pattern_name='reuters:page', query_string=True, permanent=False)),
]
