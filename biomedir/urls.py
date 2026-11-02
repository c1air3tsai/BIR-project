# 網站最上層的網址設定
#   search/     HW1 + Project 2 的所有頁面（搜尋、文章、Zipf、Word2Vec、文字比對…）
#   weighting/  Project 3 Term Weighting 頁面（/analysis/term-weighting/）
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('weighting.urls')),
    path('', include('search.urls')),
]
