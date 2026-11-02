# Django app 設定檔：讓 Django 認得 weighting 這個 app（在 biomedir/settings.py 的 INSTALLED_APPS 註冊）
from django.apps import AppConfig


class WeightingConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'weighting'
    verbose_name = 'Project 3 · Term Weighting'
