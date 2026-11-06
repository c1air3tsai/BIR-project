# Django app 設定檔：讓 Django 認得 reuters 這個 app（在 biomedir/settings.py 的 INSTALLED_APPS 註冊）
from django.apps import AppConfig


class ReutersConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'reuters'
    verbose_name = 'Project 4 · Reuters embedding retrieval'
