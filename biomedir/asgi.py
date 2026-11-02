# biomedir/asgi.py
# ASGI 伺服器入口（目前用 runserver，不會用到；保留給正式部署）。

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'biomedir.settings')

application = get_asgi_application()
