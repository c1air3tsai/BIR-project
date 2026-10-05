#!/bin/sh
set -eu
cd /app

if [ "${1:-}" = "python" ] && [ "${2:-}" = "manage.py" ] && [ "${3:-}" = "runserver" ]; then
    mkdir -p /app/data/corpus /app/data/logs /app/data/fetched
    python manage.py migrate --noinput
    python manage.py prepare_container
fi

exec "$@"
