FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BIOMEDIR_DB=/app/data/db.sqlite3 \
    BIOMEDIR_CA_DIR=/app/certs \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Include roots already trusted by Windows before pip downloads dependencies.
COPY certs/ /usr/local/share/ca-certificates/biomedir/
RUN for cert in /usr/local/share/ca-certificates/biomedir/*.pem; do \
      [ -f "$cert" ] || continue; cp "$cert" "${cert%.pem}.crt"; \
    done && update-ca-certificates

COPY requirements.txt /app/
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . /app/
RUN sed -i 's/\r$//' /app/docker-entrypoint.sh \
    && chmod +x /app/docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000", "--noreload"]
