FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=settings

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "Django>=5.2,<6.0" gunicorn \
    opencv-python-headless numpy pytesseract Pillow

COPY . .

WORKDIR /app/web

CMD ["sh", "-c", "gunicorn --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:${PORT:-10000} 'django.core.wsgi:get_wsgi_application()'"]
