from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
SECRET_KEY = 'local-demo-change-before-deployment'
DEBUG = True  # local development only
ALLOWED_HOSTS = ['127.0.0.1', 'localhost']
ROOT_URLCONF = 'urls'
INSTALLED_APPS = []
MIDDLEWARE = ['django.middleware.security.SecurityMiddleware', 'django.middleware.common.CommonMiddleware', 'django.middleware.csrf.CsrfViewMiddleware']
TEMPLATES = [{'BACKEND':'django.template.backends.django.DjangoTemplates','DIRS':[BASE_DIR/'templates'],'APP_DIRS':True,'OPTIONS':{'context_processors':[]}}]
STATIC_URL = 'static/'
DEFAULT_CHARSET = 'utf-8'
import os

SECRET_KEY = os.environ["DJANGO_SECRET_KEY"]
DEBUG = False

ALLOWED_HOSTS = [
    "localhost",
    "127.0.0.1",
    ".onrender.com",
]

SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
