"""
Django settings for tablehost project.

For more information on this file, see
https://docs.djangoproject.com/en/1.7/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/1.7/ref/settings/
"""

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/1.7/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'w0*myd_!iqn&%%fm+cx)uxm@odo_74wu=q=-+6t3iah3tzruj2'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []


# Application definition

INSTALLED_APPS = (
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'apps.main',
    'apps.player',
    'apps.table',
    'libs.jinja'
)

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'tablehost.urls'

WSGI_APPLICATION = 'tablehost.wsgi.application'


# Database
# https://docs.djangoproject.com/en/1.7/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}

# Internationalization
# https://docs.djangoproject.com/en/1.7/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/1.7/howto/static-files/

STATIC_URL = '/static/'

STATICFILES_DIRS = (
    os.path.join(BASE_DIR, "static"),
)

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'NAME': 'djangotemplates',
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
            'loaders': [
                'django.template.loaders.filesystem.Loader',
                'django.template.loaders.app_directories.Loader',
            ],
        }
    },
]

JINJA_TEMPLATE_DIRS = [
    'templates',
]

MC_WS2812_CYCLES = 56933

DEFAULT_AUTO_FIELD = 'django.db.models.AutoField'

SIM_EDGE_LENGTH = None
SIM_DOWNWARDS = None
SIM_RIGHTWARDS = None
SIM_HORIZONTAL = None
SIM_DRAW_CONNECTORS = None
SIM_FPS = None
SIM_PARAM_STRUCT = 'III????I'
SIM_LED_STRUCT = f'24h?'

GIT_REMOTE_NAME = None
GIT_REMOTE_ACTIVE_BRANCH = None
LOCAL_SERIAL_CLASS = 'tablehost.uart.PatchedSerial'
UART_SIM_PORT = None
UART_PORT = 'loop://'
UART_BAUD_RATE = 500000
UART_TCP_PORT = 7777
UART_CLIENT_CONNECTION = None
UART_TCP_WRAP = False
UART_N_ECHO_TEST = 2000
UART_N_ECHO_TEST_BYTES = 20

REMOTE_SERIAL_HOST_ADDR = 'sheepdroid'

try:
    from .local_settings import *
except ImportError:
    pass
