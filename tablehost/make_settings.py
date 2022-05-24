from .settings import *
import os

LOG_LEVEL_DJANGO = 'ERROR'
LOG_LEVEL_TCP_PROXY = 'ERROR'
LOG_LEVEL_UART = 'ERROR'

UART_INI_FILE = os.environ.get('UART_INI_FILE')
UART_DEFINES_FILE = os.environ.get('UART_META_FILE')

UART_TCP_WRAP = False

LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'detailed': {
            'format': '[%(levelname)s] - %(asctime)s - %(name)s|%(pathname)s:%(lineno)s - %(message)s',
            'style': '%',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
        'detailed_console': {
            'class': 'logging.StreamHandler',
            'formatter': 'detailed',
            'level': 'DEBUG',
        },
    },
    'root': {
        'handlers': [],
        'level': 'WARNING',
    },
    'loggers': {
        'uart_com': {
            'handlers': ['detailed_console'],
            'level': LOG_LEVEL_UART,
            'propagate': True,
        },
        'tcp_proxy': {
            'handlers': ['detailed_console'],
            'level': LOG_LEVEL_TCP_PROXY,
            'propagate': True,
        },
        'django.server': {
            'handlers': [],
            'level': LOG_LEVEL_DJANGO,
            'propagate': True,
        },
        'django.request': {
            'handlers': [],
            'level': LOG_LEVEL_DJANGO,
            'propagate': True,
        },
    },
}
