import os
import django
from .settings import *


INSTALLED_APPS += ['pytest_django']

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': ':memory:',
    }
}

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cs_app.test_settings")
django.setup()

from django.core.management import call_command