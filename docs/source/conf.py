import os
import sys
import django
from recommonmark.parser import CommonMarkParser


sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../cs_app'))  # Dodaj tę linię
os.environ['DJANGO_SETTINGS_MODULE'] = 'cs_app.settings'
django.setup()

source_parsers = {
    '.md': CommonMarkParser,
}
source_suffix = ['.rst', '.md']


project = 'Catchment Simulation'
copyright = '2023, Rafał Buczyński'
author = 'Rafał Buczyński'

extensions = [    
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    ]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
