import os
import sys
import subprocess
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


def convert_readme_md_to_rst(app, docname, source):
    if docname == 'index':
        readme_path = os.path.join(app.srcdir, '..', 'README.md')
        rst_output = subprocess.check_output(['pandoc', '-f', 'markdown', '-t', 'rst', readme_path])
        source[0] = rst_output.decode('utf-8')

def setup(app):
    app.connect('source-read', convert_readme_md_to_rst)



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
