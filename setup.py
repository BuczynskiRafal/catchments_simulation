from setuptools import setup, find_packages
import codecs
import os
# python setup.py sdist
here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.6'
DESCRIPTION = "Collection of calculation method for simulate subcatchment features from Storm Water Management Model",
LONG_DESCRIPTION = "Collection of calculation method for simulate subcatchment features from Storm Water Management Model",

# Setting up
setup(
    name="catchment_simulation",
    version=VERSION,
    author="Rafał Buczyński",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=["pandas", "numpy", "swmmio", "pyswmm"],

    project_urls = {
      'Homepage': 'https://github.com/BuczynskiRafal/catchments_simulation',
    }
)