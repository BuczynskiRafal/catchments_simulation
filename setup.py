from setuptools import setup, find_packages

# python setup.py sdist
setup(
    name="catchment_simulation",
    version="0.0.1",
    description="Collection of calculation method for simulate subcatchment features from Storm Water Management Model",
    author="Rafał Buczyński",
    packages=["catchment_features_simulation"],
    install_requires=["pandas", "numpy", "swmmio", "pyswmm"],
)