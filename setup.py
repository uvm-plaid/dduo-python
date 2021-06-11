# setup.py
# python3 setup.py install
from setuptools import setup

setup(name="duet",
      version="0.1",
      install_requires=[
        'numpy',
        'sklearn',
        'pandas',
        'diffprivlib',
        'scikit-learn',
        'matplotlib'
      ],
      packages=["duet"])
