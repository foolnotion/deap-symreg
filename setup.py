#!/usr/bin/env python
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize('symreg.py')
)
