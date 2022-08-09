from Cython.Build import cythonize
from setuptools import setup

setup(
    ext_modules = cythonize("segmentator/formulas.pyx")
)

# """
# To use:
# python setup.py build_ext --inplace
# """
