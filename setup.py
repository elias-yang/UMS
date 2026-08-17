# -*- coding: utf-8 -*-


import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup


def _build_extensions():
    return cythonize([
        Extension(
            "opencood.utils.box_overlaps",
            ["opencood/utils/box_overlaps.pyx"],
            include_dirs=[numpy.get_include()],
        )
    ])


setup(
    ext_modules=_build_extensions(),
)
