#!/usr/bin/env python
import numpy
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


include_dirs = [numpy.get_include()]
define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
libraries=["m"]
extensions = [
    Extension(name="grizli.utils_c.interp",
              sources=["grizli/utils_c/interp.pyx"],
              include_dirs=include_dirs,
              libraries=libraries,
              define_macros=define_macros,
              ),
    Extension(name="grizli.utils_c.disperse",
              sources=["grizli/utils_c/disperse.pyx"],
              include_dirs=include_dirs,
              libraries=libraries,
              define_macros=define_macros,
              ),
    ]
ext_modules = cythonize(extensions, language_level=3)

setup(ext_modules=ext_modules)
