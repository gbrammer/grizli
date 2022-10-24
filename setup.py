#!/usr/bin/env python
import os
import subprocess

import numpy
from setuptools import setup
from setuptools.extension import Extension

include_dirs = [numpy.get_include()]
if os.name == 'nt':
    # Windows
    libraries = None
else:
    # Not windows
    libraries = ["m"]
define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
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

setup(ext_modules=extensions)
