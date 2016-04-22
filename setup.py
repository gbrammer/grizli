from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import os
import numpy

extensions = [
    Extension("grizli/utils_c/interp", ["grizli/utils_c/interp.pyx"],
        include_dirs = [numpy.get_include()],),
        
    # Extension("grizli/utils_c/nmf", ["grizli/utils_c/nmf.pyx"],
    #     include_dirs = [numpy.get_include()],),
    
    Extension("grizli/utils_c/disperse", ["grizli/utils_c/disperse.pyx"],
        include_dirs = [numpy.get_include()],),
    
]

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "grizli",
    version = "0.1",
    author = "Gabriel Brammer",
    author_email = "gbrammer@gmail.com",
    description = ("Grizli: Grism redshift and line analysis software"),
    license = "MIT",
    url = "http://github.com/gbrammer/grizli",
    packages=['grizli'],
    # requires=['numpy', 'scipy', 'astropy', 'drizzlepac', 'stwcs'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 1 - Planning",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    ext_modules = cythonize(extensions),
    package_data={'grizli': ['data/*']},
    # scripts=['grizli/scripts/flt_info.sh'],
)