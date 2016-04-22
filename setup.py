from distutils.core import setup
from distutils.extension import Extension

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
    cext = '.pyx'
except ImportError:
    USE_CYTHON = False
    cext = '.c'
    
import os
import numpy

extensions = [
    Extension("grizli/utils_c/interp", ["grizli/utils_c/interp"+cext],
        include_dirs = [numpy.get_include()],),
        
    # Extension("grizli/utils_c/nmf", ["grizli/utils_c/nmf"+cext],
    #     include_dirs = [numpy.get_include()],),
    
    Extension("grizli/utils_c/disperse", ["grizli/utils_c/disperse"+cext],
        include_dirs = [numpy.get_include()],),

]

if USE_CYTHON:
    extensions = cythonize(extensions)

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "grizli",
    version = "0.1.0",
    author = "Gabriel Brammer",
    author_email = "gbrammer@gmail.com",
    description = "Grism redshift and line analysis software",
    license = "MIT",
    url = "https://github.com/gbrammer/grizli",
    download_url = "https://github.com/gbrammer/grizli/tarball/0.1.0",
    packages=['grizli', 'grizli/utils_c'],
    # requires=['numpy', 'scipy', 'astropy', 'drizzlepac', 'stwcs'],
    # long_description=read('README.rst'),
    classifiers=[
        "Development Status :: 1 - Planning",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    ext_modules = extensions,
    package_data={'grizli': ['data/*']},
    # scripts=['grizli/scripts/flt_info.sh'],
)