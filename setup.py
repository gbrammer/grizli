#!/usr/bin/env python

#from distutils.core import setup
#from distutils.extension import Extension
from setuptools import setup
from setuptools.extension import Extension
from setuptools.config import read_configuration

#import ah_bootstrap
#import builtins

import subprocess

import os
import numpy

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

if not os.path.exists('grizli/utils_c/interp.pyx'):
    USE_CYTHON = False
    
if USE_CYTHON:
    cext = '.pyx'
else:
    cext = '.c'

print('C extension: {0}'.format(cext))

if os.name == 'nt':
    # Windows
    extensions = [
        Extension("grizli.utils_c.interp", ["grizli/utils_c/interp"+cext],
            include_dirs = [numpy.get_include()]),
        
        Extension("grizli.utils_c.disperse", ["grizli/utils_c/disperse"+cext],
            include_dirs = [numpy.get_include()]),
    ]
else:
    # Not windows
    extensions = [
        Extension("grizli.utils_c.interp", ["grizli/utils_c/interp"+cext],
            include_dirs = [numpy.get_include()],
            libraries=["m"]),
        
        Extension("grizli.utils_c.disperse", ["grizli/utils_c/disperse"+cext],
            include_dirs = [numpy.get_include()],
            libraries=["m"]),
    ] 

#update version
if os.path.exists('.git'):
    args = 'git describe --tags'
    p = subprocess.Popen(args.split(), stdout=subprocess.PIPE)
    long_version = p.communicate()[0].decode("utf-8").strip()
    spl = long_version.split('-')

    if len(spl) == 3:
        main_version = spl[0]
        commit_number = spl[1]
        version_hash = spl[2]
        version = f'{main_version}.dev{commit_number}'
    else:
        version_hash = '---'
        version = long_version

    print('Git version: {0}'.format(version))
else:
    # e.g., on pip
    version = long_version = version_hash = '1.3.2'

# Manual version tag
version = '1.4.0' # New aws tools

version_str =f"""# git describe --tags
__version__ = "{version}"
__long_version__ = "{long_version}"
__version_hash__ = "{version_hash}" """

fp = open('grizli/version.py','w')
fp.write(version_str)
fp.close()

if USE_CYTHON:
    extensions = cythonize(extensions, language_level="3")

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "grizli",
    version = version,
    author = "Gabriel Brammer",
    author_email = "gbrammer@gmail.com",
    description = "Grism redshift and line analysis software",
    license = "MIT",
    url = "https://github.com/gbrammer/grizli",
    download_url = "https://github.com/gbrammer/grizli/tarball/{0}".format(version),
    packages=['grizli', 'grizli/pipeline', 'grizli/utils_c', 'grizli/tests', 'grizli/galfit', 'grizli/aws'],
    install_requires = ['numpy','cython','astropy'], 
    classifiers=[
        "Development Status :: 1 - Planning",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    ext_modules = extensions,
    package_data={'grizli': ['data/*', 'data/*fits.gz', 'data/templates/*', 'data/templates/stars/*', 'data/templates/fsps/*']},
    # scripts=['grizli/scripts/flt_info.sh'],
)
