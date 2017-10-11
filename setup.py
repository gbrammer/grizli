# ASSUMPTIONS:
# 1. in anaconda (preferably with astroconda distribution)
# 2. python 3 (for sewpy git fork)

#from distutils.core import setup
#from distutils.extension import Extension
from setuptools import setup
from setuptools.extension import Extension

import os
import numpy
import pip
import subprocess

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

extensions = [
    Extension("grizli.utils_c.interp", ["grizli/utils_c/interp"+cext],
        include_dirs = [numpy.get_include()],),
        
    # Extension("grizli/utils_c/nmf", ["grizli/utils_c/nmf"+cext],
    #     include_dirs = [numpy.get_include()],),
    
    Extension("grizli.utils_c.disperse", ["grizli/utils_c/disperse"+cext],
        include_dirs = [numpy.get_include()],),

]

#update version
args = 'git describe --tags'
p = subprocess.Popen(args.split(), stdout=subprocess.PIPE)
version = p.communicate()[0].decode("utf-8").strip()
#lines = open('grizli/version.py').readlines()
version_str = """# git describe --tags
__version__ = "{0}"\n""".format(version)
fp = open('grizli/version.py','w')
fp.write(version_str)
fp.close()
print('Git version: {0}'.format(version))

if USE_CYTHON:
    extensions = cythonize(extensions)

# For me 'reproject' is having trouble installing.
# Remove if that is a general problem.
pip_packages = ['peakutils', 'scikit-learn', 'astroquery', 'shapely', 'reproject', 
    'git+https://github.com/cmccully/lacosmicx.git', 'git+https://github.com/gbrammer/sewpy.git']

for package in pip_packages:
    pip.main(['install', package])

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "grizli",
    version = "0.3.0",
    author = "Gabriel Brammer",
    author_email = "gbrammer@gmail.com",
    description = "Grism redshift and line analysis software",
    license = "MIT",
    url = "https://github.com/gbrammer/grizli",
    download_url = "https://github.com/gbrammer/grizli/tarball/0.2.1",
    packages=['grizli', 'grizli/utils_c', 'grizli/tests'],
    # 'install_requires' should work, but doesn't, not sure why...
    #install_requires=['drizzlepac', 'stwcs', 'photutils', 'pysynphot', 'peakutils', 
    #    'scikit-learn', 'astroquery', 'shapely', 'reproject'],
    # long_description=read('README.rst'),
    classifiers=[
        "Development Status :: 1 - Planning",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    ext_modules = extensions,
    package_data={'grizli': ['data/*', 'data/templates/*', 'data/templates/stars/*', 'data/templates/fsps/*']},
    # scripts=['grizli/scripts/flt_info.sh'],
)
