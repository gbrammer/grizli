#from distutils.core import setup
#from distutils.extension import Extension
from setuptools import setup
from setuptools.extension import Extension

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

extensions = [
    Extension("grizli.utils_c.interp", ["grizli/utils_c/interp"+cext],
        include_dirs = [numpy.get_include()],
        libraries=["m"]),
        
    Extension("grizli.utils_c.disperse", ["grizli/utils_c/disperse"+cext],
        include_dirs = [numpy.get_include()],
        libraries=["m"]),

]

#update version
args = 'git describe --tags'
p = subprocess.Popen(args.split(), stdout=subprocess.PIPE)
version = p.communicate()[0].decode("utf-8").strip()

# version = "0.8.0"
# version = "0.9.0" # bounded fits by default
# version = "0.10.0" # Relatively small fixes, fix bug in 1D wave
# version = "0.11.0" # Refactored parameter files
version = "0.12.0" # Increment to fix tag tar files

version_str = """# git describe --tags
__version__ = "{0}"\n""".format(version)

fp = open('grizli/version.py','w')
fp.write(version_str)
fp.close()
print('Git version: {0}'.format(version))

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
    version = version,
    author = "Gabriel Brammer",
    author_email = "gbrammer@gmail.com",
    description = "Grism redshift and line analysis software",
    license = "MIT",
    url = "https://github.com/gbrammer/grizli",
    download_url = "https://github.com/gbrammer/grizli/tarball/{0}".format(version),
    packages=['grizli', 'grizli/pipeline', 'grizli/utils_c', 'grizli/tests', 'grizli/galfit'],
    # requires=['numpy', 'scipy', 'astropy', 'drizzlepac', 'stwcs'],
    # long_description=read('README.rst'),
    classifiers=[
        "Development Status :: 1 - Planning",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    ext_modules = extensions,
    package_data={'grizli': ['data/*', 'data/*fits.gz', 'data/templates/*', 'data/templates/stars/*', 'data/templates/fsps/*']},
    # scripts=['grizli/scripts/flt_info.sh'],
)
