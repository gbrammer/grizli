"""
Pipeline for flexible modeling and extration of slitless spectroscopy
"""
#__version__ = "0.1.1.dev" 
from .version import __version__

import os

## Will get ImportError: No module named disperse" if imported in the repo directory
# if os.path.exists('README.rst') & os.path.exists('LICENSE.txt'):
#     print("""
# Warning: `import grizli` will fail if the working directory is the place 
# where the code repository was cloned and compiled!
# """)

# Module imports
# from . import utils_c
# from . import utils
# from . import grismconf
# from . import model
# from . import multifit

# Test that GRIZLI system variable is set
if os.getenv('GRIZLI') is None:
    print("""
Warning: $GRIZLI system variable not set, `grizli`
won't be able to find the aXe configuration files!
(These assumed to be in $GRIZLI/CONF.)
    """)

try:
    import sep
except:
    print("""
Couldn't `import sep`.  SExtractor replaced with SEP 
in April 2018.  Install with `pip install sep`.
""")
