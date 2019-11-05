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
    GRIZLI_PATH = os.path.join(os.path.dirname(__file__), 'data/')

    print("""
Warning: $GRIZLI system variable not set and will default to {0}. 
Grizli will need to find the configuration files in $GRIZLI/CONF.
    """.format(GRIZLI_PATH))
    #GRIZLI_PATH = '/usr/local/share/grizli_home'
else:
    GRIZLI_PATH = os.getenv('GRIZLI')
    
try:
    import sep
except:
    print("""
Couldn't `import sep`.  SExtractor replaced with SEP 
in April 2018.  Install with `pip install sep`.
""")

try:
    import tristars
except:
    print("""
Couldn't `import tristars`.  Get it from https://github.com/gbrammer/tristars to enable improved blind astrometric matching with triangle asterisms.
""")
