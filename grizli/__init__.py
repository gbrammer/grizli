"""
Pipeline for flexible modeling and extration of slitless spectroscopy
"""
from .version import __version__

import os

# Will get ImportError: No module named disperse" if imported in the repo directory
# if os.path.exists('README.rst') & os.path.exists('LICENSE.txt'):
#     print("""
# Warning: `import grizli` will fail if the working directory is the place
# where the code repository was cloned and compiled!
# """)

# Test that GRIZLI system variable is set
if os.getenv('GRIZLI') is None:
    GRIZLI_PATH = os.path.join(os.path.dirname(__file__), 'data/')
else:
    GRIZLI_PATH = os.getenv('GRIZLI')
