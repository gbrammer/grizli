"""
Model grism spectra in individual FLTs

see model.py
   
"""

import os

import utils_c
import grism
import model
#import multifit

if os.getenv('GRIZLI') is None:
    print """
Warning: $GRIZLI system variable not set, `grizli`
won't be able to find the aXe configuration files!
    """
else:
    CONF_PATH = os.path.join(os.getenv('GRIZLI'), 'CONF')


