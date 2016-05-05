"""
Model grism spectra in individual FLTs

see model.py
   
"""

__version__ = "0.1.1.dev" 

import os

import utils_c
import grism
import model
#import multifit

if os.getenv('GRIZLI') is None:
    print """
Warning: $GRIZLI system variable not set, `grizli`
won't be able to find the aXe configuration files!
(These assumed to be in $GRIZLI/CONF.)
    """

