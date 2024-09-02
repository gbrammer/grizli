#!/usr/bin/env python
# encoding: utf-8
"""
Cython speedups for Grizli functions

- replaced with numba

"""

# from . import disperse
# from . import interp

# For back compatibility
from ..utils_numba import disperse
from ..utils_numba import interp

#from .disperse import *
#from .interp import *
