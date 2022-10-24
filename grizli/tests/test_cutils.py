import unittest

import numpy as np

from grizli import utils
from grizli.utils_c import interp


def test_cinterp():
    """
    Simple interpolation
    """
    xarr = np.array([0.0,1.0,2.0])
    yarr = np.array([0.0,1.0,0.0])
    result = interp.interp_c(np.array([0.5]), xarr, yarr)
    assert np.allclose(result, 0.5)


def test_cinterp_conserve():
    """
    Linear interpolation conserving the integral 
    """
    xarr = np.arange(1., 3., 0.0001)
    yarr = (np.abs(xarr-2.) <= 0.1)*1.
    
    np.random.seed(1)
    xlr = np.random.rand(10)*2+1
    xlr.sort()
    
    ylr = interp.interp_conserve_c(xlr, xarr, yarr)
    
    assert np.allclose(np.trapz(ylr, xlr), np.trapz(yarr, xarr))
    
    ylr_slow = interp.interp_conserve(xlr, xarr, yarr)
    assert np.allclose(np.trapz(ylr_slow, xlr), np.trapz(yarr, xarr))
    