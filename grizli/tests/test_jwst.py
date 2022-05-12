"""
Tests for JWST imaging and spectra, including 
"""
import os
import glob
import unittest

import numpy as np
from .. import utils, prep, multifit, fitting, GRIZLI_PATH


class JWSTFittingTools(unittest.TestCase):
    
    def test_multibeam(self):
        """
        Can we initialize a multibeam file?
        """
        path = os.path.dirname(utils.__file__)
        print(path)
        beams_file = '/Users/victoriastrait/Desktop/grizli_test_data/jw01324001001_01243.beams.fits'
        mb = multifit.MultiBeam(beams_file, group_name='jw01324001001',
                                MW_EBV=-1, fcontam=0.1, sys_err=0.03)
    
        assert(mb.N == 2)
        assert('F115W' in mb.PA)
        
        _ = mb.compute_model()
        
        spec = mb.oned_spectrum()
        
    