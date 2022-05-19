"""
Tests for JWST imaging and spectra, including 
"""
import os
import glob
import unittest

import numpy as np
from .. import utils, prep, multifit, fitting, GRIZLI_PATH


class JWSTFittingTools(unittest.TestCase):
    
    def test_config(self):
        """
        Fetch config files if CONF not found
        """
        conf_path = os.path.join(GRIZLI_PATH, 'CONF')
        if not os.path.exists(conf_path): # if CONF dir doesn't already exist, create it
            os.mkdir(conf_path)
        
        if not os.path.exists(os.path.join(conf_path,
                              'GR150C.F115W.conf')):
            print(f'Download config and calib files to {conf_path}')
            #utils.fetch_default_calibs(ACS=False)
            utils.fetch_config_files(get_epsf=True, get_jwst=True)
            files = glob.glob(f'{conf_path}/*')
            print('Files: ', '\n'.join(files))

        assert(os.path.exists(os.path.join(conf_path,
                              'GR150C.F115W.conf')))
        return True
    
    def test_multibeam(self):
        """
        Can we initialize a multibeam file?
        """
        path = os.path.dirname(utils.__file__)
        print(path)
        beams_file = path + '/tests/data/niriss_jw01324001001_test.beams.fits'
        mb = multifit.MultiBeam(beams_file, group_name='jw01324001001',
                                MW_EBV=-1, fcontam=0.1, sys_err=0.03)
    
        assert(mb.N == 2)
        assert('F115W' in mb.PA)
        
        _ = mb.compute_model()
        
        spec = mb.oned_spectrum()
        
    