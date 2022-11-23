"""
Tests for spectra, don't implement now because will fail on
travis where we haven't set up the full GRIZLI path and reference
files
"""
import os
import glob
import unittest

import numpy as np
from grizli import utils, prep, multifit, fitting, GRIZLI_PATH


class FittingTools(unittest.TestCase):
    
    def test_config(self):
        """
        Fetch config files if CONF not found
        """
        new = []
        for subd in ['iref','jref','CONF']:
            conf_path = os.path.join(GRIZLI_PATH, subd)
            if not os.path.exists(conf_path):
                new.append(subd)
                os.mkdir(conf_path)
        
        if 'CONF' in new:
            print(f'Download config and calib files to {conf_path}')
            utils.fetch_default_calibs(get_acs=False)
            utils.fetch_config_files(get_epsf=True)
            files = glob.glob(f'{conf_path}/*')
            print('Files: ', '\n'.join(files))

        assert os.path.exists(os.path.join(conf_path,
                              'G141.F140W.V4.32.conf'))
            
    def test_multibeam(self):
        """
        Can we initialize a multibeam file?
        """
        path = os.path.dirname(utils.__file__)
        print(path)
        beams_file = os.path.join(path, 'tests', 'data', 'j033216m2743_00152.beams.fits')
        mb = multifit.MultiBeam(beams_file, group_name='j033216m2743',
                                MW_EBV=-1, fcontam=0.1, sys_err=0.03)
    
        assert mb.N == 2
        assert 'G102' in mb.PA
        
        expn, expt = mb.compute_exptime()
        assert np.allclose(expt['G102'], 1102.93, rtol=1.e-2)
        
        mb.compute_model()
        
        spec = mb.oned_spectrum()
        
    def test_redshift_fit(self):
        """
        Can we run the full fit?
        """
        path = os.path.dirname(utils.__file__)
        data_path = os.path.join(path, 'tests', 'data')
    
        res = fitting.run_all_parallel(152, zr=[1.7, 1.8], verbose=True,
                                 root=os.path.join(data_path, 'j033216m2743'),
                                 args_file=os.path.join(data_path, 'fit_args.npy'), 
                                 get_output_data=True, 
                                 protect=False)
        
        assert np.allclose(res[2].meta['z_map'][0], 1.7429, rtol=1.e-2)
            
        # Clean up
        files = glob.glob('j033216m2743_00152*')
        for file in files:
            os.remove(file)
