"""
Tests for spectra, don't implement now because will fail on
travis where we haven't set up the full GRIZLI path and reference
files
"""
import os
import glob
import unittest

import numpy as np
from .. import utils, prep, multifit, fitting


class FittingTools(unittest.TestCase):
    pass
    # def test_multibeam(self):
    #     path = os.path.dirname(utils.__file__)
    #     print(path)
    #     beams_file = path+'/tests/data/j033216m2743_00152.beams.fits'
    #     mb = multifit.MultiBeam(beams_file, group_name='j033216m2743',
    #                             MW_EBV=-1, fcontam=0.1, sys_err=0.03)
    #
    #
    # def test_redshift_fit(self):
    #     path = os.path.dirname(utils.__file__)
    #     data_path = path +'/tests/data/'
    #
    #     res = fitting.run_all_parallel(152, zr=[1.7, 1.8], verbose=False,
    #                              root=data_path+'j033216m2743',
    #                              args_file=data_path+'fit_args.npy')
    #
    #     # Clean up
    #     files = glob.glob('j033216m2743_00152*')
    #     for file in files:
    #         os.remove(file)
