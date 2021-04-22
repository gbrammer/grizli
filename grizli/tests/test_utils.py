import unittest

import numpy as np
from .. import utils

class UtilsTester(unittest.TestCase):
    def test_log_zgrid(self):
        """
        Logarithmic-spaced grid
        """
        value = np.array([0.1,  0.21568801,  0.34354303,  
                          0.48484469,  0.64100717, 0.8135934])
                          
        np.testing.assert_allclose(utils.log_zgrid([0.1, 1], 0.1), value, rtol=1e-06, atol=0, equal_nan=False, err_msg='', verbose=True)
    
    
    def test_arg_dict(self):
        """
        Argument parsing
        """
        
        argv = 'arg1 --p1=1 --l1=1,2,3 --pdict.k1=1 -flag'.split()
        args, kwargs = utils.argv_to_dict(argv)
        
        assert(args == ['arg1'])
        
        result = {'p1': 1, 'l1': [1, 2, 3], 'pdict': {'k1': 1}, 'flag': True}
        
        # Test that dictionaries are the same using __repr__ method
        assert(kwargs.__repr__() == result.__repr__())
                
        