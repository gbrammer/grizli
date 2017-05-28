import unittest

import numpy as np
from .. import utils

class Dummy(unittest.TestCase):  
    def test_log_zgrid(self):
        value = np.array([ 0.1       ,  0.21568801,  0.34354303,  0.48484469,  0.64100717, 0.8135934 ])
        np.testing.assert_allclose(utils.log_zgrid([0.1,1],0.1), value, rtol=1e-06, atol=0, equal_nan=False, err_msg='', verbose=True)
  