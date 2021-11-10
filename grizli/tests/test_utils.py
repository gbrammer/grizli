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


    def test_ctime(self):
        """
        ctime conversion
        """
        mtime = 'Mon Sep 16 11:23:27 2019'
        iso = utils.ctime_to_iso(mtime, strip_decimal=True)
        assert(iso == '2019-09-16 11:23:27')

        iso = utils.ctime_to_iso(mtime, strip_decimal=False)
        assert(iso == '2019-09-16 11:23:27.000')


    def test_multiprocessing_ndfilter(self):
        """
        Test the multiprocessing filter
        """
        import scipy.ndimage as nd
        
        rnd = np.random.normal(size=(512,512))
        
        fsize = 3
        f_serial = nd.median_filter(rnd, size=fsize)
        
        for n_proc in [-1, 0, 4, 20]:
            for cutout_size in [128, 256, 1024]:
                f_mp = utils.multiprocessing_ndfilter(rnd,
                                                      nd.median_filter,
                                                      size=fsize,
                                                      cutout_size=cutout_size,
                                                      n_proc=n_proc,
                                                      verbose=False)

                assert(np.allclose(f_serial, f_mp))
        
        footprint = np.ones((fsize,fsize), dtype=bool)
        
        for n_proc in [-1, 0, 4, 20]:
            for cutout_size in [128, 256, 1024]:
                f_mp = utils.multiprocessing_ndfilter(rnd,
                                                      nd.median_filter,
                                                      size=None,
                                                      footprint=footprint,
                                                      cutout_size=cutout_size,
                                                      n_proc=n_proc,
                                                      verbose=False)

                assert(np.allclose(f_serial, f_mp))
        
        # Passing arguments
        filter_args = (50,)
        n_proc = 4
        cutout_size = 128
        
        f_mp = utils.multiprocessing_ndfilter(rnd,
                                              nd.percentile_filter,
                                              filter_args=filter_args,
                                              size=None,
                                              footprint=footprint,
                                              cutout_size=cutout_size,
                                              n_proc=n_proc,
                                              verbose=False)

        assert(np.allclose(f_serial, f_mp))


    def test_unique(self):
        """
        Test ``Unique`` helper
        """
        
        data = [1,1,1,2,2,9]
        
        for d in [data, np.array(data)]:
            un = utils.Unique(d, verbose=False)
            
            # Unique values
            assert(len(un.values) == 3)
            assert(np.allclose(un.values, [1,2,9]))
            
            # Array indices
            assert(np.allclose(un.indices, [0, 0, 0, 1, 1, 2]))
            
            # Missing key 
            assert(un[-1].sum() == 0)
            
            # Existing key
            assert(un[1].sum() == 3)
            
            # __iter__ and __get__ methods
            for v in un:
                #print(v)
                assert(np.allclose(un.array[un[v]], v))