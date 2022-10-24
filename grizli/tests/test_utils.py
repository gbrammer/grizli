import unittest

import numpy as np

from grizli import utils


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
            for (v, vs) in un:
                #print(v)
                assert(np.allclose(un.array[un[v]], v))
                assert(np.allclose(un[v], vs))
                assert(np.allclose(un.array[vs], v))
    
    
    def test_sip_rot90(self):
        """
        Test functions to rotate a SIP FITS WCS/Header by steps of 90 degrees
        """
        import astropy.wcs
        import astropy.io.fits
        import matplotlib.pyplot as plt
        
        # NIRCam header
        hdict = {'WCSAXES': 2,
         'CRPIX1': 1023.898,
         'CRPIX2': 1024.804,
         'CD1_1': 1.1134906417684e-05,
         'CD1_2': 1.3467824656321e-05,
         'CD2_1': 1.3420087352956e-05,
         'CD2_2': -1.1207451566394e-05,
         'CDELT1': 1.0,
         'CDELT2': 1.0,
         'CUNIT1': 'deg',
         'CUNIT2': 'deg',
         'CTYPE1': 'RA---TAN-SIP',
         'CTYPE2': 'DEC--TAN-SIP',
         'CRVAL1': 214.96738973536,
         'CRVAL2': 52.90902634452,
         'LONPOLE': 180.0,
         'LATPOLE': 52.90902634452,
         'MJDREF': 0.0,
         'RADESYS': 'ICRS',
         'A_ORDER': 5,
         'A_0_2': -1.5484928795603e-06,
         'A_0_3': -8.2999183617139e-12,
         'A_0_4': -5.7190213371208e-15,
         'A_0_5': 4.07621451394597e-18,
         'A_1_1': -1.1539811084739e-05,
         'A_1_2': 1.5338981510392e-09,
         'A_1_3': -5.0844995554426e-14,
         'A_1_4': 7.75217206451159e-17,
         'A_2_0': 1.90893037341374e-06,
         'A_2_1': -9.4268503941122e-11,
         'A_2_2': 3.36511405738904e-15,
         'A_2_3': 1.01913003404162e-17,
         'A_3_0': 1.43597235355948e-09,
         'A_3_1': -4.6987250431983e-14,
         'A_3_2': 1.57099611614835e-16,
         'A_4_0': 1.02466627979516e-14,
         'A_4_1': 8.67685711756312e-18,
         'A_5_0': 8.38989007060477e-17,
         'B_ORDER': 5,
         'B_0_2': -6.7240563576112e-06,
         'B_0_3': 1.56045160756803e-09,
         'B_0_4': -3.1953236128479e-14,
         'B_0_5': 6.10311418213124e-17,
         'B_1_1': 3.60929897034643e-06,
         'B_1_2': -1.0268629323178e-10,
         'B_1_3': 1.71406384407525e-14,
         'B_1_4': 8.84616612338102e-18,
         'B_2_0': 4.89531710581344e-06,
         'B_2_1': 1.39303581747843e-09,
         'B_2_2': -8.0509590680251e-15,
         'B_2_3': 1.49127030961831e-16,
         'B_3_0': -1.337273982495e-11,
         'B_3_1': 1.81206781900953e-14,
         'B_3_2': 1.17056127262783e-17,
         'B_4_0': 2.30644152575784e-14,
         'B_4_1': 8.12712563368295e-17,
         'B_5_0': 4.96075368013397e-18,
         'NAXIS': 2,
         'NAXIS1': 2048,
         'NAXIS2': 2048}

        h = astropy.io.fits.Header()
        for k in hdict:
            h[k] = hdict[k]

        wcs = astropy.wcs.WCS(h, relax=True)

        for rot in range(-5,6):
            _ = utils.sip_rot90(wcs, rot, compare=False, verbose=False)

        orig = utils.to_header(wcs)

        xp = [356, 1024]

        # Rotate 90 degrees twice
        new, new_wcs, desc = utils.sip_rot90(orig, 1,
                                             compare=False, verbose=False)
        new2, new2_wcs, desc2 = utils.sip_rot90(new, 1,
                                             compare=False, verbose=False)

        # Rotate 180
        new2b, new2b_wcs, desc2b = utils.sip_rot90(orig, 2,
                                             compare=False, verbose=False)

        # Test coordinates
        rd = wcs.all_pix2world(np.atleast_2d(xp), 1)
        rd1 = new_wcs.all_pix2world(np.atleast_2d([xp[1], 2048-xp[0]]), 1)
        assert(np.allclose(rd, rd1))

        rd2 = new2b_wcs.all_pix2world(2048-np.atleast_2d(xp), 1)
        assert(np.allclose(rd, rd2))

        # Back to start
        newx, newx_wcs, descx = utils.sip_rot90(new2b, 2,
                                                compare=False, verbose=False)

        for i in range(new['A_ORDER']+1):
            for j in range(new['B_ORDER']+1):
                Aij  = f'A_{i}_{j}'
                Bij  = f'B_{i}_{j}'
                if Aij not in new:
                    continue

                assert(np.allclose(new2[Aij], new2b[Aij]))
                assert(np.allclose(new2[Bij], new2b[Bij]))
                assert(np.allclose(newx[Aij], orig[Aij]))
                assert(np.allclose(newx[Bij], orig[Bij]))

        print('sip_rot90 tests passed')
        
        plt.close('all')