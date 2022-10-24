import unittest

import os

import numpy as np

from grizli import grismconf as grizliconf
from grizli import GRIZLI_PATH


def test_transform():
    """
    Test JWST transforms
    """
    import astropy.io.fits as pyfits
    
    for instrument in ['NIRISS','NIRCAM','WFC3']:
        for grism in 'RC':
            for module in 'AB':
                
                tr = grizliconf.JwstDispersionTransform(instrument=instrument,
                                            grism=grism, module=module)
                
                #print(instrument, grism, module, tr.forward(1024, 1024))
                assert(np.allclose(tr.forward(1024, 1024), 1024))
                
                # Forward, Reverse
                x0 = np.array([125., 300])
                to = tr.forward(*x0)
                fro = np.squeeze(tr.reverse(*to))
                
                assert(np.allclose(x0-fro, 0.))

    # From header
    nis = pyfits.Header()
    nis['INSTRUME'] = 'NIRISS'

    rot90 = {'GR150C':2,  # 180 degrees
             'GR150R':3,   # 270 degrees CW
             }

    for gr in rot90:
        nis['FILTER'] = gr
        nis['PUPIL'] = 'F150W'

        tr = grizliconf.JwstDispersionTransform(header=nis)
        assert(tr.instrument == nis['INSTRUME'])
        assert(tr.grism == nis['FILTER'])
        assert(tr.rot90 == rot90[gr])


def test_read():
    
    CONF_PATH = os.path.join(GRIZLI_PATH, 'CONF')
    
    wfc3_file = os.path.join(CONF_PATH, 'G141.F140W.V4.32.conf')
    if os.path.exists(wfc3_file):
        conf = grizliconf.aXeConf(wfc3_file)
        
    try:
        import grismconf
        has_grismconf = True
    except ImportError:
        has_grismconf = False
    
    wfc3_gc = os.path.join(CONF_PATH, 'GRISM_WFC3/IR/G141.conf')
    if os.path.exists(wfc3_gc) & has_grismconf:
        conf = grizliconf.TransformGrismconf(conf_file=wfc3_gc)
        