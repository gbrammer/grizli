import unittest

import os

import numpy as np

from .. import grismconf, GRIZLI_PATH

def test_transform():
    """
    Test JWST transforms
    """
    
    for instrument in ['NIRISS','NIRCAM','WFC3']:
        for grism in 'RC':
            for module in 'AB':
                
                tr = grismconf.JwstDispersionTransform(instrument=instrument, 
                                            grism=grism, module=module)
                
                #print(instrument, grism, module, tr.forward(1024, 1024))
                assert(np.allclose(tr.forward(1024, 1024), 1024))
                
                # Forward, Reverse
                x0 = np.array([125., 300])
                to = tr.forward(*x0)
                fro = np.squeeze(tr.reverse(*to))
                
                assert(np.allclose(x0-fro, 0.))


def test_read():
    
    CONF_PATH = os.path.join(GRIZLI_PATH, 'CONF')
    
    wfc3_file = os.path.join(CONF_PATH, 'G141.F140W.V4.32.conf')
    if os.path.exists(wfc3_file):
        conf = grismconf.aXeConf(wfc3_file)
        
    wfc3_gc = os.path.join(CONF_PATH, 'GRISM_WFC3/IR/G141.conf')
    if os.path.exists(wfc3_gc):
        conf = grismconf.TransformGrismconf(conf_file=wfc3_gc)
        