"""
Mask pixels at the edge of NIRCam LW
"""
import astropy.io.fits as pyfits
import glob
import numpy as np
import os

os.system('gunzip nrc_badpix_231206_*fits.gz')

nircam_edge = 8

files = glob.glob('nrc_badpix_231206_*fits')
files.sort()
for file in files:
    with pyfits.open(file, mode='update') as im:
        dq = np.zeros_like(im[0].data)
        dq[:nircam_edge,:] |= 1024
        dq[-nircam_edge:,:] |= 1024
        dq[:,:nircam_edge] |= 1024
        dq[:,-nircam_edge:] |= 1024
        im[0].data |= (dq > 0)
        
        im.flush()
        
os.system('gzip nrc_badpix_231206_*fits')
