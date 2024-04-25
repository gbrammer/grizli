"""
update LW badpix table from C. Willott

https://github.com/gbrammer/grizli/issues/200

"""

import os
import astropy.io.fits as pyfits
import numpy as np
from grizli import utils

new_list = ['https://github.com/gbrammer/grizli/files/13878784/badpixlist_along_20240108.txt', 
 'https://github.com/gbrammer/grizli/files/13878785/badpixlist_blong_20240108.txt']

old_list = ['nrc_badpix_231206_NRCALONG.fits.gz', 'nrc_badpix_231206_NRCBLONG.fits.gz']

for new_file, old_file in zip(new_list, old_list):
    det = os.path.basename(new_file).split('_')[1]
    
    new_data = utils.read_catalog(new_file, format='ascii')

    with pyfits.open(old_file) as im:
        orig_bad = im[0].data.sum()
    
        for x, y in zip(new_data['col1'], new_data['col2']):
            im[0].data[y,x] |= 1
        
        new_bad = im[0].data.sum()
    
        msg = f'{det} prev = {orig_bad.sum()}  new = {new_bad.sum()}'
        print(msg)
        
    out_file = f'nrc_badpix_240112_NRC{det.upper()}.fits.gz'
    im.writeto(out_file, overwrite=True)
