import glob
import numpy as np
import astropy.io.fits as pyfits

# Make data arrays int zeros to zip smaller

def download_from_mast():

    import mastquery.utils

    exposure_files = [
        "jw11373003002_06101_00004_mirimage_rate.fits", # F770W
        "jw07583002001_08101_00004_nrca2_rate.fits",    # F200W-CLEAR
        "jw10790216001_03201_00003_nrcblong_rate.fits", # F356W-CLEAR
        "jw05893029001_03101_00001_nrcalong_rate.fits", # F356W-CLEAR direct
        "jw05893029001_02101_00006_nrcalong_rate.fits", # F444W-GRISMR
    ]

    for file in exposure_files:
        strip_file(file)

def strip_files():
    files = glob.glob('*fits')
    for file in files: 
        strip_file(file)

def strip_file(file):

    im = pyfits.open(file)
    for e in im:
        try:
            sh = im[e].data.shape
            im[e].data = np.zeros(sh, dtype=np.uint16)
        except:
            print(file, im[e], 'no data')
    
        if 'PATTSIZE' in im[e].header:
            #print(im[e].header['PATTSIZE'])
            im[e].header['PATTSIZE'] = 'SMALL'
            
    im.writeto(file + ".gz", overwrite=True)