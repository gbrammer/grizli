import glob
import numpy as np
import astropy.io.fits as pyfits

# Make data arrays int zeros to zip smaller

files = glob.glob('*fits')
for file in files: 
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
            
    im.writeto(file, overwrite=True)