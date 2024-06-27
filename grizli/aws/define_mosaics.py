"""
Tools for defining the mosaic WCS footprints for specific HST/JWST fields
"""

import os
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import numpy as np

from .. import utils, prep

AVAILABLE_FIELDS = """#old_ceers
ceers
#orig_glass
abell2744
abell2744par
#primerv1 
#cweb_first
sunrise
j1235
sgas1723
#nep_2mass
jwst_nep
macs0647
rxcj0600
rxj2129
hudf
gdn
ngdeep
smacs0723
macs0417
abell370
macs0416
macs1423
macs1149
#panoramic
#primer_uds
#primer_cosmos
primer_uds_north
primer_uds_south
primer_cosmos_east
primer_cosmos_west
snh0pe
abells1063
j061548m5745
spitzer_idf
elgordo
mrg0138
miri-gto
ulasj1342
sunburst
#PEARLS
gama100033
clg-1212p2733
plck-g165p67
plck-g191p62
tn-j1338m1942
abell1689
""".split()

__all__ = ['AVAILABLE_FIELDS', 'make_wcs', 'show_field_footprint']

def make_wcs(mosaic_field='primer_uds', version='v7.0', canucs_subfield='clu'):
    """
    Define a mosaic WCS
    
    Parameters
    ----------
    mosaic_field : str
        Field name
    
    version : str
        Version code
    
    Returns
    -------
    root : str
        Output file rootname
    
    half_sw : bool
        Should NIRCam SW images have 2x smaller pixels than HST/LW
    
    s3output : str
        AWS S3 path where files will be sent
    
    ref_wcs : `~astropy.wcs.WCS`
        Field WCS
    
    """
    s3output = 's3://grizli-v2/junk/'

    # s3output = None

    mos_path = 's3://grizli-v2/JwstMosaics/v2'

    root = 'smacs0723-grizli-v4.0'
    ref_file = 'smacs0723-grizli-v2-f444w-clear_drc_sci.fits.gz'
    half_sw = True

    if mosaic_field == 'old_ceers':
        root = 'egs-grizli-v3'
        ref_file = 'egs-grizli-v2-f200w-clear_drc_sci.fits.gz'
        half_sw = False

    elif mosaic_field == 'ceers':
        # Split CEERS
        hdu = utils.make_wcsheader(ra=214.9140403, dec=52.9036667, size=(16.38395*60, 5.46132*60),
                                   pixscale=0.04, theta=-310, get_hdu=True)

        hdu.header['CRPIX1'] = 12288
        hdu.header['CRPIX2'] = 4096

        w = pywcs.WCS(hdu.header)
        hne = utils.get_wcs_slice_header(w, slice(0, 24576//2), slice(0, 8192))
        hsw = utils.get_wcs_slice_header(w, slice(24576//2, 24576), slice(0, 8192))
    
        half_sw = True
        if 1:
            ref_file = pywcs.WCS(hne)
            root = 'ceers-ne-grizli-v4.0'
        else:
            ref_file = pywcs.WCS(hsw)
            root = 'ceers-sw-grizli-v4.0'
    
        if 1:
            # Full CEERS
            # hdu = utils.make_wcsheader(ra=214.9140403, dec=52.9036667, size=(16.38395*60, 5.46132*60),
            #                        pixscale=0.04, theta=-310, get_hdu=True)
            NX, NY = 18, 6
            hdu = utils.make_wcsheader(ra=214.92, dec=52.87,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=-310., get_hdu=True)
    
            hdu.header['CRPIX1'] = NX*2048//2
            hdu.header['CRPIX2'] = NY*2048//2
    

            #hdu.header['CRPIX1'] = 12288
            #hdu.header['CRPIX2'] = 4096

            ref_file = pywcs.WCS(hdu.header)
            half_sw = False
        
            root = 'ceers-full-grizli-v5.0'
            root = 'ceers-full-grizli-v6.1' # with Naidu direct imaging
            # root = f'ceers-full-grizli-{version}' # with Naidu direct imaging
        
            root = f'ceers-full-grizli-{version}' # with new weights
        
    elif mosaic_field == 'orig_glass':
        if 0:
            if not os.path.exists('glassp-f444w_drz_sci.fits'):
                os.system('wget "https://www.dropbox.com/sh/lbj7f53szr9if0z/AADR87M2k9OOc1Y65eq-JE9Ta?dl=0&preview=glassp-f444w_drz_sci.fits" -O glass-pascal.zip')
                os.system('unzip glass-pascal.zip  glassp-f444w_drz_sci.fits')
        
            ref_file = 'glassp-f444w_drz_sci.fits'
        else:
            mos_path = 's3://grizli-v2/JwstMosaics/v3'
            ref_file = 'glass-grizli-v3.3-f444w-clear_drc_sci.fits.gz'

        #root = 'glass-grizli-v3'
        #half_sw = False
    
        root = 'glass-grizli-v4.0'
        half_sw = True

    elif mosaic_field.startswith('abell2744'):
    
        # Abell 2744 cluster for uncover
    
        with open('abell2744.wcs.txt','w') as fp:
            fp.write("""SIMPLE  =                    T / conforms to FITS standard
BITPIX  =                    8 / array data type
NAXIS   =                    2 / number of array dimensions
NAXIS1  =                12070
NAXIS2  =                11564
WCSAXES =                    2 / Number of coordinate axes
CRPIX1  =               5123.5 / Pixel coordinate of reference point
CRPIX2  =               4207.5 / Pixel coordinate of reference point
PC1_1   = -1.1111111111111E-05 / Coordinate transformation matrix element
PC2_2   =  1.1111111111111E-05 / Coordinate transformation matrix element
CDELT1  =                  1.0 / [deg] Coordinate increment at reference point
CDELT2  =                  1.0 / [deg] Coordinate increment at reference point
CUNIT1  = 'deg'                / Units of coordinate increment and value
CUNIT2  = 'deg'                / Units of coordinate increment and value
CTYPE1  = 'RA---TAN'           / Right ascension, gnomonic projection
CTYPE2  = 'DEC--TAN'           / Declination, gnomonic projection
CRVAL1  =               3.5875 / [deg] Coordinate value at reference point
CRVAL2  =          -30.3966667 / [deg] Coordinate value at reference point
LONPOLE =                180.0 / [deg] Native longitude of celestial pole
LATPOLE =          -30.3966667 / [deg] Native latitude of celestial pole
MJDREF  =                  0.0 / [d] MJD of fiducial time
RADESYS = 'ICRS'               / Equatorial coordinate system
BSCALE  =                    1
BZERO   =                 -128
END""")
    
        hivo = pyfits.Header.fromtextfile('abell2744.wcs.txt')
        ivo_wcs = pywcs.WCS(hivo)
        ivo_orig = pywcs.WCS(utils.to_header(ivo_wcs))
    
        # Expand to integer N*2048
        hivo = pyfits.Header.fromtextfile('abell2744.wcs.txt')
        dx = 12288 - hivo['NAXIS1']
        dy = 12288 - hivo['NAXIS2']
        hivo['crpix1'] += dx//2 + 0.5
        hivo['crpix2'] += dy//2 + 0.5
        hivo['NAXIS1'] = hivo['NAXIS2'] = 12288
        
        root = 'abell2744clu-grizli-v5'
        root = 'abell2744clu-grizli-v5.1' # First UNCOVER visit and redo GLASS
        root = 'abell2744clu-grizli-v5.2' # Second UNCOVER visit
    
        root = 'abell2744clu-grizli-v5.3' # with new glass parallels
    
        # Expand to include full mosaic
        hivo = pyfits.Header.fromtextfile('abell2744.wcs.txt')
        dim = (int(10*2048), int(7*2048))

        dim = (int(10.5*2048), int(8*2048)) # > 6.6 for the full DD field

        dx = dim[0] - hivo['NAXIS1']
        dy = dim[1] - hivo['NAXIS2']
        hivo['crpix1'] += dx//2 + 0.5
        hivo['crpix2'] += dy//2 + 0.5

        hivo['crpix1'] -= 2800
        hivo['crpix2'] -= 1024
        hivo['crpix2'] += 256+512 # > 6.6 for the full DD field

        hivo['NAXIS1'] = dim[0]
        hivo['NAXIS2'] = dim[1]
    
        root = 'abell2744clu-grizli-v5.4' # with new glass parallels
        root = 'abell2744clu-grizli-v5.5' # with last SN DDT visit

        root = 'abell2744clu-grizli-v5.6' # checking F277W
        root = 'abell2744clu-grizli-v6.0' # All with new flats

        root = 'abell2744clu-grizli-v6.5' # New weight strategy
        root = 'abell2744clu-grizli-v6.6' # Testing bad pixels & weights
        root = 'abell2744clu-grizli-v6.7' # Testing bad pixels & weights, good to go?

        root = f'abell2744clu-grizli-{version}' # Testing bad pixels & weights, good to go?

        if mosaic_field == 'abell2744par':
            dim = (int(4*2048), int(6*2048)) # > 6.6 for the full DD field
    
            dx = dim[0] - hivo['NAXIS1']
            dy = dim[1] - hivo['NAXIS2']
            hivo['crpix1'] += dx//2 + 0.5
            hivo['crpix2'] += dy//2 + 0.5
    
            hivo['crpix1'] -= 2800
            hivo['crpix2'] -= 1024
            hivo['crpix2'] += 256+512 # > 6.6 for the full DD field

            hivo['crpix1'] += 4096*2+512 - 256
            hivo['crpix2'] += 4096*2+1024+512

            hivo['NAXIS1'] = dim[0]
            hivo['NAXIS2'] = dim[1]
        
            root = f'abell2744par-grizli-{version}' # Testing bad pixels & weights, good to go?
        
        ivo_wcs = pywcs.WCS(hivo)
        ref_file = pywcs.WCS(utils.to_header(ivo_wcs))

        half_sw = True
    
        if 0:
            # Slice for NIRISS field
            NP = 2048*16
            hivo = pyfits.Header.fromtextfile('abell2744.wcs.txt')
            dx = NP - hivo['NAXIS1']
            dy = NP - hivo['NAXIS2']
            hivo['crpix1'] += dx//2 + 0.5
            hivo['crpix2'] += dy//2 + 0.5
            hivo['NAXIS1'] = hivo['NAXIS2'] = NP
        
            xivo_wcs = pywcs.WCS(hivo)
        
            # Centered
            rc, dc = 3.4573691, -30.3660376
        
            xc, yc = np.asarray(np.round(np.squeeze(xivo_wcs.all_world2pix([rc], [dc], 0))),dtype=int)
            #print('xxx', xc, yc, NP)
        
            NX = 5*2048
            xivo_wcs_head = utils.get_wcs_slice_header(xivo_wcs, 
                                                       slice(xc-NX//2, xc+NX//2),
                                                       slice(yc-NX//2, yc+NX//2))
            root = 'abell2744nis-grizli-v5.1' # First UNCOVER visit and redo GLASS
            root = 'abell2744nis-grizli-v5.2' # UNCOVER visits 2,3,4
        
        
            # New pointing fully including GLASS + UNCOVER/NIS
            rc, dc = 3.4793842, -30.3562169
        
            xc, yc = np.asarray(np.round(np.squeeze(xivo_wcs.all_world2pix([rc], [dc], 0))),dtype=int)
            #print('xxx', xc, yc, NP)
        
            NX = 6*2048
            xivo_wcs_head = utils.get_wcs_slice_header(xivo_wcs, 
                                                       slice(xc-NX//2, xc+NX//2),
                                                       slice(yc-NX//2, yc+NX//2))

            root = 'abell2744par-grizli-v5.3' # UNCOVER + GLASS
        
            ref_file = pywcs.WCS(xivo_wcs_head)
            #ref_file = pywcs.WCS(utils.to_header(ivo_wcs))
    
    elif mosaic_field == 'primerv1':
    
        mos_path = 's3://grizli-v2/JwstMosaics/primer/v1'
        root = 'primer-grizli-v3.3'
        ref_file = 'primer-grizli-v1-f444w-clear_drc_sci.fits.gz'

    elif mosaic_field.startswith('primer_uds'):
    
        ###### PRIMER UDS
    
        # Split primer
        hdu = utils.make_wcsheader(ra=34.3438954, dec=-5.1506928, size=(409.600, 163.840),
                                   pixscale=0.04, theta=-74.9, get_hdu=True)

        hdu.header['CRPIX1'] = 10240//2
        hdu.header['CRPIX2'] = 4096//2

        root = 'primer-w-grizli-v4.0'
        ref_file = pywcs.WCS(hdu.header)

        if 1:
            hdu = utils.make_wcsheader(ra=34.4322866, dec=-5.1418212, size=(327.680, 163.840),
                                       pixscale=0.04, theta=-71.98, get_hdu=True)

            hdu.header['CRPIX1'] = 8192//2
            hdu.header['CRPIX2'] = 4096//2

            ref_file = pywcs.WCS(hdu.header)
            root = 'primer-e-grizli-v4.0'
    
        if 1:
            # PRIMER UDS full
            NX, NY = 15, 11
            hdu = utils.make_wcsheader(ra=34.372, dec=-5.21,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)
    
            hdu.header['CRPIX1'] = NX*2048//2
            hdu.header['CRPIX2'] = NY*2048//2
        
            half_sw = False
            root = 'primer-uds-grizli-v5.0'
            root = 'primer-uds-grizli-v6.0'

            root = f'primer-uds-grizli-{version}'

            ref_file = pywcs.WCS(hdu.header)

            if 0:
                w = pywcs.WCS(hdu.header)

                root = 'primer-uds-se-grizli-v5.0'
                h = utils.get_wcs_slice_header(w, slice(0, 30720//2+1024), slice(0, int(2048*6.5)))

                if 0:
                    root = 'primer-uds-sw-grizli-v5.0'
                    h = utils.get_wcs_slice_header(w, slice(30720//2-1024,30721), slice(0, int(2048*6.5)))

                ref_file = pywcs.WCS(h)

            # Slightly bigger for new visits, August 2023
            NX, NY = 15, 12
            hdu = utils.make_wcsheader(ra=34.372, dec=-5.21,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)
    
            hdu.header['CRPIX1'] = NX*2048//2
            hdu.header['CRPIX2'] = NY*2048//2
            hdu.header['CRPIX2'] -= 1024 - 200
            hdu.header['CRPIX1'] -= 256
        
            half_sw = False

            root = f'primer-uds-grizli-{version}'

            ref_file = pywcs.WCS(hdu.header)

            # Slice into halves
            if 'north' in mosaic_field:
                wh = utils.get_wcs_slice_header(ref_file, slice(0, NX*2048),
                                                slice(NY*2048//2-1024, NY*2048))
            
                ref_file = pywcs.WCS(wh)
                root = f'primer-uds-north-grizli-{version}'
            
            if 'south' in mosaic_field:
                wh = utils.get_wcs_slice_header(ref_file, slice(0, NX*2048),
                                                slice(0, NY*2048//2+1024))
            
                ref_file = pywcs.WCS(wh)
                root = f'primer-uds-south-grizli-{version}'
        
    elif mosaic_field == 'cweb_first':
        # CWeb first visits
        NX, NY = 9, 11

        hdu = utils.make_wcsheader(ra=149.945,  dec=2.372,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)
    
        root = 'cosweb-grizli-v6.0'
        
        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2

        ref_file = pywcs.WCS(hdu.header)
        
        half_sw = False
        print(root, hdu.data.shape)

    elif mosaic_field.startswith('primer_cosmos'):
        # primer COSMOS
        NX, NY = 13, 18
        #16.7' x 24.0' mosaics centered at (α, δ) = (150.139583, 2.333333)
        hdu = utils.make_wcsheader(ra=150.139583,  dec=2.333333,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)
    
        # Shift
        NX, NY = 9, 15
        #16.7' x 24.0' mosaics centered at (α, δ) = (150.139583, 2.333333)
    
        ra, dec = 150.12, 2.32 # 5.0

        NX, NY = 8.5, 15.5
        ra, dec = 150.119, 2.325 # 6.0

        hdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)

        if 1:
            # NIRCam
            NX, NY = 6, 11.5
    
            hdu = utils.make_wcsheader(ra=150.15,  dec=2.355,
                                       size=(NX*2048*0.04, NY*2048*0.04),
                                       pixscale=0.04, theta=0., get_hdu=True)
        
            # Epoch 2
            hdu = utils.make_wcsheader(ra=150.11,  dec=2.3,
                                       size=(NX*2048*0.04, NY*2048*0.04),
                                       pixscale=0.04, theta=0., get_hdu=True)

            # Full
            NX, NY = 9, 16
            hdu = utils.make_wcsheader(ra=150.125,  dec=2.325,
                                       size=(NX*2048*0.04, NY*2048*0.04),
                                       pixscale=0.04, theta=0., get_hdu=True)

            # Match UDS
            NX, NY = 9, 16
            hdu = utils.make_wcsheader(ra=150.125,  dec=2.325,
                                       size=(NX*2048*0.04, NY*2048*0.04),
                                       pixscale=0.04, theta=0., get_hdu=True)

        root = 'primer-cosmos-grizli-v5.0'
        root = 'primer-cosmos-grizli-v6.0'
        root = 'primer-cosmos-grizli-v6.1'
        root = 'primer-cosmos-grizli-v6.5' # with median_err weighting
    
        root = f'primer-cosmos-grizli-{version}' # with median_err weighting

        if 0:
            # Center on MIRI
            root = 'primer-cosmos-grizli-v5.0-miri'
            NX, NY = 6, 11.5

            hdu = utils.make_wcsheader(ra=150.07,  dec=2.27,
                                       size=(NX*2048*0.04, NY*2048*0.04),
                                       pixscale=0.04, theta=0., get_hdu=True)
    
        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2

        ref_file = pywcs.WCS(hdu.header)
        
        half_sw = False

        # Slice into halves
        if 'west' in mosaic_field:
            wh = utils.get_wcs_slice_header(ref_file, slice(NX*2048//2-1024, NX*2048), slice(0, NY*2048))
        
            ref_file = pywcs.WCS(wh)
            root = f'primer-cosmos-west-grizli-{version}'
        
        if 'east' in mosaic_field:
            wh = utils.get_wcs_slice_header(ref_file, slice(0, NX*2048//2+1024), slice(0, NY*2048))
        
            ref_file = pywcs.WCS(wh)
            root = f'primer-cosmos-east-grizli-{version}'

    elif mosaic_field == 'sunrise':
        root = 'sunrise-grizli-v3.3'
        mos_path = 's3://grizli-v2/JwstMosaics/whl0137/arc-v2'
        ref_file = 'sunrise-grizli-v2-f444w-clear_drc_sci.fits.gz'
    
        if not os.path.exists(ref_file):
            os.system(f'aws s3 cp {mos_path}/{ref_file} . ')
    
        hdu = pyfits.open(ref_file)[0]
    
        NX, NY = 4, 7

        hdu = utils.make_wcsheader(ra=24.355, dec=-8.457,
                                       size=(NX*2048*0.04, NY*2048*0.04),
                                       pixscale=0.04, theta=0., get_hdu=True)

        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2

        w = pywcs.WCS(hdu.header)
        ref_file = w

        half_sw = True

        root = f'sunrise-grizli-{version}'
    
        ref_file = pywcs.WCS(hdu.header)

    elif mosaic_field == 'j1235':
    
        root = 'j1235-grizli-v4.0'
        mos_path = 's3://grizli-v2/JwstMosaics/j1235/v0'
        ref_file = 'j1235-f444w-clear_drc_sci.fits.gz'        
    
        NX, NY, ra, dec = 6, 10, 188.960, 4.9535

        hdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                       size=(NX*2048*0.04, NY*2048*0.04),
                                       pixscale=0.04, theta=0., get_hdu=True)

        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
        w = pywcs.WCS(hdu.header)

        root = f'j1235-grizli-{version}'

        sr = utils.SRegion(w)
        print(sr.region[0])
        
        print(root, hdu.data.shape, w.pixel_shape)
    
        ref_file = w

    elif mosaic_field == 'sgas1723':
    
        mos_path = 's3://grizli-v2/JwstMosaics/v3'
        root = 'sgas1723-grizli-v4.0'
        ref_file = 'sgas1723-grizli-v3.3-f444w-clear_drc_sci.fits.gz'        
    
        #######
        NX, NY, ra, dec = 3, 3, 260.904, 34.197

        hdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                       size=(NX*2048*0.04, NY*2048*0.04),
                                       pixscale=0.04, theta=0., get_hdu=True)

        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
        w = pywcs.WCS(hdu.header)

        root = f'sgas1723-grizli-{version}'

        sr = utils.SRegion(w)
        print(sr.region[0])
        
        print(root, hdu.data.shape, w.pixel_shape)
    
        ref_file = w

    elif mosaic_field == 'nep_2mass':
        mos_path = 's3://grizli-v2/JwstMosaics/nep-2mass'
        root = 'nep-2mass-grizli-v4.0'
        ref_file = 'nep-2mass-f444w-clear_drc_sci.fits.gz'        

    elif mosaic_field == 'jwst_nep':
        # windhorst NEP
        hdu = utils.make_wcsheader(ra=260.7226654, dec=65.7640067,
                                   size=(8.192*60, 2.7306667*60),
                                   pixscale=0.04, theta=-107, get_hdu=True)

        hdu.header['CRPIX1'] = 12288//2
        hdu.header['CRPIX2'] = 4096//2

        w = pywcs.WCS(hdu.header)
        ref_file = w
        root = 'jwst-nep-tdf-nrc-grizli-v4.0'
        root = 'jwst-nep-tdf-nrc-grizli-v6.0'
        root = f'jwst-nep-tdf-nrc-grizli-{version}'
    
        half_sw = True
    
        if 0:
            # NIRISS
            hdu = utils.make_wcsheader(ra=260.6513919, dec=65.8580014,
                                       size=(5.46133*60, 2.7306667*60),
                                       pixscale=0.04, theta=-107, get_hdu=True)

            hdu.header['CRPIX1'] = 4096
            hdu.header['CRPIX2'] = 4096//2

            w = pywcs.WCS(hdu.header)
            ref_file = w
            root = 'jwst-nep-tdf-nis-grizli-v4.0'
            half_sw = False

    elif mosaic_field == 'macs0647':
        # MACS 0647
        NX, NY = 7, 5
    
        ra=101.9482378; dec=70.2297032
    
        hdu = utils.make_wcsheader(ra=ra, dec=dec,
                                       size=(NX*2048*0.04, NY*2048*0.04),
                                       pixscale=0.04, theta=0., get_hdu=True)

        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2

        w = pywcs.WCS(hdu.header)
        ref_file = w
        root = 'macs0647-grizli-v4.0'
        root = 'macs0647-grizli-v5.0'
        root = 'macs0647-grizli-v6.0'
        root = f'macs0647-grizli-{version}'

        half_sw = True

    elif mosaic_field == 'rxcj0600':
    
        NX, NY = 5, 5

        hdu = utils.make_wcsheader(ra=90.04, dec=-20.145,
                                       size=(NX*2048*0.04, NY*2048*0.04),
                                       pixscale=0.04, theta=0., get_hdu=True)

        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2

        w = pywcs.WCS(hdu.header)
        ref_file = w
        root = 'rxcj0600-grizli-v5.0'
        root = f'rxcj0600-grizli-{version}'
    
        half_sw = True
    
    elif mosaic_field == 'macs0417':
        # MACS0417
    
        # Full 
        ra, dec, size = 64.3958333, -11.9091667, (20*60, 20*60)
        fhdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                   size=size,
                                   pixscale=0.04, theta=0., get_hdu=True)
        # full_wcs = 
        fhdu.header['CRPIX1'] = fhdu.header['NAXIS1']//2
        fhdu.header['CRPIX2'] = fhdu.header['NAXIS2']//2
        full_wcs = pywcs.WCS(fhdu.header)
    
        NX, NY, ra, dec, root = 5, 6, 64.399172, -11.898611, 'macs0417clu'
        NX, NY, ra, dec, root = 4, 5, 64.364162, -11.815276, 'macs0417ncf'
        #NX, NY, ra, dec, root = 4, 4, 64.440418, -11.994024, 'macs0417nsf'
    
        root += '-v3'
    
        ########
    
        if canucs_subfield == 'clu':
            NX, NY, ra, dec, root = 5, 6, 64.399172, -11.898611, 'macs0417-clu'
        elif canucs_subfield == 'ncf':
            NX, NY, ra, dec, root = 4, 5, 64.364162, -11.815276, 'macs0417-ncf'
        else:
            NX, NY, ra, dec, root = 4, 4, 64.440418, -11.994024, 'macs0417nsf'
    
        root += f'-grizli-{version}'
    
        hdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)
    
        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
        w = pywcs.WCS(hdu.header)
    
        # Slices
        llr, lld = w.all_pix2world([0],[0], 0)
        ll = np.squeeze(full_wcs.all_world2pix(llr, lld, 0))
        llx, lly = np.asarray(np.floor(ll),dtype=int)
    
        wh = utils.get_wcs_slice_header(full_wcs,
                                        slice(llx, llx+NX*2048),
                                        slice(lly, lly+NY*2048))
        w = pywcs.WCS(wh)
    
        print(root, fhdu.data.shape, llx, lly, w.pixel_shape)
    
        ref_file = w
        half_sw = True
        del(fhdu)
    
    elif mosaic_field == 'rxj2129':
        # RXJ2129
        
        NX, NY, ra, dec, root = 4, 4, 322.4198400, 0.0923168, 'rxj2129-grizli-v4.0'

        hdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)
    
        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
        w = pywcs.WCS(hdu.header)
        
        print(root, hdu.data.shape, w.pixel_shape)

        root = f'rxj2129-grizli-{version}'
    
        ref_file = w

    elif mosaic_field == 'hudf':
        # HUDF around Williams GO-1963 pointing
        
        NX, NY, ra, dec, root = 5, 11, 53.1421273, -27.8047896, 'hudf-grizli-v4.0'

        hdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)
    
        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
        w0 = pywcs.WCS(hdu.header)

        if 1:
            # South region around NIRISS pointing
            hsl = utils.get_wcs_slice_header(w0, slice(2048,(NX-1)*2048), slice(0,3*2048+256))
            w = pywcs.WCS(hsl)
            root = 'hudf-nis-grizli-v4.0'
        else:
            # Larger field around the HUDF proper that should include both 
            # NIRCam modules from GO-1963
            hsl = utils.get_wcs_slice_header(w0, slice(0,NX*2048), slice(3*2048, 8*2048))
            w = pywcs.WCS(hsl)
            root = 'hudf-nrc-grizli-v5.0'
    
        if 1:
            # Shift for FRESCO
            NX, NY, ra, dec = 6, 6.5, 53.128, -27.8034
        
            # Bigger for GTO
            NX, NY, ra, dec = 7.0, 7.0, 53.142, -27.798

            hdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                       size=(NX*2048*0.04, NY*2048*0.04),
                                       pixscale=0.04, theta=0., get_hdu=True)

            hdu.header['CRPIX1'] = NX*2048//2
            hdu.header['CRPIX2'] = NY*2048//2
        
            root = 'gds-grizli-v5.0'
            root = 'gds-grizli-v5.1' # with full FRESCO and reprocessing in SW
            root = 'gds-grizli-v6.0' # with full FRESCO and reprocessing in SW
            root = 'gds-grizli-v6.1' # with full FRESCO and reprocessing in SW
            root = 'gds-grizli-v6.6' # re-weighting

            # hdu.header['CRPIX1'] += int(2048*2.5)
            # hdu.header['NAXIS1'] += int(2048*2.5)

            # hdu.header['CRPIX2'] += int(2048*0.5)
            # hdu.header['NAXIS2'] += int(2048*0.5)

            root = f'gds-grizli-{version}'
            # root = 'gds-grizli-v6.6' # include ngdeep

            w = pywcs.WCS(hdu.header)

        sr = utils.SRegion(w)
        print(sr.region[0])
        
        print(root, hdu.data.shape, w.pixel_shape)
    
        ref_file = w

    elif mosaic_field == 'gds-sw':    
        # SW corner with additional GTO data
        NX, NY, ra, dec = 7.0, 7.0, 53.093, -27.87

        hdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)

        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
    
        root = f'gds-sw-grizli-{version}'

        w = pywcs.WCS(hdu.header)

        sr = utils.SRegion(w)
        print(sr.region[0])
        
        print(root, hdu.data.shape, w.pixel_shape)
    
        ref_file = w

    elif mosaic_field == 'gdn':
    
        # FRESCO-N
        NX, NY, ra, dec = 8.5, 8.5, 189.195, 62.247

        hdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)

        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
        w = pywcs.WCS(hdu.header)

        root = 'gdn-grizli-v5.1' # with full FRESCO and reprocessing in SW

        root = 'gdn-grizli-v6.0' # uniform processing, flats, etc.
        root = 'gdn-grizli-v6.5' # new weighting, bit of panoramic
        
        root = f'gdn-grizli-{version}' # new weighting, bit of panoramic

        # Shift to include GTO data
        NX, NY = 10, 14
        hdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)

        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2    
        w = pywcs.WCS(hdu.header)

        h = utils.get_wcs_slice_header(w, slice(1024, NX*2048),
                                       slice(1*2048, int((NY-2.8)*2048)))
        
        w = pywcs.WCS(h)

        sr = utils.SRegion(w)
        print(sr.region[0])
        
        print(root, hdu.data.shape, w.pixel_shape)
    
        ref_file = w
        half_sw = True
    
    elif mosaic_field == 'ngdeep':
        # NGDEEP
        NX, NY = 4, 5
    
        hdu = utils.make_wcsheader(ra=53.2526057, dec=-27.8393697,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)

        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
    
        # Include GTO data
        NX, NY = 7, 7
        hdu = utils.make_wcsheader(ra=53.2526057, dec=-27.8393697,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)

        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2

        w = pywcs.WCS(hdu.header)

        h = utils.get_wcs_slice_header(w, slice(400,NX*2048-2048-1024-512-200 - 300), slice(64,NY*2048-2048+512+256 - 100 - 1024))
    
        w = pywcs.WCS(h)
    
        w = pywcs.WCS(hdu.header)
        root = 'ngdeep-grizli-v5.1'
        root = 'ngdeep-grizli-v6.0'
        root = 'ngdeep-grizli-v6.1'
        root = 'ngdeep-grizli-v6.2' # test new flats
        root = f'ngdeep-grizli-{version}' # test new flats

        print(root, hdu.data.shape, w.pixel_shape)
    
        ref_file = w

    elif mosaic_field == 'miri-gto':

        # WCS from Steven Gillman
        
        import astropy.io.fits as pyfits
        head = pyfits.Header.fromstring("""WCSAXES =                    2 / Number of coordinate axes                      
CRPIX1  =           2271.35596 / Pixel coordinate of reference point            
CRPIX2  =           2028.36145 / Pixel coordinate of reference point            
CD1_1   =    -0.88429963653668 / Coordinate transformation matrix element       
CD1_2   =     0.46691985695738 / Coordinate transformation matrix element       
CD2_1   =     0.46691985695738 / Coordinate transformation matrix element       
CD2_2   =     0.88429963653668 / Coordinate transformation matrix element       
CDELT1  =  1.1111111111111E-05 / [deg] Coordinate increment at reference point  
CDELT2  =  1.1111111111111E-05 / [deg] Coordinate increment at reference point  
CUNIT1  = 'deg'                / Units of coordinate increment and value        
CUNIT2  = 'deg'                / Units of coordinate increment and value        
CTYPE1  = 'RA---TAN'           / Right ascension, gnomonic projection           
CTYPE2  = 'DEC--TAN'           / Declination, gnomonic projection               
CRVAL1  =           53.1682472 / [deg] Coordinate value at reference point      
CRVAL2  =          -27.7826748 / [deg] Coordinate value at reference point      
LONPOLE =                180.0 / [deg] Native longitude of celestial pole       
LATPOLE =          -27.7826748 / [deg] Native latitude of celestial pole        
MJDREF  =                  0.0 / [d] MJD of fiducial time                       
DATE-BEG= '2022-12-02T11:57:26.247' / ISO-8601 time at start of observation     
MJD-BEG =      59915.498220451 / [d] MJD at start of observation                
DATE-AVG= '2022-12-05T13:08:53.193' / ISO-8601 time at midpoint of observation  
MJD-AVG =      59918.547837878 / [d] MJD at midpoint of observation             
DATE-END= '2022-12-20T14:25:55.148' / ISO-8601 time at end of observation       
MJD-END =      59933.601332731 / [d] MJD at end of observation                  
XPOSURE =           148842.028 / [s] Exposure (integration) time                
TELAPSE =           148842.028 / [s] Elapsed time (start to stop)               
OBSGEO-X=     -136146536.01846 / [m] observatory X-coordinate                   
OBSGEO-Y=      1593184880.7529 / [m] observatory Y-coordinate                   
OBSGEO-Z=      397800097.00595 / [m] observatory Z-coordinate                   
RADESYS = 'ICRS'               / Equatorial coordinate system                   
VELOSYS =             11393.68 / [m/s] Velocity towards source                  
NAXIS   =                    2                                                  
NAXIS1  =                 3841                                                  
NAXIS2  =                 4049                                                  """, sep='\n')

        for i in [1,2]:
            for j in [1,2]:
                head[f'CD{i}_{j}'] *= head['CDELT1']

        head['CDELT1'] = 1.
        head['CDELT2'] = 1.
        
        w = ref_file = pywcs.WCS(head)
        root = f'gds-miri-hudf-grizli-{version}'
        print(root, w.pixel_shape)
        half_sw = False
    
    elif mosaic_field == 'smacs0723':
        # SMACS0723
        ra, dec = 110.75, -73.47
    
        NX, NY = 4,4
    
        hdu = utils.make_wcsheader(ra=ra, dec=dec,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)

        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
    
        if not os.path.exists('smacs0723-grizli-v4.0-f444w-clear_drc_sci.fits.gz'):
            os.system('wget "https://s3.amazonaws.com/grizli-v2/JwstMosaics/v4/smacs0723-grizli-v4.0-f444w-clear_drc_sci.fits.gz" -O smacs0723-grizli-v4.0-f444w-clear_drc_sci.fits.gz')
       
        hdu = pyfits.open('smacs0723-grizli-v4.0-f444w-clear_drc_sci.fits.gz')[0]
    
        w = pywcs.WCS(hdu.header)
        root = 'smacs0723-grizli-v6.0'
        root = f'smacs0723-grizli-{version}'
        
        print(root, hdu.data.shape, w.pixel_shape)
    
        ref_file = w

    elif mosaic_field == 'abell370':
        # CANUCS Abell 370
        NX, NY = 12,8
        
        # Bigger for making slices
        NX, NY = 30, 30
    
        hdu = utils.make_wcsheader(ra=39.9863750, dec=-1.5969444,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)

        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
    
    
        full_wcs = pywcs.WCS(hdu.header)

        if canucs_subfield == 'ncf':
            r0, d0, nnx, nny, fi = 40.06, -1.623, 2, 2.5, 'ncf'
        elif canucs_subfield == 'nsf':
            r0, d0, nnx, nny, fi = 39.90, -1.55, 1.5, 1.5, 'nsf'
        else:
            r0, d0, nnx, nny, fi = 39.98, -1.594, 2, 2.5, 'clu'

        if canucs_subfield == 'full':
            r0, d0, nnx, nny, fi = 40.022500, -1.643395, 3.5, 4.5, 'full'

        #nsf = 39.90, -1.540
        #clu = 39.9863750, -1.5969444
    
        root = f'abell370-{fi}-grizli-v5.0'
        root = f'abell370-{fi}-grizli-v6.0'
        root = f'abell370-{fi}-grizli-{version}' 

        xi, yi = np.asarray(np.squeeze(full_wcs.all_world2pix([r0], [d0], 0)),dtype=int)
    
        wh = utils.get_wcs_slice_header(full_wcs,
                                        slice(xi-int(nnx*2048), xi+int(nnx*2048)),
                                        slice(yi-int(nny*2048), yi+int(nny*2048)))
        w = pywcs.WCS(wh)
    
        print(root, hdu.data.shape, w.pixel_shape)
    
        ref_file = w

    elif mosaic_field == 'macs0416':
    
        # CANUCS MACS 0416
        NX, NY = 12,10
        NX, NY = 20, 20
        
        hdu = utils.make_wcsheader(ra=64.0566667, dec=-24.0897222,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)

        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
    
    
        full_wcs = pywcs.WCS(hdu.header)
    
        if canucs_subfield == 'nsf':
            r0, d0, nnx, nny, fi = 63.95, -24.069, 1.5, 1.5, 'nsf'
        elif canucs_subfield == 'ncf':
            r0, d0, nnx, nny, fi = 64.144, -24.096, 2, 2.5, 'ncf'
        else:
            r0, d0, nnx, nny, fi = 64.046, -24.084, 2.5, 2.5, 'clu'
        
        if canucs_subfield == 'full':
            r0, d0, nnx, nny, fi =  64.098600,  -24.135550, 4, 4.5, 'full'
    
        root = f'macs0416-{fi}-grizli-v5.0'
        root = f'macs0416-{fi}-grizli-v6.0'
        root = f'macs0416-{fi}-grizli-{version}'

        xi, yi = np.asarray(np.squeeze(full_wcs.all_world2pix([r0], [d0], 0)),dtype=int)
    
        wh = utils.get_wcs_slice_header(full_wcs,
                                        slice(xi-int(nnx*2048), xi+int(nnx*2048)),
                                        slice(yi-int(nny*2048), yi+int(nny*2048)))
        w = pywcs.WCS(wh)
    
        # w = full_wcs
    
        print(root, hdu.data.shape, w.pixel_shape)
    
        ref_file = w
    
    elif mosaic_field == 'macs1423':
    
        # MACS1423
        ra, dec = 215.9500000, 24.0811111
    
        NX, NY = 12,10
    
        hdu = utils.make_wcsheader(ra=ra, dec=dec,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)

        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
    
    
        full_wcs = pywcs.WCS(hdu.header)
    
        if canucs_subfield == 'nsf':
            r0, d0, nnx, nny, fi = 63.95, -24.069, 1.5, 1.5, 'nsf'
        elif canucs_subfield == 'ncf':
            r0, d0, nnx, nny, fi = 215.8747807, 24.1489968, 1.5, 2., 'ncf'
        elif canucs_subfield == 'clu':
            r0, d0, nnx, nny, fi = 215.9448026, 24.1006250, 1.5, 2.5, 'clu'
        else:
            r0, d0, nnx, nny, fi = 215.909, 24.123, 3.0, 3.5, 'full'

        root = f'macs1423-{fi}-grizli-v5.0'
        root = f'macs1423-{fi}-grizli-v5.2' # updated photometric calibration

        root = f'macs1423-{fi}-grizli-v6.0' # stripe fixes, etc.
        root = f'macs1423-{fi}-grizli-{version}'

        xi, yi = np.asarray(np.squeeze(full_wcs.all_world2pix([r0], [d0], 0)),dtype=int)
    
        wh = utils.get_wcs_slice_header(full_wcs,
                                        slice(xi-int(nnx*2048), xi+int(nnx*2048)),
                                        slice(yi-int(nny*2048), yi+int(nny*2048)))
        w = pywcs.WCS(wh)
    
        # w = full_wcs
    
        print(root, hdu.data.shape, w.pixel_shape)
    
        ref_file = w

    elif mosaic_field == 'macs1149':
        ra, dec = 215.9500000, 24.0811111
        ra, dec = 177.3910, 22.3825
        ra, dec = 177.3910, 22.33

        NX, NY = 12,12
    
        hdu = utils.make_wcsheader(ra=ra, dec=dec,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)

        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
    
    
        full_wcs = pywcs.WCS(hdu.header)
    
        if canucs_subfield == 'nsf':
            r0, d0, nnx, nny, fi = 63.95, -24.069, 1.5, 1.5, 'nsf'
        elif canucs_subfield == 'ncf':
            r0, d0, nnx, nny, fi = 215.8747807, 24.1489968, 1.5, 2., 'ncf'
        elif canucs_subfield == 'clu':
            r0, d0, nnx, nny, fi = 177.3910, 22.3825, 2.0, 2.5, 'clu'

            # Single mosaic?
            r0, d0, nnx, nny, fi = 177.395, 22.34, 2.5, 4, 'clu'

        else:
            r0, d0, nnx, nny, fi = 177.395, 22.34, 2.5, 4, 'full'
            r0, d0, nnx, nny, fi = 177.395, 22.34, 2.5, 4, 'clu'

        # r0, d0, nnx, nny, fi = 215.909, 24.123, 3.0, 3.5, 'full'

    #     root = f'macs1423-{fi}-grizli-v5.0'
    #     root = f'macs1423-{fi}-grizli-v5.2' # updated photometric calibration

        root = f'macs1149-{fi}-grizli-v6.0' # stripe fixes, etc.
        root = f'macs1149-{fi}-grizli-v6.1' # Chris level 1
        root = f'macs1149-{fi}-grizli-{version}'

        xi, yi = np.asarray(np.squeeze(full_wcs.all_world2pix([r0], [d0], 0)),dtype=int)
    
        wh = utils.get_wcs_slice_header(full_wcs,
                                        slice(xi-int(nnx*2048), xi+int(nnx*2048)),
                                        slice(yi-int(nny*2048), yi+int(nny*2048)))
        w = pywcs.WCS(wh)
    
        # w = full_wcs
    
        print(root, hdu.data.shape, w.pixel_shape)
    
        ref_file = w
    
    elif mosaic_field == 'abells1063':

        root = 'abells1063'
        
        NX, NY, ra, dec = 3, 3, 342.1839985, -44.5308919

        hdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                       size=(NX*2048*0.04, NY*2048*0.04),
                                       pixscale=0.04, theta=0., get_hdu=True)

        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
        w = pywcs.WCS(hdu.header)

        root = f'abells1063-grizli-{version}'

        sr = utils.SRegion(w)
        print(sr.region[0])
        
        print(root, hdu.data.shape, w.pixel_shape)
    
        ref_file = w

    elif mosaic_field == 'j061548m5745':

        root = 'j061548m5745'
        
        NX, NY, ra, dec = 4, 5, 93.955650, -57.757050

        hdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                       size=(NX*2048*0.04, NY*2048*0.04),
                                       pixscale=0.04, theta=0., get_hdu=True)

        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
        w = pywcs.WCS(hdu.header)

        root = f'j061548m5745-grizli-{version}'

        sr = utils.SRegion(w)
        print(sr.region[0])
        
        print(root, hdu.data.shape, w.pixel_shape)
    
        ref_file = w
    
    elif mosaic_field == 'panoramic':
    
        # PANORAMIC
    
        # visit 2
        ra, dec = 233.7421818, 23.4198190
        xroot = 'panoramic-j153500p2325'
        NX, NY = 2., 4
    
        # Visit 1
        ra, dec = 256.8404526, 58.8787615
        xroot = 'panoramic-j170720p5853'
        NX, NY = 3, 4.5

        hdu = utils.make_wcsheader(ra=ra, dec=dec,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)

        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
    
        w = pywcs.WCS(hdu.header)
    
        root = xroot + '-grizli-v6.0'
    
        print(root, hdu.data.shape, w.pixel_shape)
    
        ref_file = w
    
    elif mosaic_field == 'elgordo':

        # El Gordo cluster, Windhorst
        NX, NY, ra, dec = 3, 5, 15.7360648, -49.2355885

        hdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)
    
        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
        w = pywcs.WCS(hdu.header)
        
        root = f'elgordo-grizli-{version}'

        print(root, hdu.data.shape, w.pixel_shape)

        ref_file = w
    
    elif (mosaic_field in ['snh0pe', 'j112716p4228']):
        
        NX, NY, ra, dec = 4, 4, 171.816, 42.4622

        hdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)
        
        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
        w = pywcs.WCS(hdu.header)
        
        root = f'j112716p4228-grizli-{version}'

        print(root, hdu.data.shape, w.pixel_shape)

        ref_file = w
        
    elif (mosaic_field in ['spitzer_idf']):
        
        NX, NY, ra, dec = 5, 3.5, 265.0347875, 68.9741119
        
        hdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)
        
        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
        w = pywcs.WCS(hdu.header)
        
        root = f'spitzer_idf-grizli-{version}'

        print(root, hdu.data.shape, w.pixel_shape)

        ref_file = w

    elif mosaic_field == 'mrg0138':

        # MRG0138 Newman et al.
        NX, NY, ra, dec = 3, 3, 24.522, -21.93

        hdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)
    
        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
        w = pywcs.WCS(hdu.header)
        
        root = f'mrg0138-grizli-{version}'

        print(root, hdu.data.shape, w.pixel_shape)

        ref_file = w

    elif mosaic_field == 'ulasj1342':

        # ULAS-J1342
        NX, NY, ra, dec = 4.5, 7.5, 205.555, 9.462

        hdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)
    
        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
        w = pywcs.WCS(hdu.header)
        
        root = f'ulasj1342-grizli-{version}'

        print(root, hdu.data.shape, w.pixel_shape)

        ref_file = w

    elif mosaic_field == 'sunburst':

        # ULAS-J1342
        NX, NY, ra, dec = 4, 4, 237.5297488, -78.1917476

        hdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)
    
        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
        w = pywcs.WCS(hdu.header)
        
        root = f'sunburst-grizli-{version}'

        print(root, hdu.data.shape, w.pixel_shape)

        ref_file = w

    elif mosaic_field == 'cosmos-transients':
        NX, NY, ra, dec = 5, 5, 150.1250419, 2.3661324
        
        hdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)
    
        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
        w = pywcs.WCS(hdu.header)
        
        root = f'cosmos-transients-{version}'

        print(root, hdu.data.shape, w.pixel_shape)

        ref_file = w

    ## PEARLS fields
    elif mosaic_field == 'gama100033':
        NX, NY, ra, dec = 3, 5, 130.58659471, 1.61764667
        
        hdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)
    
        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
        w = pywcs.WCS(hdu.header)
        
        root = f'gama100033-{version}'

        print(root, hdu.data.shape, w.pixel_shape)

        ref_file = w

    elif mosaic_field == 'clg-1212p2733':
        NX, NY, ra, dec = 3, 5, 183.08573496, 27.57669020
        
        hdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)
    
        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
        w = pywcs.WCS(hdu.header)
        
        root = f'clg-1212p2733-{version}'

        print(root, hdu.data.shape, w.pixel_shape)

        ref_file = w

    elif mosaic_field == 'plck-g165p67':
        NX, NY, ra, dec = 5, 4, 171.78782682, 42.47523529
        
        hdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)
    
        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
        w = pywcs.WCS(hdu.header)
        
        root = f'plck-g165p67-{version}'

        print(root, hdu.data.shape, w.pixel_shape)

        ref_file = w

    elif mosaic_field == 'plck-g191p62':
        NX, NY, ra, dec = 5, 5, 161.16195126, 33.83078874
        
        hdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)
    
        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
        w = pywcs.WCS(hdu.header)
        
        root = f'plck-g191p62-{version}'

        print(root, hdu.data.shape, w.pixel_shape)

        ref_file = w    

    elif mosaic_field == 'tn-j1338m1942':
        NX, NY, ra, dec = 4, 5, 204.61036521, -19.67247443
        
        hdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)
    
        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
        w = pywcs.WCS(hdu.header)
        
        root = f'tn-j1338m1942-{version}'

        print(root, hdu.data.shape, w.pixel_shape)

        ref_file = w    

    elif mosaic_field == 'abell1689':
        NX, NY, ra, dec = 4, 4, 197.8765, -1.33
        
        hdu = utils.make_wcsheader(ra=ra,  dec=dec,
                                   size=(NX*2048*0.04, NY*2048*0.04),
                                   pixscale=0.04, theta=0., get_hdu=True)
    
        hdu.header['CRPIX1'] = NX*2048//2
        hdu.header['CRPIX2'] = NY*2048//2
        w = pywcs.WCS(hdu.header)
        
        root = f'abell1689-grizli-{version}'

        print(root, hdu.data.shape, w.pixel_shape)

        ref_file = w    

    else:
        raise ValueError(f'Field {mosaic_field} not defined')
    
    if isinstance(ref_file, str):
        print(root, ref_file, half_sw, s3output)
        if not os.path.exists(ref_file):
            os.system(f'aws s3 cp {mos_path}/{ref_file} .')
        
        im = pyfits.open(ref_file)
        ref_wcs = pywcs.WCS(im[0].header)
    else:
        ref_wcs = ref_file
        print(root, 'WCS', half_sw, s3output, ref_wcs.pixel_shape)
    
    return root, half_sw, s3output, ref_wcs


def show_field_footprint(mosaic_field='primer_uds', filters="'F444W-CLEAR','F770W'", extra="", output_path='./', version='v7.0', canucs_subfield='clu'):
    """
    Make a figure showing the a field footprint and exposure overlaps
    
    Parameters
    ----------
    mosaic_field : str
        Field name
    
    filters : str
        List of filter names 
    
    extra : str
        Extra SQL query parameters on the `exposure_files` table
    
    output_path : str
        Path where figure will be saved
    
    Returns
    -------
    fig : `matplotlib.Figure`
        Figure object
    
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from grizli_aws import field_tiles
    from grizli.aws import db
    
    root, half_sw, s3output, ref_wcs = make_wcs(mosaic_field=mosaic_field,
                                                version=version,
                                                canucs_subfield=canucs_subfield,
                                                )
    ref_file = ref_wcs
    #output_path = os.getcwd()
    
    tile_npix = 4096 + 512
    pscale = 0.066
    pscale = 0.040 # / 80 / 40 / 20 mas

    tile_arcmin = tile_npix*pscale/60
    print('Tile size, arcmin = ', tile_arcmin)

    tiles = field_tiles.define_tiles(ra=3.5875000, dec=-30.3966667, 
                                     size=(20,20), 
                                     tile_size=tile_arcmin, 
                                     overlap=512*pscale/60, field=root, 
                                     pixscale=0.04, theta=0)

    # if '2744' in root:
    #     pt = 'point(3.585, -30.396)'

    # elif root[:3] in ['gds','hud']:
    #     pt = 'point(53.1239038,-27.7980685)'
    #     ivo_orig = ref_file
    # else:
    if 1:
        pt = f'point({ref_file.wcs.crval[0]}, {ref_file.wcs.crval[1]})'
        ivo_orig = ref_file
    
    # filters = """'F430M-CLEAR','F444W-CLEAR','F480M-CLEAR','F200W','F770W',
    #                   'CLEARP-F444W','F356W-CLEAR','CLEARP-F430M','CLEARP-F480M'"""
    # filters = """'F444W-CLEAR','F770W'"""

    #extra = "AND dataset like 'jw01837%%'"
    #extra = ""

    glass_fp = db.SQL(f"""
    select file, filter, instrume, footprint from exposure_files
    where (filter in ({filters})
                      OR instrume in ('NIRISS')) {extra}
    AND polygon(footprint) && polygon(circle({pt}, 1.0))
    """)
    
    nx, ny = ref_wcs.pixel_shape
    
    dims = np.array([1, ny/nx])
    dims = np.array([1,1])
    
    fig, ax = plt.subplots(1,1,figsize=8*dims/dims.max())

    xy = []

    for w in [ivo_orig, ref_file][1:]:
        wfp = w.calc_footprint().T
        xy.append(wfp)
    
        pl = plt.plot(*wfp)
        sr = utils.SRegion(wfp)
        for p in sr.patch(fc=pl[0].get_color(), ec='None', alpha=0.05, zorder=-10,
                          label=root):
            ax.add_patch(p)
        
        ax.legend()
        
        plt.scatter(*w.wcs.crval, color=pl[0].get_color())
        print(w.wcs.crval, w.wcs.crpix)

    if ('primer_uds' in mosaic_field) | (1):
        aa = db.SQL(f"""select footprint from assoc_table
        where polygon(circle({pt}, 1.0)) @> point(ra, dec)
        AND filter in ('F444W-CLEAR')
        AND proposal_id in ('1837')
        """)
        for frp in aa['footprint']:
            sr = utils.SRegion(frp)
            for p in sr.patch(fc='r', ec='None', alpha=0.05, zorder=-10):
                ax.add_patch(p)
        
    xy = np.mean(np.hstack(xy), axis=1)
    print(xy)
    plt.scatter(*xy, marker='x')
      
    # for k in tiles:
    #     w = tiles[k]
    #     plt.plot(*w.calc_footprint().T, color='k', alpha=0.1)
    
    # ax = plt.gca()

    for fp, fi in zip(glass_fp['footprint'], glass_fp['filter']):
        sr = utils.SRegion(fp)
        if fi == 'F160W':
            c = 'skyblue'
        else:
            c = '0.5'
        
        for p in sr.patch(ec=c, fc='None',alpha=0.5):
            ax.add_patch(p)
        
    ax.set_aspect(1./np.cos(-ref_file.wcs.crval[1]/180*np.pi))
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.grid()
    
    fig.tight_layout(pad=1)
    
    fig.savefig(os.path.join(output_path, f'{root}_footprint.png'))

    return fig

