"""
Scripts to combine FLT exposures at a single orientation / sub-pixel position
to speed up the spectral processing.
"""
import copy
import glob

import numpy as np

import matplotlib.pyplot as plt

import astropy.wcs as pywcs
import astropy.io.fits as pyfits

from . import utils
from . import GRIZLI_PATH

def combine_flt(files=[], output='exposures_cmb.fits', grow=1,
                add_padding=True, pixfrac=0.5, kernel='point',
                verbose=True, clobber=True, ds9=None):
    """Drizzle distorted FLT frames to an "interlaced" image
        
    Parameters
    ----------
    files : list of strings
        Filenames of FLT files to combine
    
    output : str
        Output filename of the combined file.  Convention elsewhere is to use
        an "_cmb.fits" extension to distinguish from "_flt.fits".
        
    grow : int
        Factor by which to `grow` the FLT frames to interlaced outputs.  For 
        example, `grow=2` results in 2x2 interlacing.
    
    add_padding : True
        Expand pixel grid to accommodate all dithered exposures.  WCS is 
        preserved but "CRPIX" will change.
        
    pixfrac : float
        Drizzle pixfrac (for kernels other than 'point')
    
    kernel : {'point', 'square'}
        Drizzle kernel. The 'point' kernel is effectively interlacing and is
        best for preserving the noise properties of the final combined image.
        However, can result in empty pixels given the camera distortions
        depending on the dithering of the input exposures.
    
    ds9 : `~grizli.ds9.DS9`
        Display the progress of the script to a DS9 window.
        
    verbose : bool
        Print logging information
        
    clobber : bool
        Overwrite existing files
    
    Returns
    -------
    Creates combined images
    
    """
    import numpy.linalg
    from stsci.tools import asnutil
    from drizzlepac import astrodrizzle
    
    ###  `files` is an ASN filename, not  a list of exposures
    if '_asn.fits' in files:
        asn = asnutil.readASNTable(files)
        files = ['{0}_flt.fits'.format(flt) for flt in asn['order']]
        if output == 'combined_flt.fits':
            output = '{0}_cmb.fits'.format(asn['output'])
            
    if False:
        files=glob.glob('ibhj3*flt.fits')
        files.sort()
        grism_files = files[1::2]

        ### FIGS
        info = catIO.Table('info')
        pas = np.cast[int](info['PA_V3']*10)/10.
        pa_list = np.unique(pas)
        grism_files = info['FILE'][(info['FILTER'] == 'G102') & 
                                   (pas == pa_list[0])]
        
        files = grism_files
        #utils = grizlidev.utils
        
    f0 = pyfits.open(files[0])
    h0 = f0[0].header.copy()
    h0['EXPTIME'] = 0.
    h0['NFILES'] = (len(files), 'Number of combined files')
    
    out_wcs = pywcs.WCS(f0[1].header, relax=True)
    out_wcs.pscale = utils.get_wcs_pscale(out_wcs)
    # out_wcs.pscale = np.sqrt(out_wcs.wcs.cd[0,0]**2 +
    #                          out_wcs.wcs.cd[1,0]**2)*3600.
    
    ### Compute maximum offset needed for padding
    if add_padding:
        ra0, de0 = out_wcs.all_pix2world([0],[0],0)
    
        x0 = np.zeros(len(files))
        y0 = np.zeros(len(files))
    
        for i, file in enumerate(files):
            hx = pyfits.getheader(file, 0)
            h0['EXPTIME'] += hx['EXPTIME']
            h0['FILE{0:04d}'.format(i)] = (file, 
                                        'Included file #{0:d}'.format(i))
        
            h = pyfits.getheader(file, 1)
            flt_wcs = pywcs.WCS(h, relax=True)
            x0[i], y0[i] = flt_wcs.all_world2pix(ra0, de0, 0)
    
        xmax = np.abs(x0).max()
        ymax = np.abs(y0).max()
        padx = 50*int(np.ceil(xmax/50.))
        pady = 50*int(np.ceil(ymax/50.))
        pad = np.maximum(padx, pady)*grow

        if verbose:
            print('Maximum shift (x, y) = ({0:6.1f}, {1:6.1f}), pad={2:d}'.format(xmax, ymax, pad))
    else:
        pad = 0
        
    inter_wcs = out_wcs.deepcopy()
    if grow > 1:
        inter_wcs.wcs.cd /= grow
        for i in range(inter_wcs.sip.a_order+1):
            for j in range(inter_wcs.sip.a_order+1):
                inter_wcs.sip.a[i,j] /= grow**(i+j-1)

        for i in range(inter_wcs.sip.b_order+1):
            for j in range(inter_wcs.sip.b_order+1):
                inter_wcs.sip.b[i,j] /= grow**(i+j-1)

        inter_wcs._naxis1 *= grow
        inter_wcs._naxis2 *= grow
        inter_wcs.wcs.crpix *= grow
        inter_wcs.sip.crpix[0] *= grow
        inter_wcs.sip.crpix[1] *= grow

        if grow > 1:
            inter_wcs.wcs.crpix += grow/2.
            inter_wcs.sip.crpix[0] += grow/2.
            inter_wcs.sip.crpix[1] += grow/2.
    
    inter_wcs._naxis1 += pad
    inter_wcs._naxis2 += pad
    inter_wcs.wcs.crpix += pad
    inter_wcs.sip.crpix[0] += pad
    inter_wcs.sip.crpix[1] += pad

    outh = inter_wcs.to_header(relax=True)
    for key in outh:
        if key.startswith('PC'):
            outh.rename_keyword(key, key.replace('PC','CD'))
    
    outh['GROW'] = grow, 'Grow factor'
    outh['PAD'] = pad, 'Image padding'
    outh['BUNIT'] = h['BUNIT']
    
    sh = (1014*grow + 2*pad, 1014*grow + 2*pad)
    outsci = np.zeros(sh, dtype=np.float32)
    outwht = np.zeros(sh, dtype=np.float32)
    outctx = np.zeros(sh, dtype=np.int32)

    ## Pixel area map
    # PAM_im = pyfits.open(os.path.join(os.getenv('iref'), 'ir_wfc3_map.fits'))
    # PAM = PAM_im[1].data
    
    for i, file in enumerate(files):
        im = pyfits.open(file)
        
        if verbose:
            print('{0:3d} {1:s} {2:6.1f} {3:6.1f} {4:10.2f}'.format(i+1, file,
                                x0[i], y0[i], im[0].header['EXPTIME']))

        dq = utils.unset_dq_bits(im['DQ'].data, okbits=608,
                                           verbose=False)
        wht = 1./im['ERR'].data**2
        wht[(im['ERR'].data == 0) | (dq > 0) | (~np.isfinite(wht))] = 0
        wht[im['SCI'].data < -3*im['ERR'].data] = 0

        wht = np.cast[np.float32](wht)

        exp_wcs = pywcs.WCS(im[1].header, relax=True)
        exp_wcs.pscale = utils.get_wcs_pscale(exp_wcs)
                                 
        #pf = 0.5
        # import drizzlepac.wcs_functions as dwcs
        # xx = out_wcs.deepcopy()
        # #xx.all_pix2world = xx.wcs_world2pix
        # map = dwcs.WCSMap(exp_wcs, xx)

        astrodrizzle.adrizzle.do_driz(im['SCI'].data, exp_wcs, wht,
                                      inter_wcs, outsci, outwht, outctx,
                                      1., 'cps', 1, 
                                      wcslin_pscale=exp_wcs.pscale,
                                      uniqid=1,
                                      pixfrac=pixfrac, kernel=kernel, 
                                      fillval=0, stepsize=10,
                                      wcsmap=SIP_WCSMap)   

        if ds9 is not None:
            ds9.view(outsci, header=outh)

    #outsci /= out_wcs.pscale**2
    rms = 1/np.sqrt(outwht)
    mask = (outwht == 0) | (rms > 100)
    rms[mask] = 0
    outsci[mask] = 0.
    
    hdu = [pyfits.PrimaryHDU(header=h0)]
    hdu.append(pyfits.ImageHDU(data=outsci/grow**2, header=outh, name='SCI'))
    hdu.append(pyfits.ImageHDU(data=rms/grow**2, header=outh, name='ERR'))
    hdu.append(pyfits.ImageHDU(data=mask*1024, header=outh, name='DQ'))
    
    pyfits.HDUList(hdu).writeto(output, clobber=clobber, output_verify='fix')
                    
def combine_visits_and_filters(grow=1, pixfrac=0.5, kernel='point', 
                               filters=['G102', 'G141'], skip=None,
                               split_visit=False, split_quadrants=True, 
                               clobber=True, ds9=None, verbose=True):
    """Make combined FLT files for all FLT files in the working directory separated by targname/visit/filter
    
    Parameters
    ----------
    grow : int
        Factor by which to `grow` the FLT frames to interlaced outputs.  For 
        example, `grow=2` results in 2x2 interlacing.
    
    split_visit : bool
        If `True`, then separate by all TARGNAME/visit otherwise group by 
        TARGNAME and combine visits.
        
    split_quadrants : bool
        Split by 2x2 sub-pixel dither positions
    
    filters : list of strings
        Only make products for exposures that use these filters
        
    pixfrac : float
        Drizzle pixfrac (for kernels other than 'point')
    
    kernel : {'point', 'square'}
        Drizzle kernel. The 'point' kernel is effectively interlacing and is
        best for preserving the noise properties of the final combined image.
        However, can result in empty pixels given the camera distortions
        depending on the dithering of the input exposures.
    
    skip : `slice` or None
        Slice of the overall list of visits to process a subset
        
    ds9 : `pyds9.DS9`
        Display the progress of the script to a DS9 window.
        
    verbose : bool
        Print logging information
        
    clobber : bool
        Overwrite existing files
    
    Returns
    -------
    nothing but creates "cmb" combined files 
    """
    files=glob.glob('i*flt.fits')
    output_list, xx = utils.parse_flt_files(files, uniquename=split_visit)
    
    if skip is None:
        skip = slice(0,len(output_list))
        
    for i in range(len(output_list))[skip]:
        key = output_list[i]['product']
        filter=key.split('-')[-1].upper()
        if filter not in filters:
            continue
        
        output = '{0}_cmb.fits'.format(key)

        if verbose:
            print('\n -- Combine: {0} -- \n'.format(output))
            
        if split_quadrants:
            combine_quadrants(files=output_list[i]['files'], output=output,
                              ref_pixel=[507,507], 
                              pixfrac=pixfrac, kernel=kernel, clobber=clobber,
                              ds9=ds9, verbose=verbose)
        else:
            combine_flt(files=output_list[i]['files'], output=output, 
                    grow=grow, ds9=ds9, verbose=verbose, 
                    clobber=clobber, pixfrac=pixfrac, kernel=kernel)

def get_shifts(files, ref_pixel=[507, 507]):
    """Compute relative pixel shifts based on header WCS
    
    Parameters
    ----------
    files : list of exposure filenames
    
    ref_pixel : [int, int] or [float, float]
        Reference pixel for the computed shifts
    
    Returns
    -------
    h : `~astropy.io.fits.Header`
        Header of the first exposure modified with the total exposure time
        and filenames of the input files in the combination.
    
    xshift, yshift : array-like
        Computed pixel shifts
        
    """
    f0 = pyfits.open(files[0])
    h0 = f0[0].header.copy()
    h0['EXPTIME'] = 0.
    
    out_wcs = pywcs.WCS(f0[1].header, relax=True)
    out_wcs.pscale = utils.get_wcs_pscale(out_wcs)
    
    ### Offsets
    ra0, de0 = out_wcs.all_pix2world([ref_pixel[0]],[ref_pixel[1]],0)

    x0 = np.zeros(len(files))
    y0 = np.zeros(len(files))

    for i, file in enumerate(files):
        hx = pyfits.getheader(file, 0)
        h0['EXPTIME'] += hx['EXPTIME']
        h0['FILE{0:04d}'.format(i)] = file, 'Included file #{0:d}'.format(i)
    
        h = pyfits.getheader(file, 1)
        flt_wcs = pywcs.WCS(h, relax=True)
        x0[i], y0[i] = flt_wcs.all_world2pix(ra0, de0, 0)

    return h0, x0-ref_pixel[0], y0-ref_pixel[1]

def split_pixel_quadrant(dx, dy, figure='quadrants.png', show=False):
    """Group offsets by their sub-pixel quadrant
    
    Parameters
    ----------
    dx, dy : array-like
        Pixel shifts of a list of exposures, for example output from 
        `~grizli.combine.get_shifts`.
    
    figure : str
        If not an empty string, save a diagnostic figure showing the 
        derived sub-pixel quadrants.
    
        .. plot::
            :include-source:

            import matplotlib.pyplot as plt
            import numpy as np
            import grizli.combine

            ### 3D-HST dither pattern
            dx = np.array([0, 10, 6.5, -3.5])
            dy = np.array([0, 3.5, 10, 6.5])
            
            ### This computes the quadrants and also generates the plot
            quad = grizli.combine.split_pixel_quadrant(dx, dy, show=True)
            print quad
            # {0: array([0]), 1: array([2]), 2: array([1]), 3: array([3])}
            plt.show()
    
    show : bool
        Don't close the generated figure (for online docs)
        
    Returns
    -------
    quad : dict
        Dictionary with keys of integers specifying each of 4 sub-pixel 
        quadrants and entries of array indices based on the input `dx` and 
        `dy` arrays.
            
    """
    from matplotlib.ticker import MultipleLocator
    
    xq = np.cast[int](np.round((dx - np.floor(dx))*2)) % 2
    yq = np.cast[int](np.round((dy - np.floor(dy))*2)) % 2
    
    ### Test, show sub-pixel centers in a figure
    if figure:
        xf = ((dx - np.floor(dx)))
        yf = ((dy - np.floor(dy)))
        
        colors = np.array([['r','g'],['b','k']])
        
        fig = plt.figure(figsize=[6,6])
        ax = fig.add_subplot(111)
        
        ax.scatter(xf, yf, c=colors[xq, yq], marker='o', alpha=0.8)
        
        box = np.array([-0.25, 0.25])
        for i in range(2):
            for j in range(2):
                ax.fill_between(j*0.5 + box, i*0.5 + 0.25, i*0.5 - 0.25, 
                                color=colors[j,i], alpha=0.05)
                
                ax.fill_between(j*0.5 + box+1, i*0.5 + 0.25+1, i*0.5 - 0.25+1, 
                                color=colors[j,i], alpha=0.05)
                
                ax.fill_between(j*0.5 + box, i*0.5 + 0.25+1, i*0.5 - 0.25+1, 
                                color=colors[j,i], alpha=0.05)
                
                ax.fill_between(j*0.5 + box+1, i*0.5 + 0.25, i*0.5 - 0.25, 
                                color=colors[j,i], alpha=0.05)
                
        ax.set_xlabel(r'Sub-pixel $x$')
        ax.set_ylabel(r'Sub-pixel $y$')
        ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
        majorLocator = MultipleLocator(0.25)
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_locator(majorLocator)
        
        ax.grid()
        
        if not plt.rcParams['interactive']:
            if not show:
                fig.savefig(figure)
                plt.close(fig)
        
    index = np.arange(len(dx))
    out = {}
    for i in range(4):
        match = xq+2*yq == i
        if match.sum() > 0:
            out[i] = index[match]
    
    return out

### Superceded by `combine_visits_and_filters`
# def combine_figs():
#     """
#     Combine FIGS exposures split by offset pixel quadrant
#     """
#     from stsci.tools import asnutil
#     
#     #all_asn = glob.glob('figs-g*-g1*asn.fits')
#     all_asn = []
#     all_asn.extend(glob.glob('g[ns][0-9]*g102*asn.fits'))
#     all_asn.extend(glob.glob('gdn*g102*asn.fits'))
#     grism = 'g102'
#     
#     #all_asn = glob.glob('gn-z10*-g1*asn.fits')
#     #all_asn.extend(glob.glob('colfax-*g14*asn.fits'))
#     #all_asn = glob.glob('goodsn-*g14*asn.fits')
#     #grism = 'g141'
#     
#     roots = np.unique(['-'.join(asn.split('-')[:2]) for asn in all_asn])
#     for root in roots:
#         all_asn = glob.glob('%s*%s*asn.fits' %(root, grism))
#         angles = np.unique([asn.split('-')[-2] for asn in all_asn])
#         for angle in angles:
#             asn_files = glob.glob('%s*-%s-%s*asn.fits' %(root, angle, grism))
#             
#             grism_files = [] 
#             for file in asn_files:
#                 asn = asnutil.readASNTable(file)
#                 grism_files.extend(['%s_flt.fits' %(flt) for flt in asn['order']])
#             
#             print '%s-%s %d' %(root, angle, len(grism_files))
#             combine_quadrants(files=grism_files, output='%s-%s-%s_cmb.fits' %(root, angle, grism))
            
def combine_quadrants(files=[], output='images_cmb.fits', grow=1, 
                 pixfrac=0.5, kernel='point', ref_pixel=[507,507],
                 ds9=None, verbose=True, clobber=True):
    """Wrapper to split a list of exposures based on their shift sub-pixel quadrants

    Parameters are all passed directly to `~grizli.combine.combine_flt` but 
    separated by shift quadrant with `~grizli.combine.get_shifts` and 
    `~grizli.combine.split_pixel_quadrant`. 
    
    Parameters
    ----------
    files : list
        List of exposure filenames
    
    output : str
        Basename of the output file.  Must end in "_cmb.fits" because the 
        actual output files are derived with the following:
        
        >>> quad_output = output.replace('_cmb.fits',
                                         '_q{0:d}_cmb.fits'.format(q))
    """
    h, dx, dy = get_shifts(files, ref_pixel=ref_pixel)
    out = split_pixel_quadrant(dx, dy, 
                               figure=output.replace('.fits', '_quad.png'))
                               
    for q in out:
        print('Quadrant {0:d}, {1:d} files'.format(q, len(out[q])))
        quad_output = output.replace('_cmb.fits',
                                     '_q{0:d}_cmb.fits'.format(q))
        combine_flt(files=np.array(files)[out[q]], output=quad_output, 
                    grow=grow, pixfrac=pixfrac, kernel=kernel, 
                    ds9=ds9, verbose=verbose, clobber=clobber)    

# Default mapping function based on PyWCS
class SIP_WCSMap:
    def __init__(self,input,output,origin=1):
        """Sample class to demonstrate how to define a coordinate transformation

        Modified from `drizzlepac.wcs_functions.WCSMap` to use full SIP header
        in the `forward` and `backward` methods. Use this class to drizzle to
        an output distorted WCS, e.g., 
            
            >>> drizzlepac.astrodrizzle.do_driz(..., wcsmap=SIP_WCSMap)
        """

        # Verify that we have valid WCS input objects
        self.checkWCS(input,'Input')
        self.checkWCS(output,'Output')

        self.input = input
        self.output = copy.deepcopy(output)
        #self.output = output

        self.origin = origin
        self.shift = None
        self.rot = None
        self.scale = None

    def checkWCS(self,obj,name):
        try:
            assert isinstance(obj, pywcs.WCS)
        except AssertionError:
            print(name +' object needs to be an instance or subclass of a PyWCS object.')
            raise

    def forward(self,pixx,pixy):
        """ Transform the input pixx,pixy positions in the input frame
            to pixel positions in the output frame.

            This method gets passed to the drizzle algorithm.
        """
        # This matches WTRAXY results to better than 1e-4 pixels.
        skyx,skyy = self.input.all_pix2world(pixx,pixy,self.origin)
        #result= self.output.wcs_world2pix(skyx,skyy,self.origin)
        result= self.output.all_world2pix(skyx,skyy,self.origin)
        return result

    def backward(self,pixx,pixy):
        """ Transform pixx,pixy positions from the output frame back onto their
            original positions in the input frame.
        """
        #skyx,skyy = self.output.wcs_pix2world(pixx,pixy,self.origin)
        skyx,skyy = self.output.all_pix2world(pixx,pixy,self.origin)
        result = self.input.all_world2pix(skyx,skyy,self.origin)
        return result

    def get_pix_ratio(self):
        """ Return the ratio of plate scales between the input and output WCS.
            This is used to properly distribute the flux in each pixel in 'tdriz'.
        """
        return self.output.pscale / self.input.pscale

    def xy2rd(self,wcs,pixx,pixy):
        """ Transform input pixel positions into sky positions in the WCS provided.
        """
        return wcs.all_pix2world(pixx,pixy,1)
    def rd2xy(self,wcs,ra,dec):
        """ Transform input sky positions into pixel positions in the WCS provided.
        """
        return wcs.wcs_world2pix(ra,dec,1)
