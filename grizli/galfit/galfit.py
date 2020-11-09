"""
Helper scripts for running galfit
"""
import os
import glob

from collections import OrderedDict
import numpy as np

import astropy.io.fits as pyfits
import astropy.wcs as pywcs

from grizli import utils

try:
    from .psf import DrizzlePSF
except:
    from grizli.galfit.psf import DrizzlePSF

# Constraints
# see https://users.obs.carnegiescience.edu/peng/work/galfit/EXAMPLE.CONSTRAINTS

LINKED_CONSTRAINTS = """# small shifts, limited re, q
# linked x, y, q, pa
2  x   -4 4       # relative to initial guess
2  y   -4 4       # relative to initial guess
2  re  0.51 to 10
2  n   0.4 to 10
2  q   0.15 to 1

3  n   0.4 to 10
3  re  0.51 to 20

2  mag  12 to 35
3  mag  12 to 35

2-3 mag -3 3

3_2 x offset
3_2 y offset
3_2 q  offset
3_2 pa offset
"""

SERSIC_CONSTRAINTS = """# Component {comp}
{comp} x  -3 3
{comp} y  -3 3
{comp} mag  15 to 29
{comp} re  0.2 to 30
{comp} q   0.1 to 1.
{comp} n   0.5 to 8

"""

GALFIT_IMAGES = """===============================================================================
# IMAGE and GALFIT CONTROL PARAMETERS
A) {input}           # Input data image (FITS file)
B) {output}       # Output data image block
C) {sigma}                # Sigma image name (made from data if blank or "none")
D) {psf}   #        # Input PSF image and (optional) diffusion kernel
E) {psf_sample}                   # PSF fine sampling factor relative to data
F) {mask}                # Bad pixel mask (FITS image or ASCII coord list)
G) {constraints}                # File with parameter constraints (ASCII file)
H) 1    {xmax}   1    {ymax}   # Image region to fit (xmin xmax ymin ymax)
I) {sh}    {sh}          # Size of the convolution box (x y)
J) {ab_zeropoint:.3f}       # Magnitude photometric zeropoint
K) {ps:.3f} {ps:.3f}        # Plate scale (dx dy)    [arcsec per pixel]
O) regular             # Display type (regular, curses, both)
P) 0                   # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps

# INITIAL FITTING PARAMETERS
#
#   For object type, the allowed functions are:
#       nuker, sersic, expdisk, devauc, king, psf, gaussian, moffat,
#       ferrer, powsersic, sky, and isophote.
#
#   Hidden parameters will only appear when they're specified:
#       C0 (diskyness/boxyness),
#       Fn (n=integer, Azimuthal Fourier Modes),
#       R0-R10 (PA rotation, for creating spiral structures).
#
# -----------------------------------------------------------------------------
#   par)    par value(s)    fit toggle(s)    # parameter description
# -----------------------------------------------------------------------------
"""


class GalfitModel(object):
    def __str__(self, wcs=None):
        s = '# Object ID: {0}\n'.format(self.id)
        s += ' 0) {0}\n'.format(self.type)
        for k in self.pidx:
            s += '{0:>2}) '.format(self.pidx[k])
            if isinstance(self.pdict[k], list):
                s += ' '.join(['{0:.4f}'.format(v) for v in self.pdict[k]])
                if k != 'output':
                    s += ' '
                    s += ' '.join(['{0:d}'.format(v) for v in self.pfree[k]])
            else:
                s += ' {0:.4f}'.format(self.pdict[k])
                if k != 'output':
                    s += ' {0:d}'.format(self.pfree[k])

            s += ' # {0}\n'.format(k)

        return s


class SersicModel(GalfitModel):
    def __init__(self, position=[0, 0], magnitude=20., r_e=1., n=4., q=0.2, pa=0, output=0, id=0, wcs=None):
        self.type = 'sersic'
        self.id = 0

        if wcs is not None:
            self.ra, self.dec = np.array(wcs.all_world2pix([position[0]],
                                                          [position[1]], 0))[0]

        else:
            self.ra = self.dec = None

        self.pidx = OrderedDict([('position', 1),
                                 ('magnitude', 3),
                                 ('r_e', 4),
                                 ('n', 5),
                                 ('q', 9),
                                 ('pa', 10),
                                 ('output', 'Z')])

        self.pdict = OrderedDict([('position', position),
                                  ('magnitude', magnitude),
                                  ('r_e', r_e),
                                  ('n', n),
                                  ('q', q),
                                  ('pa', pa),
                                  ('output', output)])

        self.pfree = OrderedDict([('position', [1, 1]),
                                 ('magnitude', 1),
                                 ('r_e', 1),
                                 ('n', 1),
                                 ('q', 1),
                                 ('pa', 1)])


class PSFModel(GalfitModel):
    def __init__(self, position=[0, 0], magnitude=20., output=0, id=0, wcs=None):
        self.type = 'psf'
        self.id = 0

        if wcs is not None:
            self.ra, self.dec = np.array(wcs.all_world2pix([position[0]],
                                                          [position[1]], 0))[0]

        else:
            self.ra = self.dec = None

        self.pidx = OrderedDict([('position', 1),
                                 ('magnitude', 3),
                                 ('output', 'Z')])

        self.pdict = OrderedDict([('position', position),
                                  ('magnitude', magnitude),
                                  ('output', output)])

        self.pfree = OrderedDict([('position', [1, 1]),
                                 ('magnitude', 1)])


class SkyModel(GalfitModel):
    def __init__(self, background=0, dx=0, dy=0, output=0, id=0, wcs=None, position=[0, 0]):
        self.type = 'sky'
        self.id = 0

        if wcs is not None:
            self.ra, self.dec = np.array(wcs.all_world2pix([position[0]],
                                                          [position[1]], 0))[0]

        else:
            self.ra = self.dec = None

        self.pidx = OrderedDict([('background', 1),
                                 ('dx', 2),
                                 ('dy', 3),
                                 ('output', 'Z')])

        self.pdict = OrderedDict([('background', background),
                                  ('dx', dx),
                                  ('dy', dy),
                                  ('output', output)])

        self.pfree = OrderedDict([('background', 1),
                                 ('dx', 1),
                                 ('dy', 1)])


SERSIC = """
# Object number: 1
 0) sersic                 #  object type
 1) {x0}  {y0}  1 1  #  position x, y
 3) 20.0890     1          #  Integrated magnitude
 4) 1.      1          #  R_e (half-light radius)   [pix]
 5) 4.      1          #  Sersic index n (de Vaucouleurs n=4)
 6) 0.0000      0          #     -----
 7) 0.0000      0          #     -----
 8) 0.0000      0          #     -----
 9) 0.7570      1          #  axis ratio (b/a)
10) -60.3690    1          #  position angle (PA) [deg: Up=0, Left=90]
 Z) 0                      #  output option (0 = resid., 1 = Don't subtract)
"""

DEVN4 = """
# Object number: 1
0) sersic                 #  object type
1) {x0}  {y0}  1 1  #  position x, y
3) 20.0890     1          #  Integrated magnitude
4) 1.      1          #  R_e (half-light radius)   [pix]
5) 4.      0          #  Sersic index n (de Vaucouleurs n=4)
6) 0.0000      0          #     -----
7) 0.0000      0          #     -----
8) 0.0000      0          #     -----
9) 0.7570      1          #  axis ratio (b/a)
10) -60.3690    1          #  position angle (PA) [deg: Up=0, Left=90]
Z) 0                      #  output option (0 = resid., 1 = Don't subtract)
"""

NUKER = """
# Object number: 1
 0) nuker                 #  object type
 1) {x0}  {y0}  1 1  #  position x, y
 3) 20.0890     1          #  Surface brightness
 4) 10.      1          #  Rb [pix]
 5) 4.2      1          #  Nuker powerlaw alpha
 6) 1.1      1          #     Nuker beta
 7) 0.5      1          #     Nuker gamma
 8) 0.0000      0          #     -----
 9) 0.7570      1          #  axis ratio (b/a)
10) -60.3690    1          #  position angle (PA) [deg: Up=0, Left=90]
 Z) 0                      #  output option (0 = resid., 1 = Don't subtract)
"""

DISK = """
# Object number: 1
 0) expdisk                 #  object type
 1) {x0}  {y0}  1 1  #  position x, y
 3) 20.0890     1          #  Surface brightness
 4) 2.      1          #  R_s [pix]
 9) 0.7570      1          #  axis ratio (b/a)
10) -60.3690    1          #  position angle (PA) [deg: Up=0, Left=90]
C0) -0.05 0                # diskyness(-)/boxyness(+)
 Z) 0                      #  output option (0 = resid., 1 = Don't subtract)
"""


def _tostr(obj):
    if isinstance(obj, list) | isinstance(obj, tuple):
        return ' '.join(['{0}'.format(o) for o in obj])
    else:
        return '{0}'.format(obj)


class GalfitObject(object):
    def __repr__(self):
        return self.__str__()

    def __getitem__(self, k):
        return self.pdict[k]

    def __setitem__(self, k, value):
        self.pdict[k] = value

    def __str__(self):
        line = '# Object\n 0) {0}\n'.format(self.name)
        for k in self.pdict:
            line += '{0:>2}) {1} \t {2} \t# {3}\n'.format(self.pidx[k], _tostr(self.pdict[k]), _tostr(self.pfree[k]), k)

        line += '{0:>2}) {1} \t# {2}\n'.format('Z', self.output, 'output option (0 = resid., 1 = Don\'t subtract)')
        return line

    def set(self, **keys):
        for k in keys:
            if k in self.pdict:
                self.pdict[k] = keys[k]

    def setfree(self, **keys):
        for k in keys:
            if k in self.pfree:
                self.pfree[k] = keys[k]


class GalfitExpdisk(GalfitObject):
    def __init__(self, output=0, fix={}, **keys):
        import copy

        self.pidx = OrderedDict([('pos', 1),
                                 ('mag', 3),
                                 ('R_s', 4),
                                 ('q', 9),
                                 ('pa', 10),
                                 ('disky', 'C0')])

        self.pdef = OrderedDict([('pos', [0, 0]),
                                    ('mag', 20.),
                                    ('R_s', 1.),
                                    ('q', 0.25),
                                    ('pa', 0),
                                    ('disky', -0.05)])

        self.pfree = OrderedDict([('pos', [1, 1]),
                                  ('mag', 1),
                                  ('R_s', 1),
                                  ('n', 1),
                                  ('q', 1),
                                  ('pa', 1),
                                  ('disky', 0)])

        self.name = 'expdisk'
        self.output = output

        self.pdict = copy.deepcopy(self.pdef)
        self.set(**keys)
        self.setfree(**fix)


class GalfitSersic(GalfitObject):
    def __init__(self, output=0, disk=False, dev=False, gaussian=False, fix={}, **keys):
        """
        Sersic profile.

        Options to fix n for:
                disk, n=1
                 dev, n=4
            gaussian, n=0.5

        """
        import copy

        self.pidx = OrderedDict([('pos', 1),
                                 ('mag', 3),
                                 ('R_e', 4),
                                 ('n', 5),
                                 ('q', 9),
                                 ('pa', 10)])

        self.pdef = OrderedDict([('pos', [0, 0]),
                                    ('mag', 20.),
                                    ('R_e', 1.),
                                    ('n', 4),
                                    ('q', 0.25),
                                    ('pa', 0)])

        self.pfree = OrderedDict([('pos', [1, 1]),
                                  ('mag', 1),
                                  ('R_e', 1),
                                  ('n', 1),
                                  ('q', 1),
                                  ('pa', 1)])

        self.name = 'sersic'
        self.output = output

        self.pdict = copy.deepcopy(self.pdef)
        self.set(**keys)
        self.setfree(**fix)

        if disk:
            self.set(n=1)
            self.setfree(n=0)
        elif dev:
            self.set(n=4)
            self.setfree(n=0)
        elif gaussian:
            self.set(n=0.5)
            self.setfree(n=0)


class GalfitKing(GalfitObject):
    """King profile"""

    def __init__(self, output=0, fix={}, **keys):
        import copy

        self.pidx = OrderedDict([('pos', 1),
                                 ('mu', 3),
                                 ('Rc', 4),
                                 ('Rt', 5),
                                 ('alpha', 6),
                                 ('q', 9),
                                 ('pa', 10)])

        self.pdef = OrderedDict([('pos', [0, 0]),
                                    ('mu', 20.),
                                    ('Rc', 1.),
                                    ('Rt', 4),
                                    ('alpha', 2.0),
                                    ('q', 0.9),
                                    ('pa', 0)])

        self.pfree = OrderedDict([('pos', [1, 1]),
                                  ('mu', 1),
                                  ('Rc', 1),
                                  ('Rt', 1),
                                  ('alpha', 1),
                                  ('q', 1),
                                  ('pa', 1)])

        self.name = 'king'
        self.output = output

        self.pdict = copy.deepcopy(self.pdef)
        self.set(**keys)
        self.setfree(**fix)


class GalfitPSF(GalfitObject):
    def __init__(self, output=0, fix={}, **keys):
        import copy

        self.pidx = OrderedDict([('pos', 1),
                                 ('mag', 3)])

        self.pdef = OrderedDict([('pos', [0, 0]),
                                  ('mag', 20.)])

        self.pfree = OrderedDict([('pos', [1, 1]),
                                  ('mag', 1)])

        self.name = 'psf'
        self.output = output

        self.pdict = copy.deepcopy(self.pdef)
        self.set(**keys)
        self.setfree(**fix)


class GalfitSky(GalfitObject):
    def __init__(self, output=0, fix={}, **keys):
        import copy

        self.pidx = OrderedDict([('bg', 1),
                                 ('dx', 2),
                                 ('dy', 3)])

        self.pdef = OrderedDict([('bg', 0.),
                                  ('dx', 0.),
                                  ('dy', 0.)])

        self.pfree = OrderedDict([('bg', 1),
                                  ('dx', 0),
                                  ('dy', 0)])

        self.name = 'sky'
        self.output = output

        self.pdict = copy.deepcopy(self.pdef)
        self.set(**keys)
        self.setfree(**fix)


class Galfitter(object):
    def __init__(self, root='sdssj1723+3411', filter='f140w', galfit_exec='galfit', catfile=None, segfile=None):
        self.root = root
        self.galfit_exec = galfit_exec

        self.filter = filter
        data = self.get_data(filter=filter, catfile=catfile, segfile=segfile)
        self.sci, self.wht, self.seg, self.cat, self.wcs = data

        self.flt_files = self.get_flt_files(sci_hdu=self.sci)
        self.DPSF = DrizzlePSF(flt_files=self.flt_files, info=None, driz_image=self.sci.filename())

    def get_data(self, filter='f140w', catfile=None, segfile=None):
        import glob

        sci_file = glob.glob('{0}-{1}_dr?_sci.fits'.format(self.root, filter))

        sci = pyfits.open(sci_file[0])
        wht = pyfits.open(sci_file[0].replace('sci', 'wht'))

        if segfile is None:
            segfile = sci_file[0].split('_dr')[0]+'_seg.fits'

        seg = pyfits.open(segfile)

        if catfile is None:
            catfile = '{0}-{1}.cat.fits'.format(self.root, filter)

        cat = utils.GTable.gread(catfile)

        wcs = pywcs.WCS(sci[0].header, relax=True)
        return sci, wht, seg, cat, wcs

    def get_flt_files(self, sci_hdu=None):
        flt_files = []
        h = self.sci[0].header
        for i in range(h['NDRIZIM']):
            file = h['D{0:03d}DATA'.format(i+1)].split('[sci')[0]
            if file not in flt_files:
                flt_files.append(file)

        return flt_files

    @staticmethod
    def fit_arrays(sci, wht, seg, psf, id=None, platescale=0.06, exptime=0, path='/tmp', root='gf', galfit_exec='galfit', use_subprocess=False, ab_zeropoint=26, gaussian_guess=False, components=[GalfitSersic()], recenter=True, psf_sample=1, write_constraints=True):
        """
        Fit array data with Galfit

        """
        import subprocess
        from astropy.coordinates import Angle
        import astropy.units as u

        rms = 1/np.sqrt(wht)  # *exptime
        if exptime > 0:
            rms = np.sqrt((rms*exptime)**2+sci*exptime*(sci > 0))/exptime

        rms[wht == 0] = 1e30

        if id is not None:
            mask = ((seg > 0) & (seg != id)) | (wht == 0)
        else:
            mask = wht == 0

        sh = sci.shape[0]

        fp = open('{0}/{1}.feedme'.format(path, root), 'w')
        sci_file = '{0}/{1}_sci.fits'.format(path, root)
        constraints_file = sci_file.replace('_sci.fits', '.constraints')

        cmd = GALFIT_IMAGES.format(input=sci_file,
                                   output=sci_file.replace('_sci', '_out'),
                                   sigma=sci_file.replace('_sci', '_rms'),
                                   psf=sci_file.replace('_sci', '_psf'),
                                   mask=sci_file.replace('_sci', '_mask'),
                                   constraints=constraints_file,
                                   xmax=sh, ymax=sh, sh=sh,
                                   ps=platescale, psf_sample=psf_sample,
                                   ab_zeropoint=ab_zeropoint)

        fp.write(cmd)

        if gaussian_guess:
            fit, q, theta = fit_gauss(sci)

        if (not os.path.exists(constraints_file)) & (write_constraints):
            fpc = open(constraints_file, 'w')
        else:
            fpc = None

        for i, comp in enumerate(components):
            if recenter:
                comp.set(pos=[sh/2., sh/2.])

            if gaussian_guess:
                comp.set(q=q, pa=Angle.wrap_at(theta*u.rad, 360*u.deg).to(u.deg).value)

            fp.write(str(comp))

            if fpc is not None:
                fpc.write(SERSIC_CONSTRAINTS.format(comp=i+1))

        fp.close()
        if fpc is not None:
            fpc.close()

        pyfits.writeto(sci_file, data=sci, overwrite=True)
        pyfits.writeto(sci_file.replace('_sci', '_rms'),
                       data=rms, overwrite=True)
        pyfits.writeto(sci_file.replace('_sci', '_mask'),
                       data=mask*1, overwrite=True)
        pyfits.writeto(sci_file.replace('_sci', '_psf'),
                       data=psf, overwrite=True)

        for ext in ['out', 'model']:
            fi = sci_file.replace('_sci', '_'+ext)
            if os.path.exists(fi):
                os.remove(fi)

        call_string = '{0} {1}/{2}.feedme'.format(galfit_exec, path, root)
        if use_subprocess:
            p = subprocess.Popen(call_string.split(), stdout=subprocess.PIPE)
            stdout_output = p.communicate()[0]
            with open('{0}/{1}.stdout.txt'.format(path, root), 'w') as fp:
                fp.write(stdout_output.decode('utf-8'))
                
        else:
            os.system(call_string)

        # Move galfit.[latest] to output file
        gf_files = glob.glob('galfit.[0-9]*')
        gf_files.sort()
        gf_log = sci_file.replace('_sci.fits', '.gfmodel')

        # Results dictionary
        res = {}
        res['rms'] = rms
        res['mask'] = mask*1
        im = pyfits.open(sci_file.replace('_sci', '_out'))
        res['model'] = im[2]
        res['resid'] = im[3]
        
        if len(gf_files) > 0:
            os.system('mv {0} {1}'.format(gf_files[-1], gf_log))
            lines = open(gf_log).readlines()
            res['ascii'] = lines

        return res

    def fit_object(self, id=449, radec=(None, None), size=40, components=[GalfitSersic()], recenter=True, get_mosaic=True, gaussian_guess=False, get_extended=True):
        """
        Fit an object
        """
        if id is not None:
            rd = self.cat['X_WORLD', 'Y_WORLD'][self.cat['NUMBER'] == id][0]
            radec = tuple(rd)

        xy = self.wcs.all_world2pix([radec[0]], [radec[1]], 0)
        xy = np.array(xy).flatten()
        xp = np.cast[int](np.round(xy))

        slx, sly, wcs_slice = self.DPSF.get_driz_cutout(ra=radec[0], dec=radec[1], get_cutout=False, size=size)
        #drz_cutout = self.get_driz_cutout(ra=ra, dec=dec, get_cutout=True)
        h = self.sci[0].header
        psf = self.DPSF.get_psf(ra=radec[0], dec=radec[1], filter=self.filter.upper(), wcs_slice=wcs_slice, pixfrac=h['D001PIXF'], kernel=h['D001KERN'], get_extended=get_extended)

        exptime = h['EXPTIME']
        sci = self.sci[0].data[sly, slx]  # *exptime
        wht = self.wht[0].data[sly, slx]
        rms = 1/np.sqrt(wht)  # *exptime
        rms = np.sqrt((rms*exptime)**2+sci*exptime*(sci > 0))/exptime

        rms[wht == 0] = 1e30

        seg = self.seg[0].data[sly, slx]
        if id is not None:
            mask = ((seg > 0) & (seg != id)) | (wht == 0)
        else:
            mask = wht == 0

        sh = sci.shape[0]
        path = '/tmp/'

        psf_file = '{0}-{1}_psf_{2:05d}.fits'.format(self.root, self.filter, id)

        fp = open(path+'galfit.feedme', 'w')

        fp.write(GALFIT_IMAGES.format(input=path+'gf_sci.fits', output=path+'gf_out.fits', sigma=path+'gf_rms.fits', psf=psf_file, mask=path+'gf_mask.fits', xmax=sh, ymax=sh, sh=sh, ps=self.DPSF.driz_pscale, psf_sample=1, constraints='none', ab_zeropoint=26))

        if gaussian_guess:
            fit, q, theta = fit_gauss(sci)

        from astropy.coordinates import Angle
        import astropy.units as u

        for comp in components:
            if recenter:
                comp.set(pos=[sh/2., sh/2.])

            if gaussian_guess:
                comp.set(q=q, pa=Angle.wrap_at(theta*u.rad, 360*u.deg).to(u.deg).value)

            fp.write(str(comp))

        fp.close()

        pyfits.writeto(path+'gf_sci.fits', data=sci, overwrite=True)
        pyfits.writeto(path+'gf_rms.fits', data=rms, overwrite=True)
        pyfits.writeto(path+'gf_mask.fits', data=mask*1, overwrite=True)
        #pyfits.writeto(path+'gf_psf.fits', data=psf[1].data, overwrite=True)

        pyfits.writeto(psf_file, data=psf[1].data, overwrite=True)

        for ext in ['out', 'model']:
            if os.path.exists(path+'gf_{0}.fits'.format(ext)):
                os.remove(path+'gf_{0}.fits'.format(ext))

        os.system('{0} {1}/galfit.feedme'.format(self.galfit_exec, path))

        #out = pyfits.open('/tmp/gf_out.fits')

        # Get in DRZ frame
        gf_file = glob.glob('galfit.[0-9]*')[-1]
        lines = open(gf_file).readlines()
        for il, line in enumerate(lines):
            if line.startswith('A)'):
                lines[il] = 'A) {0}     # Input data image (FITS file)\n'.format(self.sci.filename())
            if line.startswith('B)'):
                lines[il] = 'B) {0}-{1}_galfit_{2:05d}.fits      # Output data image block\n'.format(self.root, self.filter, id)
            if line.startswith('P) 0'):
                lines[il] = "P) 1                   # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps\n"

            if line.startswith('H)'):
                out_sh = self.sci[0].data.shape
                lines[il] = "H) 1    {0}   1    {1}   # Image region to fit (xmin xmax ymin ymax)\n".format(out_sh[1], out_sh[0])

            if line.startswith(' 1)'):
                xy = np.cast[float](line.split()[1:3])
                lines[il] = ' 1) {0}  {1}  1 1  #  Position x, y\n'.format(xy[0]+slx.start, xy[1]+sly.start)

        fp = open('galfit.{0}.{1:05d}'.format(self.filter, id), 'w')
        fp.writelines(lines)
        fp.close()

        os.system('{0} galfit.{1}.{2:05d}'.format(self.galfit_exec, self.filter, id))

        model = pyfits.open('{0}-{1}_galfit_{2:05d}.fits'.format(self.root, self.filter, id), mode='update')
        model[0].data /= self.sci[0].header['EXPTIME']
        model.flush()

        return(lines, model)

        model = pyfits.open('/tmp/gf_model.fits')
        if False:
            full_model = self.sci[0].data*0

        full_model += model[0].data
        pyfits.writeto('{0}-{1}_galfit.fits'.format(self.root, self.filter), data=full_model, header=self.sci[0].header, overwrite=True)


def fit_gauss(sci):
    from astropy.modeling import models, fitting

    sh = sci.shape
    gau = models.Gaussian2D(amplitude=sci.max(), x_mean=sh[0]/2, y_mean=sh[0]/2., x_stddev=None, y_stddev=None, theta=None, cov_matrix=None)

    lm = fitting.LevMarLSQFitter()

    yp, xp = np.indices(sh)
    fit = lm(gau, xp, yp, sci)

    q = (fit.x_stddev/fit.y_stddev)**2
    theta = fit.theta.value
    if q > 1:
        q = 1./q
        theta += np.pi/2.

    return fit, q, theta
