"""
Demonstrate aXe trace polynomials.

Initial code taken from `(Brammer, Pirzkal, & Ryan 2014) <https://github.com/WFC3Grism/CodeDescription>`_, which contains a detailed
explanation how the grism configuration parameters and coefficients are defined and evaluated.
"""

import os
from collections import OrderedDict

import numpy as np

from . import GRIZLI_PATH, utils


class aXeConf():
    def __init__(self, conf_file='WFC3.IR.G141.V2.5.conf'):
        """Read an aXe-compatible configuration file

        Parameters
        ----------
        conf_file: str
            Filename of the configuration file to read

        """
        if conf_file is not None:
            self.conf = self.read_conf_file(conf_file)
            self.conf_dict = self.conf
            self.conf_file = conf_file
            self.count_beam_orders()

            # Global XOFF/YOFF offsets
            if 'XOFF' in self.conf.keys():
                self.xoff = np.float(self.conf['XOFF'])
            else:
                self.xoff = 0.

            if 'YOFF' in self.conf.keys():
                self.yoff = np.float(self.conf['YOFF'])
            else:
                self.yoff = 0.

    def read_conf_file(self, conf_file='WFC3.IR.G141.V2.5.conf'):
        """Read an aXe config file, convert floats and arrays

        Parameters
        ----------
        conf_file: str
            Filename of the configuration file to read.

        Parameters are stored in an OrderedDict in `self.conf`.
        """
        conf = OrderedDict()
        fp = open(conf_file)
        lines = fp.readlines()
        fp.close()

        for line in lines:
            # empty / commented lines
            if (line.startswith('#')) | (line.strip() == '') | ('"' in line):
                continue

            # split the line, taking out ; and # comments
            spl = line.split(';')[0].split('#')[0].split()
            param = spl[0]
            if len(spl) > 2:
                value = np.cast[float](spl[1:])
            else:
                try:
                    value = float(spl[1])
                except:
                    value = spl[1]

            conf[param] = value

        return conf

    def count_beam_orders(self):
        """Get the maximum polynomial order in DYDX or DLDP for each beam
        """
        self.orders = {}
        for beam in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
            order = 0
            while 'DYDX_{0:s}_{1:d}'.format(beam, order) in self.conf.keys():
                order += 1

            while 'DLDP_{0:s}_{1:d}'.format(beam, order) in self.conf.keys():
                order += 1

            self.orders[beam] = order-1

    def get_beams(self):
        """Get beam parameters and read sensitivity curves
        """
        import os
        from astropy.table import Table, Column

        self.dxlam = OrderedDict()
        self.nx = OrderedDict()
        self.sens = OrderedDict()
        self.beams = []

        for beam in self.orders:
            if self.orders[beam] > 0:
                self.beams.append(beam)
                self.dxlam[beam] = np.arange(self.conf['BEAM{0}'.format(beam)].min(), self.conf['BEAM{0}'.format(beam)].max(), dtype=int)
                self.nx[beam] = int(self.dxlam[beam].max()-self.dxlam[beam].min())+1
                self.sens[beam] = Table.read('{0}/{1}'.format(os.path.dirname(self.conf_file), self.conf['SENSITIVITY_{0}'.format(beam)]))
                #self.sens[beam].wave = np.cast[np.double](self.sens[beam]['WAVELENGTH'])
                #self.sens[beam].sens = np.cast[np.double](self.sens[beam]['SENSITIVITY'])

                # Need doubles for interpolating functions
                for col in self.sens[beam].colnames:
                    data = np.cast[np.double](self.sens[beam][col])
                    self.sens[beam].remove_column(col)
                    self.sens[beam].add_column(Column(data=data, name=col))

                # Scale BEAM F
                if (beam == 'F') & ('G141' in self.conf_file):
                    self.sens[beam]['SENSITIVITY'] *= 0.35

                if (beam == 'B') & ('G141' in self.conf_file):
                    if self.conf['SENSITIVITY_B'] == 'WFC3.IR.G141.0th.sens.1.fits':
                        self.sens[beam]['SENSITIVITY'] *= 2

                # wave = np.cast[np.double](self.sens[beam]['WAVELENGTH'])
                # sens = np.cast[np.double](self.sens[beam]['SENSITIVITY']
                # self.sens[beam]['WAVELENGTH'] = np.cast[np.double](self.sens[beam]['WAVELENGTH'])
                # self.sens[beam]['SENSITIVITY'] = )

        self.beams.sort()

    def field_dependent(self, xi, yi, coeffs):
        """aXe field-dependent coefficients

        See the `aXe manual <http://axe.stsci.edu/axe/manual/html/node7.html#SECTION00721200000000000000>`_ for a description of how the field-dependent coefficients are specified.

        Parameters
        ----------
        xi, yi : float or array-like
            Coordinate to evaluate the field dependent coefficients, where
            `xi = x-REFX` and `yi = y-REFY`.

        coeffs : array-like
            Field-dependency coefficients

        Returns
        -------
        a : float or array-like
            Evaluated field-dependent coefficients

        """
        # number of coefficients for a given polynomial order
        # 1:1, 2:3, 3:6, 4:10, order:order*(order+1)/2
        if hasattr(coeffs, '__len__'):
            order = int(-1+np.sqrt(1+8*len(coeffs))) // 2
        else:
            order = 1

        # Build polynomial terms array
        # $a = a_0+a_1x_i+a_2y_i+a_3x_i^2+a_4x_iy_i+a_5yi^2+$ ...
        xy = []
        for _p in range(order):
            for _py in range(_p+1):
                # print 'x**%d y**%d' %(p-py, px)
                xy.append(xi**(_p - _py) * yi**(_py))

        # Evaluate the polynomial, allowing for N-dimensional inputs
        a = np.sum((np.array(xy).T*coeffs).T, axis=0)

        return a

    def evaluate_dp(self, dx, dydx):
        """Evalate arc length along the trace given trace polynomial coefficients

        Parameters
        ----------
        dx : array-like
            x pixel to evaluate

        dydx : array-like
            Coefficients of the trace polynomial

        Returns
        -------
        dp : array-like
            Arc length along the trace at position `dx`.

        For `dydx` polynomial orders 0, 1 or 2, integrate analytically.
        Higher orders must be integrated numerically.

        **Constant:**
            .. math:: dp = dx

        **Linear:**
            .. math:: dp = \sqrt{1+\mathrm{DYDX}[1]}\cdot dx

        **Quadratic:**
            .. math:: u = \mathrm{DYDX}[1] + 2\ \mathrm{DYDX}[2]\cdot dx

            .. math:: dp = (u \sqrt{1+u^2} + \mathrm{arcsinh}\ u) / (4\cdot \mathrm{DYDX}[2])

        """
        # dp is the arc length along the trace
        # $\lambda = dldp_0 + dldp_1 dp + dldp_2 dp^2$ ...

        poly_order = len(dydx)-1
        if (poly_order == 2):
            if np.abs(np.unique(dydx[2])).max() == 0:
                poly_order = 1

        if poly_order == 0:  # dy=0
            dp = dx
        elif poly_order == 1:  # constant dy/dx
            dp = np.sqrt(1+dydx[1]**2)*(dx)
        elif poly_order == 2:  # quadratic trace
            u0 = dydx[1]+2*dydx[2]*(0)
            dp0 = (u0*np.sqrt(1+u0**2)+np.arcsinh(u0))/(4*dydx[2])
            u = dydx[1]+2*dydx[2]*(dx)
            dp = (u*np.sqrt(1+u**2)+np.arcsinh(u))/(4*dydx[2])-dp0
        else:
            # high order shape, numerical integration along trace
            # (this can be slow)
            xmin = np.minimum((dx).min(), 0)
            xmax = np.maximum((dx).max(), 0)
            xfull = np.arange(xmin, xmax)
            dyfull = 0
            for i in range(1, poly_order):
                dyfull += i*dydx[i]*(xfull-0.5)**(i-1)

            # Integrate from 0 to dx / -dx
            dpfull = xfull*0.
            lt0 = xfull < 0
            if lt0.sum() > 1:
                dpfull[lt0] = np.cumsum(np.sqrt(1+dyfull[lt0][::-1]**2))[::-1]
                dpfull[lt0] *= -1

            #
            gt0 = xfull > 0
            if gt0.sum() > 0:
                dpfull[gt0] = np.cumsum(np.sqrt(1+dyfull[gt0]**2))

            dp = np.interp(dx, xfull, dpfull)
            if dp[-1] == dp[-2]:
                dp[-1] = dp[-2]+np.diff(dp)[-2]

        return dp

    def get_beam_trace(self, x=507, y=507, dx=0., beam='A', fwcpos=None):
        """Get an aXe beam trace for an input reference pixel and list of output x pixels `dx`

        Parameters
        ----------
        x, y : float or array-like
            Evaluate trace definition at detector coordinates `x` and `y`.

        dx : float or array-like
            Offset in x pixels from `(x,y)` where to compute trace offset and
            effective wavelength

        beam : str
            Beam name (i.e., spectral order) to compute.  By aXe convention,
            `beam='A'` is the first order, 'B' is the zeroth order and
            additional beams are the higher positive and negative orders.

        fwcpos : None or float
            For NIRISS, specify the filter wheel position to compute the
            trace rotation

        Returns
        -------
        dy : float or array-like
            Center of the trace in y pixels offset from `(x,y)` evaluated at
            `dx`.

        lam : float or array-like
            Effective wavelength along the trace evaluated at `dx`.

        """
        NORDER = self.orders[beam]+1

        xi, yi = x-self.xoff, y-self.yoff
        xoff_beam = self.field_dependent(xi, yi, 
                                         self.conf['XOFF_{0}'.format(beam)])
        yoff_beam = self.field_dependent(xi, yi, 
                                         self.conf['YOFF_{0}'.format(beam)])

        # y offset of trace (DYDX)
        dydx = np.zeros(NORDER)  # 0 #+1.e-80
        dydx = [0]*NORDER

        for i in range(NORDER):
            if 'DYDX_{0:s}_{1:d}'.format(beam, i) in self.conf.keys():
                coeffs = self.conf['DYDX_{0:s}_{1:d}'.format(beam, i)]
                dydx[i] = self.field_dependent(xi, yi, coeffs)

        # $dy = dydx_0+dydx_1 dx+dydx_2 dx^2+$ ...

        dy = yoff_beam
        for i in range(NORDER):
            dy += dydx[i]*(dx-xoff_beam)**i

        # wavelength solution
        dldp = np.zeros(NORDER)
        dldp = [0]*NORDER

        for i in range(NORDER):
            if 'DLDP_{0:s}_{1:d}'.format(beam, i) in self.conf.keys():
                coeffs = self.conf['DLDP_{0:s}_{1:d}'.format(beam, i)]
                dldp[i] = self.field_dependent(xi, yi, coeffs)

        self.eval_input = {'x': x, 'y': y, 'beam': beam, 'dx': dx,
                           'fwcpos': fwcpos}
        self.eval_output = {'xi': xi, 'yi': yi, 'dldp': dldp, 'dydx': dydx,
                            'xoff_beam': xoff_beam, 'yoff_beam': yoff_beam,
                            'dy': dy}

        dp = self.evaluate_dp(dx-xoff_beam, dydx)
        # ## dp is the arc length along the trace
        # ## $\lambda = dldp_0 + dldp_1 dp + dldp_2 dp^2$ ...
        # if self.conf['DYDX_ORDER_%s' %(beam)] == 0:   ## dy=0
        #     dp = dx-xoff_beam
        # elif self.conf['DYDX_ORDER_%s' %(beam)] == 1: ## constant dy/dx
        #     dp = np.sqrt(1+dydx[1]**2)*(dx-xoff_beam)
        # elif self.conf['DYDX_ORDER_%s' %(beam)] == 2: ## quadratic trace
        #     u0 = dydx[1]+2*dydx[2]*(0)
        #     dp0 = (u0*np.sqrt(1+u0**2)+np.arcsinh(u0))/(4*dydx[2])
        #     u = dydx[1]+2*dydx[2]*(dx-xoff_beam)
        #     dp = (u*np.sqrt(1+u**2)+np.arcsinh(u))/(4*dydx[2])-dp0
        # else:
        #     ## high order shape, numerical integration along trace
        #     ## (this can be slow)
        #     xmin = np.minimum((dx-xoff_beam).min(), 0)
        #     xmax = np.maximum((dx-xoff_beam).max(), 0)
        #     xfull = np.arange(xmin, xmax)
        #     dyfull = 0
        #     for i in range(1, NORDER):
        #         dyfull += i*dydx[i]*(xfull-0.5)**(i-1)
        #
        #     ## Integrate from 0 to dx / -dx
        #     dpfull = xfull*0.
        #     lt0 = xfull <= 0
        #     if lt0.sum() > 1:
        #         dpfull[lt0] = np.cumsum(np.sqrt(1+dyfull[lt0][::-1]**2))[::-1]
        #         dpfull[lt0] *= -1
        #     #
        #     gt0 = xfull >= 0
        #     if gt0.sum() > 0:
        #         dpfull[gt0] = np.cumsum(np.sqrt(1+dyfull[gt0]**2))
        #
        #     dp = np.interp(dx-xoff_beam, xfull, dpfull)

        # Evaluate dldp
        lam = dp*0.
        for i in range(NORDER):
            lam += dldp[i]*dp**i

        # NIRISS rotation?
        if fwcpos is not None:
            if 'FWCPOS_REF' not in self.conf.keys():
                print('Parameter fwcpos={0} supplied but no FWCPOS_REF in {1:s}'.format(fwcpos, self.conf_file))
                return dy, lam

            order = self.conf['DYDX_ORDER_{0}'.format(beam)]
            if order != 2:
                print('ORDER={0:d} not supported for NIRISS rotation'.format(order))
                return dy, lam

            theta = (fwcpos - self.conf['FWCPOS_REF'])/180*np.pi*1
            theta *= -1  # DMS rotation
            # print('DMS')

            if theta == 0:
                return dy, lam

            # For the convention of swapping/inverting axes for GR150C
            # if 'GR150C' in self.conf_file:
            #     theta = -theta

            # If theta is small, use a small angle approximation.
            # Otherwise, 1./tan(theta) blows up and results in numerical
            # noise.
            xp = (dx-xoff_beam)/np.cos(theta)
            if (1-np.cos(theta) < 5.e-8):
                #print('Approximate!', xoff_beam, np.tan(theta))
                dy = dy + (dx-xoff_beam)*np.tan(theta)
                delta = 0.
                # print('Approx')
            else:
                # Full transformed trace coordinates
                c = dydx
                #print('Not approx')

                beta = c[1]+2*c[2]*xp-1/np.tan(theta)
                chi = c[0]+c[1]*xp+c[2]*xp**2
                if theta < 0:
                    psi = (-beta+np.sqrt(beta**2-4*c[2]*chi))
                    psi *= 1./2/c[2]/np.tan(theta)
                    delta = psi*np.tan(theta)
                    dy = dx*np.tan(theta) + psi/np.cos(theta)
                else:
                    psi = (-beta-np.sqrt(beta**2-4*c[2]*chi))
                    psi *= 1./2/c[2]/np.tan(theta)
                    delta = psi*np.tan(theta)
                    dy = dx*np.tan(theta) + psi/np.cos(theta)

            # Evaluate wavelength at 'prime position along the trace
            dp = self.evaluate_dp(xp+delta, dydx)

            lam = dp*0.
            for i in range(NORDER):
                lam += dldp[i]*dp**i

        return dy, lam

    def show_beams(self, xy=None, beams=['E', 'D', 'C', 'B', 'A']):
        """
        Make a demo plot of the beams of a given configuration file
        """
        import matplotlib.pyplot as plt

        x0, x1 = 507, 507
        dx = np.arange(-800, 1200)

        if 'WFC3.UV' in self.conf_file:
            x0, x1 = 2073, 250
            dx = np.arange(-1200, 1200)
        if 'G800L' in self.conf_file:
            x0, x1 = 2124, 1024
            dx = np.arange(-1200, 1200)
        
        if xy is not None:
            x0, x1 = xy
            
        s = 200  # marker size
        fig, ax = plt.subplots(figsize=[10, 3])
        ax.scatter(0, 0, marker='s', s=s, color='black', edgecolor='0.8',
                    label='Direct')

        for beam in beams:
            if 'XOFF_{0}'.format(beam) not in self.conf.keys():
                continue

            xoff = self.field_dependent(x0, x1, self.conf['XOFF_{0}'.format(beam)])
            dy, lam = self.get_beam_trace(x0, x1, dx=dx, beam=beam)
            xlim = self.conf['BEAM{0}'.format(beam)]
            ok = (dx >= xlim[0]) & (dx <= xlim[1])
            sc = ax.scatter(dx[ok]+xoff, dy[ok], c=lam[ok]/1.e4, marker='s', s=s,
                        alpha=0.5, edgecolor='None')
            ax.text(np.median(dx[ok]), np.median(dy[ok])+1, beam,
                     ha='center', va='center', fontsize=14)
            print('Beam {0}, lambda=({1:.1f} - {2:.1f})'.format(beam, lam[ok].min(), lam[ok].max()))

        ax.grid()
        ax.set_xlabel(r'$\Delta x$' + f' (x0={x0})')
        ax.set_ylabel(r'$\Delta y$' + f' (y0={x1})')

        cb = plt.colorbar(sc, pad=0.01, fraction=0.05)
        cb.set_label(r'$\lambda\,(\mu\mathrm{m})$')
        ax.set_title(self.conf_file)
        fig.tight_layout(pad=0.1)
        #plt.savefig('{0}.pdf'.format(self.conf_file))
        
        return fig


def coeffs_from_astropy_polynomial(p):
    """
    Get field-dependent coefficients in aXe format from an 
    `astropy.modeling.polynomial.Polynomial2D` model
    
    Parameters
    ----------
    p : `astropy.modeling.polynomial.Polynomial2D`
        Polynomial model
    
    Returns
    -------
    coeffs : array-like
        Reordered array of coefficients
    
    """
    coeffs = []
    for _p in range(p.degree+1):
        for _py in range(_p+1):
            # print 'x**%d y**%d' %(_p-_py, _py)
            _px = _p - _py
            pname = f'c{_px}_{_py}'
            if pname in p.param_names:
                pix = p.param_names.index(pname)
                coeffs.append(p.parameters[pix])
            else:
                coeffs.append(0.0)
    
    return np.array(coeffs)


def get_config_filename(instrume='WFC3', filter='F140W',
                        grism='G141', module=None, chip=1):
    """Generate a config filename based on the instrument, filter & grism combination.

    Config files assumed to be found the directory specified by the `$GRIZLI`
    environment variable, i.e., `${GRIZLI}/CONF`.

    Parameters
    ----------
    instrume : {'ACS', 'WFC3', 'NIRISS', 'NIRCam', 'WFIRST'}
        Instrument used

    filter : str
        Direct image filter.  This is only used for WFC3/IR, where the grism
        configuration files have been determined for each direct+grism
        combination separately based on the filter wedge offsets of the
        filters.

    grism : str
        Grism name.  Valid combinations are the following:

            ACS : G800L (assumed)
            WFC3 : G102, G141
            NIRISS : GR150R, GR150C
            NIRCam : F322W2, F356W, F430M, F444W, F460M
            WFIRST : (basic assumptions about the WFI grism)

    chip : int
        For ACS/WFC and UVIS, specifies the chip to use.  Note that this
        is switched with respect to the header EXTNAME extensions:

            EXTVER = 1 is extension 1 / (SCI,1) of the flt/flc files but
            corresponds to CCDCHIP = 2 and the ACS.WFC3.CHIP2 config files.

            and

            EXTVER = 2 is extension 4 / (SCI,2) of the flt/flc files but
            corresponds to CCDCHIP = 1 and the ACS.WFC3.CHIP1 config files.

    Returns
    -------
    conf_file : str
        String path of the configuration file.

    """
    if instrume == 'ACS':
        conf_file = os.path.join(GRIZLI_PATH,
                    'CONF/ACS.WFC.CHIP{0:d}.Stars.conf'.format(chip))
        if not os.path.exists(conf_file):
            conf_file = os.path.join(GRIZLI_PATH,
                        'CONF/ACS.WFC.CHIP{0:d}.Cycle13.5.conf'.format(chip))

    if instrume == 'WFC3':
        if grism == 'G280':
            conf_file = os.path.join(GRIZLI_PATH, 'CONF/G280/',
         'WFC3.UVIS.G280.cal/WFC3.UVIS.G280.CHIP{0:d}.V2.0.conf'.format(chip))

            return conf_file

        conf_file = os.path.join(GRIZLI_PATH,
                                 'CONF/{0}.{1}.V4.32.conf'.format(grism, filter))

        # When direct + grism combination not found for WFC3 assume F140W
        if not os.path.exists(conf_file):
            conf_file = os.path.join(GRIZLI_PATH,
                                 'CONF/{0}.{1}.V4.32.conf'.format(grism, 'F140W'))

    if instrume == 'NIRISS':
        
        conf_files = []
        conf_files.append(os.path.join(GRIZLI_PATH,
                            'CONF/{0}.{1}.221215.conf'.format(grism, filter)))
        conf_files.append(os.path.join(GRIZLI_PATH,
                            'CONF/{0}.{1}.220725.conf'.format(grism, filter)))
        conf_files.append(os.path.join(GRIZLI_PATH,
                            'CONF/{0}.{1}.conf'.format(grism, filter)))
        conf_files.append(os.path.join(GRIZLI_PATH,
                             'CONF/NIRISS.{0}.conf'.format(filter)))
        
        for conf_file in conf_files:
            if os.path.exists(conf_file):
                #print(f'NIRISS: {conf_file}')
                break
            else:
                #print(f'skip NIRISS: {conf_file}')
                pass
                
        # if not os.path.exists(conf_file):
        #     print('CONF/{0}.{1}.conf'.format(grism, filter))
        #     conf_file = os.path.join(GRIZLI_PATH,
        #                          'CONF/NIRISS.{0}.conf'.format(filter))

    # if instrume == 'NIRCam':
    #     conf_file = os.path.join(GRIZLI_PATH,
    #         'CONF/aXeSIM_NC_2016May/CONF/NIRCam_LWAR_{0}.conf'.format(grism))
    if instrume in ['NIRCAM']:
        #conf_file = os.path.join(GRIZLI_PATH,
        #                         f'CONF/NIRCam.A.{filter}.{grism}.conf')
        
        fi = grism
        gr = filter[-1] # R, C
        # conf_file = os.path.join(GRIZLI_PATH,
        #             f'CONF/GRISM_NIRCAM/gNIRCAM.{fi}.mod{module}.{gr}.conf')
        #
        # conf_file = os.path.join(GRIZLI_PATH,
        #             f'CONF/GRISM_NIRCAM/V2/NIRCAM_{fi}_mod{module}_{gr}.conf')

        conf_file = os.path.join(GRIZLI_PATH,
                    f'CONF/GRISM_NIRCAM/V4/NIRCAM_{fi}_mod{module}_{gr}.conf')
        
    elif instrume == 'NIRCAMA':
        fi = grism
        gr = filter[-1] # R, C
        conf_file = os.path.join(GRIZLI_PATH,
                    f'CONF/GRISM_NIRCAM/gNIRCAM.{fi}.modA.{gr}.conf')

        #conf_file = os.path.join(GRIZLI_PATH,
        #                         f'CONF/NIRCam.B.{filter}.{grism}.conf')

    elif instrume == 'NIRCAMB':
        fi = grism
        gr = filter[-1] # R, C
        conf_file = os.path.join(GRIZLI_PATH,
                    f'CONF/GRISM_NIRCAM/gNIRCAM.{fi}.modB.{gr}.conf')

        #conf_file = os.path.join(GRIZLI_PATH,
        #                         f'CONF/NIRCam.B.{filter}.{grism}.conf')

    if instrume == 'WFIRST':
        conf_file = os.path.join(GRIZLI_PATH, 'CONF/WFIRST.conf')

    if instrume == 'WFI':
        conf_file = os.path.join(GRIZLI_PATH, 'CONF/Roman.G150.conf')

    if instrume == 'SYN':
        conf_file = os.path.join(GRIZLI_PATH, 'CONF/syn.conf')

    # Euclid NISP, config files @
    # http://www.astrodeep.eu/euclid-spectroscopic-simulations/

    if instrume == 'NISP':
        if grism == 'BLUE':
            conf_file = os.path.join(GRIZLI_PATH, 'CONF/Euclid.Gblue.0.conf')
        else:
            conf_file = os.path.join(GRIZLI_PATH, 'CONF/Euclid.Gred.0.conf')

    return conf_file


class JwstDispersionTransform(object):
    """
    Rotate NIRISS and NIRCam coordinates such that slitless dispersion has
    wavelength increasing towards +x.  Also works for HST, but does nothing.
    """
    def __init__(self, instrument='NIRCAM', module='A', grism='R', conf_file=None, header=None):
        
        self.instrument = instrument
        self.module = module
        self.grism = grism
        
        if conf_file is not None:
            self.base = os.path.basename(conf_file.split('.conf')[0])
        else:
            self.base = None
            
        if conf_file is not None:
            if 'NIRISS' in conf_file:
                # NIRISS_F200W_GR150R.conf
                self.instrument = 'NIRISS'
                self.grism = self.base.split('_')[2][-1]
                self.module = 'A'
            elif 'NIRCAM' in conf_file:
                # NIRCAM_F444W_modA_R.conf
                self.instrument = 'NIRCAM'
                self.grism = self.base[-1]
                self.module = self.base[-3]
            else:
                # NIRCAM_F444W_modA_R.conf
                self.instrument = 'HST'
                self.grism = 'G141'
                self.module = 'A'
        elif header is not None:
            if 'INSTRUME' in header:
                self.instrument = header['INSTRUME']
            
            if 'MODULE' in header:
                self.module = module
            
            if self.instrument == 'NIRCAM':
                if 'PUPIL' in header:
                    self.grism = header['PUPIL']
            else:
                if 'FILTER' in header:
                    self.grism = header['FILTER']


    @property
    def array_center(self):
        """
        Center of rotation
        
        Maybe this is 1020 for NIRISS?
        """
        if self.instrument == 'HST':
            return np.array([507, 507])
        else:
            return np.array([1024, 1024])


    @property
    def rotation(self):
        """
        Clockwise rotation (degrees) from detector to wavelength increasing
        towards +x direction
        """
        if self.instrument == 'NIRCAM':
            if self.module == 'A':
                if self.grism in ['R', 'GRISMR']:
                    rotation = 0.
                elif self.grism in ['C', 'GRISMC']:
                    rotation = 90.
                else:
                    raise ValueError(f'NIRCAM {self.grism} must be GRISMR/C')
                    
            elif self.module == 'B':
                if self.grism in ['R', 'GRISMR']:
                    rotation = 180.
                elif self.grism in ['C', 'GRISMC']:
                    rotation = 90.
                else:
                    raise ValueError(f'NIRCAM {self.grism} must be GRISMR/C')
            else:
                raise ValueError(f'NIRCAM {self.module} must be A/B')
                
        elif self.instrument == 'NIRISS':
            if self.grism in ['GR150R','R']:
                rotation = 270.
            elif self.grism in ['GR150C','C']:
                rotation = 180.
            else:
                raise ValueError(f'NIRISS {self.grism} must be GR150R/C')
            
        else:
            # e.g., WFC3, ACS
            rotation = 0.
            
        return rotation


    @property
    def rot90(self):
        """
        Rotations are all multiples of 90 for now, so 
        compute values that can be passed to `numpy.rot90` for rotating
        2D image arrays
        """
        return int(np.round(self.rotation/90))


    @property 
    def trace_axis(self):
        """Which detector axis corresponds to increasing wavelength
        """
        if self.instrument == 'NIRCAM':
            if self.module == 'A':
                if self.grism == 'R':
                    axis = '+x'
                else:
                    axis = '+y'
            else:
                if self.grism == 'R':
                    axis = '-x'
                else:
                    axis = '+y'

        elif self.instrument == 'NIRISS':
            if self.grism == 'R':
                axis = '-y'
            else:
                axis = '-x'
        else:
            # e.g., WFC3, ACS
            axis = '+x'
            
        return axis


    @staticmethod
    def rotate_coordinates(x, y, theta, center):
        """
        Rotate cartesian coordinates ``x`` and ``y`` by angle ``theta`` 
        (radians) about ``center``
        """
        _mat = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
        
        x1 = np.atleast_1d(x)
        y1 = np.atleast_1d(y)
        
        return ((np.array([x1, y1]).T-center).dot(_mat)+center).T


    def forward(self, x, y):
        """
        Forward transform, detector to +x
        """
        theta = self.rotation/180*np.pi
        return self.rotate_coordinates(x, y, theta, self.array_center)


    def reverse(self, x, y):
        """
        Reverse transform, +x to detector
        """
        theta = -self.rotation/180*np.pi
        return self.rotate_coordinates(x, y, theta, self.array_center)


class TransformGrismconf(object):
    """
    Transform GRISMCONF-format configuration files to grizli convention of
    wavelength increasing towards +x 

    See https://github.com/npirzkal/GRISMCONF and config files at, e.g., 
    https://github.com/npirzkal/GRISM_NIRCAM.

    """
    def __init__(self, conf_file=''):
        """
        Parameters
        ----------
        conf_file : str
            Configuration filename
        
        """
        import grismconf
        
        self.conf_file = conf_file
        self.conf = grismconf.Config(conf_file)
        
        self.transform = JwstDispersionTransform(conf_file=conf_file)

        self.order_names = {'A':'+1',
                           'B':'0',
                           'C':'+2',
                           'D':'+3',
                           'E':'-1',
                           'F':'+4'}
        
        self.beam_names = {}
        for k in self.order_names:
            self.beam_names[self.order_names[k]] = k

        self.dxlam = OrderedDict()
        self.nx = OrderedDict()
        self.sens = OrderedDict()
        
        self.xoff = 0.0
        self.yoff = 0.0

        self.conf_dict = {}


    @property
    def orders(self):
        """
        GRISMCONF order names, like '+1', '0', '+2', etc.
        """
        return self.conf.orders


    @property
    def beams(self):
        """
        aXe beam names like 'A','B','C', etc.
        """
        beams = [self.beam_names[k] for k in self.orders]
        return beams


    def get_beam_trace(self, x=1024, y=1024, dx=0., beam='A', fwcpos=None):
        """
        Function analogous to `grizli.grismconf.aXeConf.get_beam_trace` but
        that accounts for the different dispersion axes of JWST grisms
        
        Parameters
        ----------
        x, y : float
            Reference position in the rotated frame

        dx : array-like
            Offset in pixels along the trace

        beam : str
            Grism order, translated from +1, 0, +2, +3, -1 = A, B, C, D, E

        fwcpos : float
            NIRISS rotation *(not implemented)*

        Returns
        -------
        dy : float or array-like
            Center of the trace in y pixels offset from `(x,y)` evaluated at
            `dx`.

        lam : float or array-like
            Effective wavelength along the trace evaluated at `dx`.

        """
        from astropy.modeling.models import Polynomial2D
        
        x0 = np.squeeze(self.transform.reverse(x, y))
        
        if self.transform.trace_axis == '+x':
            t_func = self.conf.INVDISPX
            trace_func = self.conf.DISPY
            delta = 1*dx
        elif self.transform.trace_axis == '-x':
            t_func = self.conf.INVDISPX
            trace_func = self.conf.DISPY
            delta = -1*dx
        elif self.transform.trace_axis == '+y':
            t_func = self.conf.INVDISPY
            trace_func = self.conf.DISPX
            delta = 1*dx
        else: # -y
            t_func = self.conf.INVDISPY
            trace_func = self.conf.DISPX
            delta = -1*dx

        #print('xref: ', self.conf_file, self.transform.trace_axis)
        #print(x0, t_func)

        t = t_func(self.order_names[beam], *x0, delta)
        tdx = self.conf.DISPX(self.order_names[beam], *x0, t)
        tdy = self.conf.DISPY(self.order_names[beam], *x0, t)

        rev = self.transform.forward(x0[0]+tdx, x0[1]+tdy)
        trace_dy = rev[1,:] - y
        #trace_dy = y - rev[1,:]
        
        # Trace offsets for NIRCam
        if 'V4/NIRCAM_F444W_modB_R.conf' in self.conf_file:
            trace_dy += -0.5
            
            # Shifts derived from FRESCO
            coeffs = {'c0_0': 0.3723992993620532,
                      'c1_0': -0.00011461411413576305,
                      'c2_0': -8.575199405062535e-08,
                      'c0_1': -0.0011862122093603026,
                      'c0_2': 4.1403439215806165e-07,
                      'c1_1': 1.6558275336712723e-07
                     }
            
            poly = Polynomial2D(degree=2, **coeffs)
            trace_dy += poly(x, y)
            #print(f'polynomial offset: {poly(x,y):.3f}')
            
        elif 'V4/NIRCAM_F444W_modA_R.conf' in self.conf_file:
            trace_dy += -2.5
            
            # Shifts derived from FRESCO
            coeffs = {'c0_0': 0.34191256988768415,
                      'c1_0': -0.0003378232293429956,
                      'c2_0': -9.238111910134196e-09,
                      'c0_1': -7.063720696711682e-05,
                      'c0_2': 2.5217177632321527e-08,
                      'c1_1': -1.4345820074275903e-07
                      }

            poly = Polynomial2D(degree=2, **coeffs)
            trace_dy += poly(x, y)
            #print(f'polynomial offset: {poly(x,y):.3f}')
            
        elif os.path.basename(self.conf_file) == 'NIRCAM_F444W_modA_R.conf':
            trace_dy += -2.5
        elif os.path.basename(self.conf_file) == 'NIRCAM_F444W_modA_C.conf':
            trace_dy += -0.1
            
        wave = self.conf.DISPL(self.order_names[beam], *x0, t)
        if self.transform.instrument != 'HST':
            wave *= 1.e4

        return trace_dy, wave


    def get_beams(self, nt=512):
        """
        Get beam parameters and read sensitivity curves
        
        Parameters
        ----------
        nt : int
            Number of points to sample the GRISMCONF `t` parameter
        
        Returns
        -------
        sets `dxlam`, `nx`, `sens`, attributes
        
        """
        import os
        from astropy.table import Table, Column

        t = np.linspace(0, 1, nt)

        for beam in self.beams:
            order = self.order_names[beam]
            dx = self.conf.DISPX(order, *self.transform.array_center, t)
            dy = self.conf.DISPY(order, *self.transform.array_center, t)
            lam = self.conf.DISPL(order, *self.transform.array_center, t)

            trace_axis = self.transform.trace_axis

            if trace_axis == '+x':
                xarr = 1*dx
            elif trace_axis == '-x':
                xarr = -1*dx
            elif trace_axis == '+y':
                xarr = 1*dy
            elif trace_axis == '-y':
                xarr = -1*dy

            xarr = np.cast[int](np.round(xarr))

            #self.beams.append(beam)
            self.dxlam[beam] = np.arange(xarr[0], xarr[-1], dtype=int)
            self.nx[beam] = xarr[-1] - xarr[0]+1
                        
            sens = Table()
            sens['WAVELENGTH'] = lam.astype(np.double)
            sens['SENSITIVITY'] = self.conf.SENS[order](lam).astype(np.double)
            if lam.max() < 100:
                sens['WAVELENGTH'] *= 1.e4

            self.sens[beam] = sens
            
            self.conf_dict[f'BEAM{beam}'] = np.array([xarr[0], xarr[-1]])
            self.conf_dict[f'MMAG_EXTRACT_{beam}'] = 29
        
        # Read updated sensitivity files
        if 'V4/NIRCAM_F444W' in self.conf_file:
            sens_file = self.conf_file.replace('.conf', '_ext_sensitivity.fits')
            if os.path.exists(sens_file):
                # print(f'Replace sensitivity: {sens_file}')
                new = utils.read_catalog(sens_file)
                _tab = utils.GTable()
                _tab['WAVELENGTH'] = new['WAVELENGTH'].astype(float)
                _tab['SENSITIVITY'] = new['SENSITIVITY'].astype(float)
                _tab['ERROR'] = new['ERROR'].astype(float)
                
                self.sens['A'] = _tab


def load_grism_config(conf_file, warnings=True):
    """Load parameters from an aXe configuration file

    Parameters
    ----------
    conf_file : str
        Filename of the configuration file

    Returns
    -------
    conf : `~grizli.grismconf.aXeConf`
        Configuration file object.  Runs `conf.get_beams()` to read the
        sensitivity curves.
    """
    if 'V3/NIRCAM' in conf_file:
        conf = TransformGrismconf(conf_file)
        conf.get_beams()
    elif 'V2/NIRCAM' in conf_file:
        conf = TransformGrismconf(conf_file)
        conf.get_beams()
    elif 'V4/NIRCAM' in conf_file:
        conf = TransformGrismconf(conf_file)
        conf.get_beams()
    else:
        conf = aXeConf(conf_file)
        conf.get_beams()
    
    # Preliminary hacks for on-sky NIRISS
    if 'GR150' in conf_file:
        if 0:
            hack_niriss = 1./1.8 * 1.1
        
            msg = f"""
     ! Scale NIRISS sensitivity by {hack_niriss:.3f} to hack gain correction
     ! and match GLASS MIRAGE simulations. Sensitivity will be updated when
     ! on-sky data available
     """
            msg = f' ! Scale NIRISS sensitivity by {hack_niriss:.3f} prelim flux correction'
            utils.log_comment(utils.LOGFILE, msg, verbose=warnings)
        else:
            hack_niriss = 1.0
            
        for b in conf.sens:
            conf.sens[b]['SENSITIVITY'] *= hack_niriss
            if 'ERROR' in conf.sens[b].colnames:
                conf.sens[b]['ERROR'] *= hack_niriss
        
        if ('F115W' in conf_file) | ('.2212' in conf_file):
            pass
            # msg = f""" !! Shift F115W along dispersion"""
            # utils.log_comment(utils.LOGFILE, msg, verbose=warnings)
            # for b in conf.beams:
            #     #conf.conf[f'DYDX_{b}_0'][0] += 0.25
            #     conf.conf[f'DLDP_{b}_0'] -= conf.conf[f'DLDP_{b}_1']*0.5
        else:
            msg = f""" !! Shift {os.path.basename(conf_file)} along dispersion"""
            utils.log_comment(utils.LOGFILE, msg, verbose=warnings)
            for b in conf.beams:                    
                #conf.conf[f'DYDX_{b}_0'][0] += 0.25
                conf.conf[f'DLDP_{b}_0'] += conf.conf[f'DLDP_{b}_1']*0.5
                
                # For red galaxy
                conf.conf[f'DLDP_{b}_0'] += conf.conf[f'DLDP_{b}_1']*0.5                
                # if 'F200W' in conf_file:
                #     conf.conf[f'DLDP_{b}_0'] += conf.conf[f'DLDP_{b}_1']*0.5

                
        #     _w = conf.sens['A']['WAVELENGTH']
        #     _w0 = (_w*conf.sens['A']['SENSITIVITY']).sum()
        #     _w0 /=  conf.sens['A']['SENSITIVITY'].sum()
        #     slope = 1.05 + 0.2 * (_w - _w0)/3000
        #     # print('xxx', conf_file, _w0)
        #     conf.sens['A']['SENSITIVITY'] *= slope
        
        if ('F150W' in conf_file) & (hack_niriss > 1.01):
            conf.sens['A']['SENSITIVITY'] *= 1.08
             
        # Scale 0th orders in F150W
        if ('F150W' in conf_file): # | ('F200W' in conf_file):
            msg = f""" ! Scale 0th order (B) by an additional x 1.5"""
            utils.log_comment(utils.LOGFILE, msg, verbose=warnings)
            conf.sens['B']['SENSITIVITY'] *= 1.5
            if 'ERROR' in conf.sens['B'].colnames:
                conf.sens['B']['ERROR'] *= 1.5
        
        # Another shift from 0723, 2744
        # if ('GR150C.F200W' in conf_file):
        #     msg = f""" !! Extra shift for GR150C.F200W"""
        #     utils.log_comment(utils.LOGFILE, msg, verbose=warnings)
        #     for b in conf.beams:
        #         pass
        #         #conf.conf[f'DYDX_{b}_0'][0] += 0.25
        #         # conf.conf[f'DLDP_{b}_0'] -= conf.conf[f'DLDP_{b}_1']*0.5
        
        # Shift x by 1 px
        # msg = f""" ! Shift NIRISS by 0.5 pix along dispersion direction"""
        # utils.log_comment(utils.LOGFILE, msg, verbose=warnings)
        # 
        # for b in conf.beams:
        #     conf.conf[f'DLDP_{b}_0'] -= conf.conf[f'DLDP_{b}_1']*0.5
                
    return conf
