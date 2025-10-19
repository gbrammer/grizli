"""
Demonstrate aXe trace polynomials.

Initial code taken from `(Brammer, Pirzkal, & Ryan 2014) <https://github.com/WFC3Grism/CodeDescription>`_, which contains a detailed
explanation how the grism configuration parameters and coefficients are defined and evaluated.
"""

import os
from collections import OrderedDict

import numpy as np

from . import GRIZLI_PATH, utils
from .jwst_utils import crds_reffiles

DEFAULT_CRDS_CONTEXT = "jwst_1123.pmap"

NIRCAM_CONF_VERSION = "V8.5"

if os.getenv("NIRCAM_CONF_VERSION") is not None:
    NIRCAM_CONF_VERSION = os.getenv("NIRCAM_CONF_VERSION")
    print(f"Use NIRCAM_CONF_VERSION = {NIRCAM_CONF_VERSION}")

if os.getenv("CRDS_CONTEXT") is not None:
    DEFAULT_CRDS_CONTEXT = os.getenv("CRDS_CONTEXT")

VERBOSITY = 3

def show_available_nircam_versions(filter="F444W", module="B", grism="R", verbose=True):
    """
    Show all available versions of the NIRCAM Grism config files

    Parameters
    ----------
    filter, module, grism : str
        NIRCAM config file identifiers:
        ``CONF/GRISM_NIRCAM/V2/NIRCAM_{filter}_mod{module}_{grism}.conf``

    verbose : bool
        Print the available versions to the terminal.

    Returns
    -------
    versions : list
        List of available config files

    """
    import glob

    files = glob.glob(
        os.path.join(
            GRIZLI_PATH, f"CONF/GRISM_NIRCAM/*/NIRCAM_{filter}_mod{module}_{grism}.conf"
        )
    )

    files.sort()

    versions = []

    for file in files:
        versions.append(file.split("/")[-2])
        if verbose:
            print(f"{versions[-1]:>8}   {file}")

    return versions


class aXeConf:
    def __init__(self, conf_file="WFC3.IR.G141.V2.5.conf"):
        """
        Read an aXe-compatible configuration file

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
            if "XOFF" in self.conf.keys():
                self.xoff = float(self.conf["XOFF"])
            else:
                self.xoff = 0.0

            if "YOFF" in self.conf.keys():
                self.yoff = float(self.conf["YOFF"])
            else:
                self.yoff = 0.0

    def read_conf_file(self, conf_file="WFC3.IR.G141.V2.5.conf"):
        """
        Read an aXe config file, convert floats and arrays

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
            if (line.startswith("#")) | (line.strip() == "") | ('"' in line):
                continue

            # split the line, taking out ; and # comments
            spl = line.split(";")[0].split("#")[0].split()
            param = spl[0]
            if len(spl) > 2:
                value = np.asarray(spl[1:], dtype=float)
            else:
                try:
                    value = float(spl[1])
                except:
                    value = spl[1]

            conf[param] = value

        return conf

    def count_beam_orders(self):
        """
        Get the maximum polynomial order in DYDX or DLDP for each beam
        """
        self.orders = {}
        for beam in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
            order = 0
            while "DYDX_{0:s}_{1:d}".format(beam, order) in self.conf.keys():
                order += 1

            while "DLDP_{0:s}_{1:d}".format(beam, order) in self.conf.keys():
                order += 1

            self.orders[beam] = order - 1

    def get_beams(self):
        """
        Get beam parameters and read sensitivity curves
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
                self.dxlam[beam] = np.arange(
                    self.conf["BEAM{0}".format(beam)].min(),
                    self.conf["BEAM{0}".format(beam)].max(),
                    dtype=int,
                )
                self.nx[beam] = int(self.dxlam[beam].max() - self.dxlam[beam].min()) + 1
                self.sens[beam] = Table.read(
                    "{0}/{1}".format(
                        os.path.dirname(self.conf_file),
                        self.conf["SENSITIVITY_{0}".format(beam)],
                    )
                )
                # self.sens[beam].wave = np.asarray(self.sens[beam]['WAVELENGTH'],dtype=np.double)
                # self.sens[beam].sens = np.asarray(self.sens[beam]['SENSITIVITY'],dtype=np.double)

                # Need doubles for interpolating functions
                for col in self.sens[beam].colnames:
                    data = np.asarray(self.sens[beam][col], dtype=np.double)
                    self.sens[beam].remove_column(col)
                    self.sens[beam].add_column(Column(data=data, name=col))

                # Scale BEAM F
                if (beam == "F") & ("G141" in self.conf_file):
                    self.sens[beam]["SENSITIVITY"] *= 0.35

                if (beam == "B") & ("G141" in self.conf_file):
                    if self.conf["SENSITIVITY_B"] == "WFC3.IR.G141.0th.sens.1.fits":
                        self.sens[beam]["SENSITIVITY"] *= 2

                # wave = np.asarray(self.sens[beam]['WAVELENGTH'],dtype=np.double)
                # sens = np.asarray(self.sens[beam]['SENSITIVITY'],dtype=np.double)
                # self.sens[beam]['WAVELENGTH'] = np.asarray(self.sens[beam]['WAVELENGTH'],dtype=np.double)
                # self.sens[beam]['SENSITIVITY'] = )

        self.beams.sort()

    def remove_beam(self, beam):
        """
        Remove a beam definition

        Parameters
        ----------
        beam : str
            Beam name to remove. Example: 'A'.
        """
        if beam in self.beams:
            ix = self.beams.index(beam)
            _ = self.beams.pop(ix)

    def reset_beam_list(self):
        """
        Reset full beam list, perhaps after removing some with ``remove_beam``
        """
        for beam in self.orders:
            if self.orders[beam] > 0:
                self.beams.append(beam)

    def field_dependent(self, xi, yi, coeffs):
        """
        aXe field-dependent coefficients

        See the `aXe manual <http://axe.stsci.edu/axe/manual/html/node7.html#SECTION00721200000000000000>`_ 
        for a description of how the field-dependent coefficients are specified.

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
        if hasattr(coeffs, "__len__"):
            order = int(-1 + np.sqrt(1 + 8 * len(coeffs))) // 2
        else:
            order = 1

        # Build polynomial terms array
        # $a = a_0+a_1x_i+a_2y_i+a_3x_i^2+a_4x_iy_i+a_5yi^2+$ ...
        xy = []
        for _p in range(order):
            for _py in range(_p + 1):
                # print 'x**%d y**%d' %(p-py, px)
                xy.append(xi ** (_p - _py) * yi ** (_py))

        # Evaluate the polynomial, allowing for N-dimensional inputs
        a = np.sum((np.array(xy).T * coeffs).T, axis=0)

        return a

    def evaluate_dp(self, dx, dydx):
        r"""Evalate arc length along the trace given trace polynomial coefficients

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

        poly_order = len(dydx) - 1
        if poly_order == 2:
            if np.abs(np.unique(dydx[2])).max() == 0:
                poly_order = 1

        if poly_order == 0:  # dy=0
            dp = dx
        elif poly_order == 1:  # constant dy/dx
            dp = np.sqrt(1 + dydx[1] ** 2) * (dx)
        elif poly_order == 2:  # quadratic trace
            u0 = dydx[1] + 2 * dydx[2] * (0)
            dp0 = (u0 * np.sqrt(1 + u0 ** 2) + np.arcsinh(u0)) / (4 * dydx[2])
            u = dydx[1] + 2 * dydx[2] * (dx)
            dp = (u * np.sqrt(1 + u ** 2) + np.arcsinh(u)) / (4 * dydx[2]) - dp0
        else:
            # high order shape, numerical integration along trace
            # (this can be slow)
            xmin = np.minimum((dx).min(), 0)
            xmax = np.maximum((dx).max(), 0)
            xfull = np.arange(xmin, xmax)
            dyfull = 0
            for i in range(1, poly_order):
                dyfull += i * dydx[i] * (xfull - 0.5) ** (i - 1)

            # Integrate from 0 to dx / -dx
            dpfull = xfull * 0.0
            lt0 = xfull < 0
            if lt0.sum() > 1:
                dpfull[lt0] = np.cumsum(np.sqrt(1 + dyfull[lt0][::-1] ** 2))[::-1]
                dpfull[lt0] *= -1

            #
            gt0 = xfull > 0
            if gt0.sum() > 0:
                dpfull[gt0] = np.cumsum(np.sqrt(1 + dyfull[gt0] ** 2))

            dp = np.interp(dx, xfull, dpfull)
            if dp[-1] == dp[-2]:
                dp[-1] = dp[-2] + np.diff(dp)[-2]

        return dp

    def get_beam_trace(self, x=507, y=507, dx=0.0, beam="A", fwcpos=None):
        """
        Get an aXe beam trace for an input reference pixel and list of output x pixels `dx`

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
        NORDER = self.orders[beam] + 1

        xi, yi = x - self.xoff, y - self.yoff
        xoff_beam = self.field_dependent(xi, yi, self.conf["XOFF_{0}".format(beam)])
        yoff_beam = self.field_dependent(xi, yi, self.conf["YOFF_{0}".format(beam)])

        # y offset of trace (DYDX)
        dydx = np.zeros(NORDER)  # 0 #+1.e-80
        dydx = [0] * NORDER

        for i in range(NORDER):
            if "DYDX_{0:s}_{1:d}".format(beam, i) in self.conf.keys():
                coeffs = self.conf["DYDX_{0:s}_{1:d}".format(beam, i)]
                dydx[i] = self.field_dependent(xi, yi, coeffs)

        # $dy = dydx_0+dydx_1 dx+dydx_2 dx^2+$ ...

        dy = yoff_beam
        for i in range(NORDER):
            dy += dydx[i] * (dx - xoff_beam) ** i

        # wavelength solution
        dldp = np.zeros(NORDER)
        dldp = [0] * NORDER

        for i in range(NORDER):
            if "DLDP_{0:s}_{1:d}".format(beam, i) in self.conf.keys():
                coeffs = self.conf["DLDP_{0:s}_{1:d}".format(beam, i)]
                dldp[i] = self.field_dependent(xi, yi, coeffs)

        self.eval_input = {"x": x, "y": y, "beam": beam, "dx": dx, "fwcpos": fwcpos}
        self.eval_output = {
            "xi": xi,
            "yi": yi,
            "dldp": dldp,
            "dydx": dydx,
            "xoff_beam": xoff_beam,
            "yoff_beam": yoff_beam,
            "dy": dy,
        }

        dp = self.evaluate_dp(dx - xoff_beam, dydx)
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
        lam = dp * 0.0
        for i in range(NORDER):
            lam += dldp[i] * dp ** i

        # NIRISS rotation?
        if fwcpos is not None:
            if "FWCPOS_REF" not in self.conf.keys():
                print(
                    "Parameter fwcpos={0} supplied but no FWCPOS_REF in {1:s}".format(
                        fwcpos, self.conf_file
                    )
                )
                return dy, lam

            order = int(self.conf["DYDX_ORDER_{0}".format(beam)])
            if order > 2:
                print(f"ORDER={order} > 2 not supported for NIRISS rotation")
                return dy, lam

            theta = (fwcpos - self.conf["FWCPOS_REF"]) / 180 * np.pi * 1
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
            xp = (dx - xoff_beam) / np.cos(theta)
            if (1 - np.cos(theta) < 5.0e-8) | (np.abs(dydx[2]) < 1.0e-8) | (order < 2):
                # print('Approximate!', xoff_beam, np.tan(theta))
                dy = dy + (dx - xoff_beam) * np.tan(theta)
                delta = 0.0
                # print('Approx')
            else:
                # Full transformed trace coordinates
                c = dydx
                # print('Not approx')

                beta = c[1] + 2 * c[2] * xp - 1 / np.tan(theta)
                chi = c[0] + c[1] * xp + c[2] * xp ** 2
                if theta < 0:
                    psi = -beta + np.sqrt(beta ** 2 - 4 * c[2] * chi)
                    psi *= 1.0 / 2 / c[2] / np.tan(theta)
                    delta = psi * np.tan(theta)
                    dy = dx * np.tan(theta) + psi / np.cos(theta)
                else:
                    psi = -beta - np.sqrt(beta ** 2 - 4 * c[2] * chi)
                    psi *= 1.0 / 2 / c[2] / np.tan(theta)
                    delta = psi * np.tan(theta)
                    dy = dx * np.tan(theta) + psi / np.cos(theta)

            # Evaluate wavelength at 'prime position along the trace
            dp = self.evaluate_dp(xp + delta, dydx)

            lam = dp * 0.0
            for i in range(NORDER):
                lam += dldp[i] * dp ** i

        return dy, lam

    def show_beams(self, xy=None, beams=["E", "D", "C", "B", "A"]):
        """
        Make a demo plot of the beams of a given configuration file

        Parameters
        ----------
        xy : tuple
            Reference pixel in direct image to evaluate the beams.

        beams : list
            List of beams to plot. 
            Default is `["E", "D", "C", "B", "A"]`.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure object.

        """
        import matplotlib.pyplot as plt

        x0, x1 = 507, 507
        dx = np.arange(-800, 1200)

        if "WFC3.UV" in self.conf_file:
            x0, x1 = 2073, 250
            dx = np.arange(-1200, 1200)
        if "G800L" in self.conf_file:
            x0, x1 = 2124, 1024
            dx = np.arange(-1200, 1200)

        if xy is not None:
            x0, x1 = xy

        s = 200  # marker size
        fig, ax = plt.subplots(figsize=[10, 3])
        ax.scatter(
            0, 0, marker="s", s=s, color="black", edgecolor="0.8", label="Direct"
        )

        for beam in beams:
            if "XOFF_{0}".format(beam) not in self.conf.keys():
                continue

            xoff = self.field_dependent(x0, x1, self.conf["XOFF_{0}".format(beam)])
            dy, lam = self.get_beam_trace(x0, x1, dx=dx, beam=beam)
            xlim = self.conf["BEAM{0}".format(beam)]
            ok = (dx >= xlim[0]) & (dx <= xlim[1])
            sc = ax.scatter(
                dx[ok] + xoff,
                dy[ok],
                c=lam[ok] / 1.0e4,
                marker="s",
                s=s,
                alpha=0.5,
                edgecolor="None",
            )
            ax.text(
                np.median(dx[ok]),
                np.median(dy[ok]) + 1,
                beam,
                ha="center",
                va="center",
                fontsize=14,
            )
            print(
                "Beam {0}, lambda=({1:.1f} - {2:.1f})".format(
                    beam, lam[ok].min(), lam[ok].max()
                )
            )

        ax.grid()
        ax.set_xlabel(r"$\Delta x$" + f" (x0={x0})")
        ax.set_ylabel(r"$\Delta y$" + f" (y0={x1})")

        cb = plt.colorbar(sc, pad=0.01, fraction=0.05)
        cb.set_label(r"$\lambda\,(\mu\mathrm{m})$")
        ax.set_title(self.conf_file)
        fig.tight_layout(pad=0.1)
        # plt.savefig('{0}.pdf'.format(self.conf_file))

        return fig

    def load_nircam_sensitivity_curve(self, verbose=True, **kwargs):
        """
        Replace +1 NIRCam sensitivity curves with Nov 10, 2023 updates

        Files generated with the calibration data of P330E from program
        CAL-1538 (K. Gordon)

        Download the FITS files from the link below and put them in
        ``$GRIZLI/CONF/GRISM_NIRCAM/``.

        https://s3.amazonaws.com/grizli-v2/JWSTGrism/NircamSensitivity/index.html

        Parameters
        ----------
        verbose : bool
            Print messages to the terminal.
        """

        if "NIRCAM" not in self.conf_file:
            return None

        pars = os.path.basename(self.conf_file).split("_")

        path = os.path.join(GRIZLI_PATH, "CONF", "GRISM_NIRCAM")

        sens_base = "nircam_wfss_sensitivity_{filter}_{pupil}_{module}.10nov23.fits"

        sens_file = sens_base.format(
            filter=pars[1], pupil="GRISM" + pars[3][0], module=pars[2][-1]
        )

        sens_file = os.path.join(path, sens_file)
        # print('xx', sens_file, os.path.exists(sens_file))

        if os.path.exists(sens_file):
            msg = "grismconf.aXeConf: replace sensitivity curve with "
            msg += f"{sens_file}"
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

            si = utils.read_catalog(sens_file).copy()
            si["ERROR"] = si["SENSITIVITY"].max() * 0.01

            self.SENS["A"] = si


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
    for _p in range(p.degree + 1):
        for _py in range(_p + 1):
            # print 'x**%d y**%d' %(_p-_py, _py)
            _px = _p - _py
            pname = f"c{_px}_{_py}"
            if pname in p.param_names:
                pix = p.param_names.index(pname)
                coeffs.append(p.parameters[pix])
            else:
                coeffs.append(0.0)

    return np.array(coeffs)


def get_config_filename(
    instrume="WFC3",
    filter="F140W",
    grism="G141",
    pupil=None,
    module=None,
    chip=1,
    use_jwst_crds=False,
    crds_context=DEFAULT_CRDS_CONTEXT,
):
    """
    Generate a config filename based on the instrument, filter & grism combination.

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

    pupil : str
        Pupil element for NIRCam grisms (e.g., 'CLEAR', 'GRISMC', 'GRISMR').

    module : str
        NIRCam module (A or B) for NIRCam grisms.

    chip : int
        For ACS/WFC and UVIS, specifies the chip to use.  Note that this
        is switched with respect to the header EXTNAME extensions:

            EXTVER = 1 is extension 1 / (SCI,1) of the flt/flc files but
            corresponds to CCDCHIP = 2 and the ACS.WFC3.CHIP2 config files.

            and

            EXTVER = 2 is extension 4 / (SCI,2) of the flt/flc files but
            corresponds to CCDCHIP = 1 and the ACS.WFC3.CHIP1 config files.

    use_jwst_crds : bool
        Use CRDS ``specwcs`` reference files for JWST instruments

    crds_context : str
        CRDS context to use for JWST reference files.

    Returns
    -------
    conf_file : str
        String path of the configuration file.

    """
    if instrume == "ACS":
        conf_file = os.path.join(
            GRIZLI_PATH, "CONF/ACS.WFC.CHIP{0:d}.Stars.conf".format(chip)
        )
        if not os.path.exists(conf_file):
            conf_file = os.path.join(
                GRIZLI_PATH, "CONF/ACS.WFC.CHIP{0:d}.Cycle13.5.conf".format(chip)
            )

    if instrume == "WFC3":
        if grism == "G280":
            conf_file = os.path.join(
                GRIZLI_PATH,
                "CONF/G280/",
                "WFC3.UVIS.G280.cal/WFC3.UVIS.G280.CHIP{0:d}.V2.0.conf".format(chip),
            )

            return conf_file

        conf_file = os.path.join(
            GRIZLI_PATH, "CONF/{0}.{1}.V4.32.conf".format(grism, filter)
        )

        # When direct + grism combination not found for WFC3 assume F140W
        if not os.path.exists(conf_file):
            conf_file = os.path.join(
                GRIZLI_PATH, "CONF/{0}.{1}.V4.32.conf".format(grism, "F140W")
            )

    if instrume == "NIRISS":

        conf_files = []
        conf_files.append(
            os.path.join(GRIZLI_PATH, "CONF/{0}.{1}.221215.conf".format(grism, filter))
        )
        conf_files.append(
            os.path.join(GRIZLI_PATH, "CONF/{0}.{1}.220725.conf".format(grism, filter))
        )
        conf_files.append(
            os.path.join(GRIZLI_PATH, "CONF/{0}.{1}.conf".format(grism, filter))
        )
        conf_files.append(
            os.path.join(GRIZLI_PATH, "CONF/NIRISS.{0}.conf".format(filter))
        )

        for conf_file in conf_files:
            if os.path.exists(conf_file):
                # print(f'NIRISS: {conf_file}')
                break
            else:
                # print(f'skip NIRISS: {conf_file}')
                pass

        # if not os.path.exists(conf_file):
        #     print('CONF/{0}.{1}.conf'.format(grism, filter))
        #     conf_file = os.path.join(GRIZLI_PATH,
        #                          'CONF/NIRISS.{0}.conf'.format(filter))

    # if instrume == 'NIRCam':
    #     conf_file = os.path.join(GRIZLI_PATH,
    #         'CONF/aXeSIM_NC_2016May/CONF/NIRCam_LWAR_{0}.conf'.format(grism))
    if instrume in ["NIRCAM"]:
        # conf_file = os.path.join(GRIZLI_PATH,
        #                         f'CONF/NIRCam.A.{filter}.{grism}.conf')

        fi = grism
        gr = filter[-1]  # R, C
        # conf_file = os.path.join(GRIZLI_PATH,
        #             f'CONF/GRISM_NIRCAM/gNIRCAM.{fi}.mod{module}.{gr}.conf')
        #
        # conf_file = os.path.join(GRIZLI_PATH,
        #             f'CONF/GRISM_NIRCAM/V2/NIRCAM_{fi}_mod{module}_{gr}.conf')

        # NIRCam preference: 8.5 > 8 > 4

        conf_file_base = os.path.join(
            GRIZLI_PATH,
            f"CONF/GRISM_NIRCAM/[[NIRCAM_VERSION]]/NIRCAM_{fi}_mod{module}_{gr}.conf",
        )

        _conf_versions = [NIRCAM_CONF_VERSION, "V8.5", "V8", "V4", "V6"]

        conf_file = None
        for NIRCAM_VERSION in _conf_versions:
            conf_file = conf_file_base.replace("[[NIRCAM_VERSION]]", NIRCAM_VERSION)
            if os.path.exists(conf_file):
                break

        if conf_file is None:
            raise ValueError

    elif instrume == "NIRCAMA":
        fi = grism
        gr = filter[-1]  # R, C
        conf_file = os.path.join(
            GRIZLI_PATH, f"CONF/GRISM_NIRCAM/gNIRCAM.{fi}.modA.{gr}.conf"
        )

        # conf_file = os.path.join(GRIZLI_PATH,
        #                         f'CONF/NIRCam.B.{filter}.{grism}.conf')

    elif instrume == "NIRCAMB":
        fi = grism
        gr = filter[-1]  # R, C
        conf_file = os.path.join(
            GRIZLI_PATH, f"CONF/GRISM_NIRCAM/gNIRCAM.{fi}.modB.{gr}.conf"
        )

        # conf_file = os.path.join(GRIZLI_PATH,
        #                         f'CONF/NIRCam.B.{filter}.{grism}.conf')

    if (instrume in ["NIRCAM", "NIRISS"]) & use_jwst_crds:
        if instrume == "NIRCAM":
            _pupil = filter
            _filter = grism
        else:
            _pupil = filter
            _filter = grism

        refs = crds_reffiles(
            instrument=instrume,
            filter=_filter,
            pupil=_pupil,
            module=module,
            date=None,
            reftypes=("photom", "specwcs"),
            header=None,
            context=crds_context,
        )

        conf_file = refs["specwcs"]
        if VERBOSITY & 2 > 0:
            print("get_conf: xxx", conf_file, grism, filter, pupil, module, instrume)

    if instrume == "WFIRST":
        conf_file = os.path.join(GRIZLI_PATH, "CONF/WFIRST.conf")

    if instrume == "WFI":
        conf_file = os.path.join(GRIZLI_PATH, "CONF/Roman.G150.conf")

    if instrume == "SYN":
        conf_file = os.path.join(GRIZLI_PATH, "CONF/syn.conf")

    # Euclid NISP, config files @
    # http://www.astrodeep.eu/euclid-spectroscopic-simulations/

    if instrume == "NISP":
        if grism == "BLUE":
            conf_file = os.path.join(GRIZLI_PATH, "CONF/Euclid.Gblue.0.conf")
        else:
            conf_file = os.path.join(GRIZLI_PATH, "CONF/Euclid.Gred.0.conf")

    return conf_file


class JwstDispersionTransform(object):
    """
    Rotate NIRISS and NIRCam coordinates such that slitless dispersion has
    wavelength increasing towards +x.  Also works for HST, but does nothing.
    """

    def __init__(
        self, instrument="NIRCAM", module="A", grism="R", conf_file=None, header=None
    ):
        """
        Parameters
        ----------
        instrument : str
            Instrument name. E.g., 'NIRCAM', 'NIRISS', 'WFC3'.

        module : str
            NIRCAM module.  Can be 'A' or 'B'.

        grism : str
            Grism name.  Can be 'R' or 'C' for NIRCAM and 'GR150R' or 'GR150C'
            for NIRISS.

        conf_file : str
            Configuration file to read instrument/grism from.
            If not provided, will try to infer from the header.

        header : `astropy.io.fits.Header`
            Header object to read instrument/grism from.
            If not provided, will try to infer from the `conf_file`.

        """

        self.instrument = instrument
        self.module = "A" if module is None else module
        self.grism = grism

        if conf_file is not None:
            self.base = os.path.basename(conf_file.split(".conf")[0])
        else:
            self.base = None

        if conf_file is not None:
            if "NIRISS" in conf_file:
                # NIRISS_F200W_GR150R.conf
                self.instrument = "NIRISS"
                self.grism = self.base.split("_")[2][-1]
                self.module = "A"
            elif "NIRCAM" in conf_file:
                # NIRCAM_F444W_modA_R.conf
                self.instrument = "NIRCAM"
                self.grism = self.base[-1]
                self.module = self.base[-3]
            else:
                # NIRCAM_F444W_modA_R.conf
                self.instrument = "HST"
                self.grism = "G141"
                self.module = "A"
        elif header is not None:
            if "INSTRUME" in header:
                self.instrument = header["INSTRUME"]

            if "MODULE" in header:
                self.module = module

            if self.instrument == "NIRCAM":
                if "PUPIL" in header:
                    self.grism = header["PUPIL"]
            else:
                if "FILTER" in header:
                    self.grism = header["FILTER"]

    @property
    def array_center(self):
        """
        Center of rotation

        Maybe this is 1020 for NIRISS?
        """
        if self.instrument == "HST":
            return np.array([507.5, 507.5])
        else:
            return np.array([1024.5, 1024.5])

    @property
    def rotation(self):
        """
        Clockwise rotation (degrees) from detector to wavelength increasing
        towards +x direction
        """
        if self.instrument == "NIRCAM":
            if self.module == "A":
                if self.grism in ["R", "GRISMR"]:
                    rotation = 0.0
                elif self.grism in ["C", "GRISMC"]:
                    rotation = 90.0
                else:
                    raise ValueError(f"NIRCAM {self.grism} must be GRISMR/C")

            elif self.module == "B":
                if self.grism in ["R", "GRISMR"]:
                    rotation = 180.0
                elif self.grism in ["C", "GRISMC"]:
                    rotation = 90.0
                else:
                    raise ValueError(f"NIRCAM {self.grism} must be GRISMR/C")
            else:
                raise ValueError(f"NIRCAM {self.module} must be A/B")

        elif self.instrument == "NIRISS":
            if self.grism in ["GR150R", "R"]:
                rotation = 270.0
            elif self.grism in ["GR150C", "C"]:
                rotation = 180.0
            else:
                raise ValueError(f"NIRISS {self.grism} must be GR150R/C")

        else:
            # e.g., WFC3, ACS
            rotation = 0.0

        return rotation

    @property
    def rot90(self):
        """
        Rotations are all multiples of 90 for now, so
        compute values that can be passed to `numpy.rot90` for rotating
        2D image arrays
        """
        return int(np.round(self.rotation / 90))

    @property
    def trace_axis(self):
        """Which detector axis corresponds to increasing wavelength"""
        if self.instrument == "NIRCAM":
            if self.module == "A":
                if self.grism == "R":
                    axis = "+x"
                else:
                    axis = "+y"
            else:
                if self.grism == "R":
                    axis = "-x"
                else:
                    axis = "+y"

        elif self.instrument == "NIRISS":
            if self.grism == "R":
                axis = "-y"
            else:
                axis = "-x"
        else:
            # e.g., WFC3, ACS
            axis = "+x"

        return axis

    @staticmethod
    def rotate_coordinates(x, y, theta, center):
        """
        Rotate cartesian coordinates ``x`` and ``y`` by angle ``theta``
        about ``center``

        Parameters
        ----------
        x, y : float or array-like
            Original detector coordinates

        theta : float
            Rotation angle in radians.

        center : array-like
            Center of rotation.

        Returns
        -------
        x, y : array-like
            Coordinates in rotated frame

        """
        _mat = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

        x1 = np.atleast_1d(x)
        y1 = np.atleast_1d(y)

        return ((np.array([x1, y1]).T - center).dot(_mat) + center).T

    def forward(self, x, y):
        """
        Forward transform, detector to +x.

        Rotate NIRISS and NIRCam coordinates such that slitless dispersion has
        wavelength increasing towards +x.  Also works for HST, but does nothing.

        Parameters
        ----------
        x, y : float or array-like
            Original detector coordinates

        Returns
        -------
        x, y : array-like
            Coordinates in rotated frame

        """
        theta = self.rotation / 180 * np.pi
        return self.rotate_coordinates(x, y, theta, self.array_center)

    def reverse(self, x, y):
        """
        Reverse transform, +x to detector.

        Rotates coordinates from the +x direction to the detector frame.

        Parameters
        ----------
        x, y : float or array-like
            Coordinates in rotated frame

        Returns
        -------
        x, y : array-like
            Original detector coordinates

        """
        theta = -self.rotation / 180 * np.pi
        return self.rotate_coordinates(x, y, theta, self.array_center)


class TransformGrismconf(object):
    """
    Transform GRISMCONF-format configuration files to grizli convention of
    wavelength increasing towards +x

    See https://github.com/npirzkal/GRISMCONF and config files at, e.g.,
    https://github.com/npirzkal/GRISM_NIRCAM.

    """

    def __init__(self, conf_file="", **kwargs):
        """
        Parameters
        ----------
        conf_file : str
            Configuration filename

        """
        import grismconf

        self.conf_file = conf_file

        if "specwcs" in conf_file:
            self.conf = CRDSGrismConf(conf_file, get_photom=True, **kwargs)
            kws = {
                "instrument": self.conf.instrument,
                "module": self.conf.module,
                "grism": self.conf.grism[-1],
            }
            self.transform = JwstDispersionTransform(**kws)

        else:
            self.conf = grismconf.Config(conf_file)
            self.transform = JwstDispersionTransform(conf_file=conf_file)

        self.order_names = {
            "A": "+1",
            "B": "0",
            "C": "+2",
            "D": "+3",
            "E": "-1",
            "F": "+4",
        }

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
        # beams = [self.beam_names[k] for k in self.orders]
        beams = []
        for k in self.orders:
            if k in self.beam_names:
                beams.append(self.beam_names[k])

        return beams

    def remove_beam(self, beam):
        """
        Remove a beam from the orders list

        Parameters
        ----------
        beam : str
            Beam name, like 'A', 'B', 'C', etc.

        Returns
        -------
        removed : bool
            True if the beam was removed, False if it was not in the list.

        """
        order_name = None
        for k in self.beam_names:
            if self.beam_names[k] == beam:
                order_name = k

        if order_name is None:
            return False

        if hasattr(self.conf, "dm_orders"):
            olist = self.conf.dm_orders
            order_name = int(order_name)
        else:
            olist = self.conf.orders

        if order_name in olist:
            ix = olist.index(order_name)
            _ = olist.pop(ix)

            return True

        else:
            return False

    def get_beam_trace(self, x=1024, y=1024, dx=0.0, beam="A", fwcpos=None):
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

        if self.transform.trace_axis == "+x":
            t_func = self.conf.INVDISPX
            trace_func = self.conf.DISPY
            delta = 1 * dx
        elif self.transform.trace_axis == "-x":
            t_func = self.conf.INVDISPX
            trace_func = self.conf.DISPY
            delta = -1 * dx
        elif self.transform.trace_axis == "+y":
            t_func = self.conf.INVDISPY
            trace_func = self.conf.DISPX
            delta = 1 * dx
        else:  # -y
            t_func = self.conf.INVDISPY
            trace_func = self.conf.DISPX
            delta = -1 * dx

        # print('xref: ', self.conf_file, self.transform.trace_axis)
        # print(x0, t_func)

        t = t_func(self.order_names[beam], *x0, delta)
        tdx = self.conf.DISPX(self.order_names[beam], *x0, t)
        tdy = self.conf.DISPY(self.order_names[beam], *x0, t)

        rev = self.transform.forward(x0[0] + tdx, x0[1] + tdy)
        trace_dy = rev[1, :] - y
        # trace_dy = y - rev[1,:]

        # Trace offsets for NIRCam
        if "V4/NIRCAM_F444W_modB_R.conf" in self.conf_file:
            trace_dy += -0.5

            # Shifts derived from FRESCO
            coeffs = {
                "c0_0": 0.3723992993620532,
                "c1_0": -0.00011461411413576305,
                "c2_0": -8.575199405062535e-08,
                "c0_1": -0.0011862122093603026,
                "c0_2": 4.1403439215806165e-07,
                "c1_1": 1.6558275336712723e-07,
            }

            poly = Polynomial2D(degree=2, **coeffs)
            trace_dy += poly(x, y)
            # print(f'polynomial offset: {poly(x,y):.3f}')

        elif "V4/NIRCAM_F444W_modA_R.conf" in self.conf_file:
            trace_dy += -2.5

            # Shifts derived from FRESCO
            coeffs = {
                "c0_0": 0.34191256988768415,
                "c1_0": -0.0003378232293429956,
                "c2_0": -9.238111910134196e-09,
                "c0_1": -7.063720696711682e-05,
                "c0_2": 2.5217177632321527e-08,
                "c1_1": -1.4345820074275903e-07,
            }

            poly = Polynomial2D(degree=2, **coeffs)
            trace_dy += poly(x, y)
            # print(f'polynomial offset: {poly(x,y):.3f}')

        elif (
            ("V8/NIRCAM" in self.conf_file)
            | ("V8.5/NIRCAM" in self.conf_file)
            | ("V9/NIRCAM" in self.conf_file)
        ):
            # print('V8: do nothing')
            pass

        elif os.path.basename(self.conf_file) == "NIRCAM_F444W_modA_R.conf":
            trace_dy += -2.5
        elif os.path.basename(self.conf_file) == "NIRCAM_F444W_modA_C.conf":
            trace_dy += -0.1

        wave = self.conf.DISPL(self.order_names[beam], *x0, t)
        if self.transform.instrument != "HST":
            wave *= 1.0e4

        return trace_dy, wave

    def get_beams(self, nt=512, min_sens=1.0e-3, **kwargs):
        """
        Get beam parameters and read sensitivity curves

        Parameters
        ----------
        nt : int
            Number of points to sample the GRISMCONF `t` parameter

        min_sens : float
            Minimum sensitivity to consider in the sensitivity curves.

        Returns
        -------
        sets `dxlam`, `nx`, `sens`, attributes

        """
        import os
        from astropy.table import Table, Column

        t = np.linspace(0, 1, nt)

        for beam in self.beams:
            order = self.order_names[beam]

            # Define t from sensitivity
            if self.conf.SENS is None:
                _swave, _ssens = self.conf.SENS_data[order]
                _swave = _swave[_ssens > min_sens * np.nanmax(_ssens)]
                _ssens = _ssens[_ssens > min_sens * np.nanmax(_ssens)]
                _dw = _swave.max() - _swave.min()
                _wgrid = np.linspace(
                    _swave.min() - 0.02 * _dw, _swave.max() + _dw * 0.02, nt
                )
                t = self.conf.INVDISPL(order, *self.transform.array_center, _wgrid)

            dx = self.conf.DISPX(order, *self.transform.array_center, t)
            dy = self.conf.DISPY(order, *self.transform.array_center, t)
            lam = self.conf.DISPL(order, *self.transform.array_center, t)

            trace_axis = self.transform.trace_axis

            if trace_axis == "+x":
                xarr = 1 * dx
            elif trace_axis == "-x":
                xarr = -1 * dx
            elif trace_axis == "+y":
                xarr = 1 * dy
            elif trace_axis == "-y":
                xarr = -1 * dy

            xarr = np.asarray(np.round(xarr), dtype=int)

            # self.beams.append(beam)
            self.dxlam[beam] = np.arange(xarr.min(), xarr.max(), dtype=int)
            self.nx[beam] = xarr.max() - xarr.min() + 1

            # Do we need to force zeroth order to be between first and second?
            if beam == "B":
                bm1 = self.beam_names["-1"]
                bp1 = self.beam_names["+1"]

                if (self.nx[beam] > 500) & (bm1 in self.nx) & (bp1 in self.nx):
                    _xmin = self.dxlam[bm1].max()
                    _xmax = self.dxlam[bp1].min()
                    self.dxlam[beam] = np.arange(_xmin, _xmax, dtype=int)
                    self.nx[beam] = _xmax - _xmin + 1

            sens = Table()
            sens["WAVELENGTH"] = lam.astype(np.double)
            if self.conf.SENS is None:
                # specwcs
                _sens = np.interp(lam, *self.conf.SENS_data[order], left=0.0, right=0.0)
                sens["SENSITIVITY"] = _sens.astype(np.double)
            else:
                sens["SENSITIVITY"] = self.conf.SENS[order](lam).astype(np.double)

            if lam.max() < 100:
                sens["WAVELENGTH"] *= 1.0e4

            self.sens[beam] = sens

            self.conf_dict[f"BEAM{beam}"] = np.array([xarr.min(), xarr.max()])
            self.conf_dict[f"MMAG_EXTRACT_{beam}"] = 29

        # Read updated sensitivity files
        if "V4/NIRCAM_F444W" in self.conf_file:
            sens_file = self.conf_file.replace(".conf", "_ext_sensitivity.fits")
            if os.path.exists(sens_file):
                # print(f'Replace sensitivity: {sens_file}')
                new = utils.read_catalog(sens_file)
                _tab = utils.GTable()
                _tab["WAVELENGTH"] = new["WAVELENGTH"].astype(float)
                _tab["SENSITIVITY"] = new["SENSITIVITY"].astype(float)
                _tab["ERROR"] = new["ERROR"].astype(float)

                self.sens["A"] = _tab

        self.load_nircam_sensitivity_curve(**kwargs)

    def load_nircam_sensitivity_curve(self, verbose=False, **kwargs):
        """
        Replace +1 NIRCam sensitivity curves with Nov 10, 2023 updates

        Files generated with the calibration data of P330E from program
        CAL-1538 (K. Gordon)

        Download the FITS files from the link below and put them in
        ``$GRIZLI/CONF/GRISM_NIRCAM/``.

        https://s3.amazonaws.com/grizli-v2/JWSTGrism/NircamSensitivity/index.html

        Parameters
        ----------
        verbose : bool
            Print messages to the terminal.
        """

        if "NIRCAM" not in self.conf_file:
            return None

        pars = os.path.basename(self.conf_file).split("_")

        path = os.path.join(GRIZLI_PATH, "CONF", "GRISM_NIRCAM")

        sens_base = "nircam_wfss_sensitivity_{filter}_{pupil}_{module}.10nov23.fits"

        sens_file = sens_base.format(
            filter=pars[1], pupil="GRISM" + pars[3][0], module=pars[2][-1]
        )

        sens_file = os.path.join(path, sens_file)
        # print('xx', sens_file, os.path.exists(sens_file))

        if os.path.exists(sens_file):
            msg = "grismconf.aXeConf: replace sensitivity curve with "
            msg += f"{sens_file}"
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

            si = utils.read_catalog(sens_file).copy()
            si["ERROR"] = si["SENSITIVITY"].max() * 0.01

            _tab = utils.GTable()
            _tab["WAVELENGTH"] = si["WAVELENGTH"].astype(float)
            _tab["SENSITIVITY"] = si["SENSITIVITY"].astype(float)
            _tab["ERROR"] = si["ERROR"].astype(float)

            self.sens["A"] = _tab


def load_grism_config(conf_file, warnings=True):
    """
    Load parameters from an aXe configuration file

    Parameters
    ----------
    conf_file : str
        Filename of the configuration file

    warnings : bool
        Print warnings to the terminal.

    Returns
    -------
    conf : `~grizli.grismconf.aXeConf`
        Configuration file object.  Runs `conf.get_beams()` to read the
        sensitivity curves.
    """
    if "V3/NIRCAM" in conf_file:
        conf = TransformGrismconf(conf_file)
        conf.get_beams()
    elif "V2/NIRCAM" in conf_file:
        conf = TransformGrismconf(conf_file)
        conf.get_beams()
    elif "V4/NIRCAM" in conf_file:
        conf = TransformGrismconf(conf_file)
        conf.get_beams()
    elif "V8/NIRCAM" in conf_file:
        conf = TransformGrismconf(conf_file)
        conf.get_beams()
    elif "V8.5/NIRCAM" in conf_file:
        conf = TransformGrismconf(conf_file)
        conf.get_beams()
    elif "V9/NIRCAM" in conf_file:
        conf = TransformGrismconf(conf_file)
        conf.get_beams()
    elif "specwcs" in conf_file:
        conf = TransformGrismconf(conf_file)
        conf.get_beams()
    else:
        conf = aXeConf(conf_file)
        conf.get_beams()

    # Preliminary hacks for on-sky NIRISS
    if "GR150" in conf_file:
        if 0:
            hack_niriss = 1.0 / 1.8 * 1.1

            msg = f"""
     ! Scale NIRISS sensitivity by {hack_niriss:.3f} to hack gain correction
     ! and match GLASS MIRAGE simulations. Sensitivity will be updated when
     ! on-sky data available
     """
            msg = f" ! Scale NIRISS sensitivity by {hack_niriss:.3f} prelim flux correction"
            utils.log_comment(utils.LOGFILE, msg, verbose=warnings)
        elif "F090W" in conf_file:
            # hack_niriss = 0.8
            hack_niriss = 1.0
        else:
            hack_niriss = 1.0

        for b in conf.sens:
            conf.sens[b]["SENSITIVITY"] *= hack_niriss
            if "ERROR" in conf.sens[b].colnames:
                conf.sens[b]["ERROR"] *= hack_niriss

        if ("F115W" in conf_file) | (".2212" in conf_file):
            pass
            # msg = f""" !! Shift F115W along dispersion"""
            # utils.log_comment(utils.LOGFILE, msg, verbose=warnings)
            # for b in conf.beams:
            #     #conf.conf[f'DYDX_{b}_0'][0] += 0.25
            #     conf.conf[f'DLDP_{b}_0'] -= conf.conf[f'DLDP_{b}_1']*0.5
        elif ("F090W" in conf_file) | (".2212" in conf_file):
            pass
        elif isinstance(conf, TransformGrismconf):
            # Don't shift new format files
            pass
        else:
            msg = f""" !! Shift {os.path.basename(conf_file)} along dispersion"""
            utils.log_comment(utils.LOGFILE, msg, verbose=warnings)
            for b in conf.beams:
                # conf.conf[f'DYDX_{b}_0'][0] += 0.25
                conf.conf[f"DLDP_{b}_0"] += conf.conf[f"DLDP_{b}_1"] * 0.5

                # For red galaxy
                conf.conf[f"DLDP_{b}_0"] += conf.conf[f"DLDP_{b}_1"] * 0.5
                # if 'F200W' in conf_file:
                #     conf.conf[f'DLDP_{b}_0'] += conf.conf[f'DLDP_{b}_1']*0.5

        #     _w = conf.sens['A']['WAVELENGTH']
        #     _w0 = (_w*conf.sens['A']['SENSITIVITY']).sum()
        #     _w0 /=  conf.sens['A']['SENSITIVITY'].sum()
        #     slope = 1.05 + 0.2 * (_w - _w0)/3000
        #     # print('xxx', conf_file, _w0)
        #     conf.sens['A']['SENSITIVITY'] *= slope

        if ("F150W" in conf_file) & (hack_niriss > 1.01):
            conf.sens["A"]["SENSITIVITY"] *= 1.08

        # Scale 0th orders in F150W
        if "F150W" in conf_file:  # | ('F200W' in conf_file):
            msg = f""" ! Scale 0th order (B) by an additional x 1.5"""
            utils.log_comment(utils.LOGFILE, msg, verbose=warnings)
            conf.sens["B"]["SENSITIVITY"] *= 1.5
            if "ERROR" in conf.sens["B"].colnames:
                conf.sens["B"]["ERROR"] *= 1.5

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


def download_jwst_crds_references(
    instruments=["NIRCAM", "NIRISS"],
    filters=[
        "F090W",
        "F115W",
        "F150W",
        "F200W",
        "F277W",
        "F356W",
        "F410M",
        "F444W",
        "F322W2",
        "F460M",
        "F480M",
    ],
    grisms=["GRISMR", "GRISMC", "GR150R", "GR150C"],
    modules=["A", "B"],
    context=DEFAULT_CRDS_CONTEXT,
    verbose=True,
):
    """
    Run `~grizli.jwst_utils.crds_reffiles` with filter and grism combinations to
    prefetch a bunch of reference files.

    Parameters
    ----------
    instruments : list
        List of instruments to download references for. 
        Default: ["NIRCAM", "NIRISS"].

    filters : list
        List of filters to download references for.
        E.g., ["F090W", "F115W", "F150W", ...].

    grisms : list
        List of grisms to download references for.
        Default: ["GRISMR", "GRISMC", "GR150R", "GR150C"].

    modules : list
        List of modules to download references for. Default: ["A", "B"].

    context : str
        CRDS context.

    verbose : bool
        Print messages to the terminal.

    """
    for instrument in instruments:
        for f in filters:
            for g in grisms:
                if instrument == "NIRCAM":
                    if (f < "F270") | g.startswith("GR150"):
                        continue

                    for m in modules:
                        _ = crds_reffiles(
                            instrument="NIRCAM",
                            filter=f,
                            pupil=g,
                            module=m,
                            reftypes=("photom", "specwcs"),
                            header=None,
                            context=context,
                            verbose=verbose,
                        )
                else:
                    if (f > "F270") | g.startswith("GRISM"):
                        continue

                    _ = crds_reffiles(
                        instrument="NIRISS",
                        filter=g,
                        pupil=f,
                        module="A",
                        reftypes=("photom", "specwcs"),
                        header=None,
                        context=context,
                        verbose=verbose,
                    )


class CRDSGrismConf:
    def __init__(
        self,
        file="references/jwst/nircam/jwst_nircam_specwcs_0136.asdf",
        get_photom=True,
        context=DEFAULT_CRDS_CONTEXT,
        **kwargs,
    ):
        """
        Helper object to replicate `grismconf` config files from CRDS products

        Parameters
        ----------
        file : str
            Filename of a CRDS ``specwcs`` file

        get_photom : bool
            Get sensitivity curves from the ``photom`` reference file

        context : str
            Explicit CRDS_CONTEXT

        Attributes
        ----------
        dm : `jwst.datamodels.NIRCAMGrismModel`
            DataModel of the ``specwcs`` reference

        meta : dict
            Metadata dictionary from `jwst.datamodels.NIRCAMGrismModel.meta.instance`

        dm_orders : list
            List of orders from `dm.orders`

        crds_parameters : dict
            CRDS parameter dictionary from `dm.get_crds_parameters()`

        dispx, dispy, displ : list, list, list
            Parameter polynomials from the reference datamodel

        SENS_data : dict
            Sensitivity data like ``{order: (wave_microns, sensitity)}``

        """
        from . import jwst_utils

        if context is not None:
            jwst_utils.CRDS_CONTEXT = context
            jwst_utils.set_crds_context(verbose=False, override_environ=True)

        self.file = file
        if file.startswith("references"):
            full_path = os.path.join(os.environ["CRDS_PATH"], file)
        elif os.path.exists(file):
            full_path = file
        elif "/references" in file:
            # Replace CRDS_PATH in absolute path
            full_path = os.path.join(
                os.environ["CRDS_PATH"], "references", file.split("references/")[1]
            )

        if not os.path.exists(full_path):
            # Try to download all of the JWST references
            download_jwst_crds_references(context=context)

        self.full_path = full_path

        self.initialize_from_datamodel()
        self.SENS = None
        self.SENS_data = None

        if get_photom:
            self.get_photom(**kwargs)
            self.load_new_sensitivity_curve(**kwargs)

    def initialize_from_datamodel(self):
        """
        Initialize polynomial objects from a `jwst.datamodel`
        """
        import copy
        import jwst.datamodels

        full_path = self.full_path

        if "nircam" in full_path:
            dm = jwst.datamodels.NIRCAMGrismModel(full_path)
        else:
            dm = jwst.datamodels.NIRISSGrismModel(full_path)

        self.meta = copy.deepcopy(dm.meta.instance)
        self.dm_orders = copy.deepcopy(dm.orders)
        self.crds_parameters = dm.get_crds_parameters()
        self.dispx = copy.deepcopy(dm.dispx)
        self.dispy = copy.deepcopy(dm.dispy)
        self.displ = copy.deepcopy(dm.displ)

    # @property
    # def meta(self):
    #     """metadata dictionary"""
    #     return self.dm.meta.instance

    @property
    def module(self):
        if self.instrument == "NIRCAM":
            return self.meta["instrument"]["module"]
        else:
            return None

    @property
    def filter(self):
        return self.meta["instrument"]["filter"]

    @property
    def pupil(self):
        return self.meta["instrument"]["pupil"]

    @property
    def grism(self):
        """
        Return filter for NIRISS, pupil for NIRCAM, filter for HST
        """
        if self.instrument in ("NIRCAM", "NIRISS"):
            if self.instrument == "NIRCAM":
                return self.pupil
            else:
                return self.filter
        else:
            # e.g., HST
            return self.filter

    @property
    def instrument(self):
        return self.meta["instrument"]["name"]

    @property
    def filter_grism(self):
        """Return combination of blocking filter + dispering element"""

        if self.instrument == "NIRISS":
            return (self.pupil, self.filter)
        else:
            return (self.filter, self.pupil)

    @property
    def instrument_setup(self):
        """Return combination of blocking filter, dispering element, (module)"""

        if self.instrument == "NIRISS":
            return (self.pupil, self.filter, None)
        else:
            return (self.filter, self.pupil, self.module)

    @property
    def orders(self):
        """String version of orders, like '+1', '+2', '0', '-1'"""

        orders = []
        for o in self.dm_orders:
            if o > 0:
                orders.append(f"+{o}")
            else:
                orders.append(f"{o}")

        return orders

    def DISPX(self, order, x0, y0, t):
        """
        Replicate grismconf.DISPX

        Evaluates dispersion polynomial from the reference datamodel
        for a given order in x direction.

        Parameters
        ----------
        order : str
            Order name like '+1', '0', '-1'

        x0, y0 : float or array-like
            Detector coordinates in the direct image

        t : float or array-like
            Evaluation point(s) of the independent trace variable

        Returns
        -------
        dispx : float or array-like
            x pixel along the trace

        """
        io = self.orders.index(order)
        dispx = self._eval_model(self.dispx[io], x0, y0, t)
        return dispx

    def DDISPX(self, order, x0, y0, t, dt=0.01):
        """
        Replicate grismconf.DDISPX

        Evaluates the derivative of the dispersion polynomial from the reference
        datamodel for a given order in x direction.

        Parameters
        ----------
        order : str
            Order name like '+1', '0', '-1'

        x0, y0 : float or array-like
            Detector coordinates in the direct image

        t : float or array-like
            Evaluation point(s) of the independent trace variable

        dt : float
            Delta t for finite difference.

        Returns
        -------
        ddispx : float or array-like
            Derivative of the x pixel along the trace

        """
        io = self.orders.index(order)
        v0 = self._eval_model(self.dispx[io], x0, y0, t)
        v1 = self._eval_model(self.dispx[io], x0, y0, t + dt)
        return (v1 - v0) / dt

    def DISPY(self, order, x0, y0, t):
        """
        Replicate grismconf.DISPY

        Evaluates dispersion polynomial from the reference datamodel for a given
        order in y direction.

        Parameters
        ----------
        order : str
            Order name like '+1', '0', '-1'

        x0, y0 : float or array-like
            Detector coordinates in the direct image

        t : float or array-like
            Evaluation point(s) of the independent trace variable

        Returns
        -------
        dispy : float or array-like
            y pixel along the trace

        """
        io = self.orders.index(order)
        dispy = self._eval_model(self.dispy[io], x0, y0, t)
        return dispy

    def DDISPY(self, order, x0, y0, t, dt=0.01):
        """
        Replicate grismconf.DDISPY

        Evaluates the derivative of the dispersion polynomial from the reference
        datamodel for a given order in y direction.

        Parameters
        ----------
        order : str
            Order name like '+1', '0', '-1'

        x0, y0 : float or array-like
            Detector coordinates in the direct image

        t : float or array-like
            Evaluation point(s) of the independent trace variable

        dt : float
            Delta t for finite difference.

        Returns
        -------
        ddispy : float or array-like
            Value(s) of the derivative of the y pixel along the trace.

        """
        io = self.orders.index(order)
        v0 = self._eval_model(self.dispy[io], x0, y0, t)
        v1 = self._eval_model(self.dispy[io], x0, y0, t + dt)
        return (v1 - v0) / dt

    def DISPXY(self, order, x0, y0, t):
        """
        Replicate grismconf.DISPXY (combination of DISPX, DISPY)

        Evaluates dispersion polynomials from the reference datamodel for a given
        order in x and y directions.
        
        Parameters
        ----------
        order : str
            Order name like '+1', '0', '-1'

        x0, y0 : float or array-like
            Detector coordinates in the direct image

        t : float or array-like
            Evaluation point(s) of the independent trace variable

        Returns
        -------
        dispx, dispy : float or array-like, float or array-like
            x and y pixels along the trace
            
        """
        io = self.orders.index(order)
        dispx = self._eval_model(self.dispx[io], x0, y0, t)
        dispy = self._eval_model(self.dispy[io], x0, y0, t)
        return dispx, dispy

    def DISPL(self, order, x0, y0, t):
        """
        Replicate grismconf.DISPL

        Evaluates dispersion polynomials from the reference datamodel for a given
        order along the dispersion axis.

        Parameters
        ----------
        order : str
            Order name like '+1', '0', '-1'

        x0, y0 : float or array-like
            Detector coordinates in the direct image

        t : float or array-like
            Evaluation point(s) of the independent trace variable

        Returns
        -------
        displ : float or array-like
            Wavelength value(s) along the trace, microns

        """
        io = self.orders.index(order)
        displ = self._eval_model(self.displ[io], x0, y0, t)
        return displ

    def DDISPL(self, order, x0, y0, t, dt=0.01):
        """
        Replicate grismconf.DDISPL

        Evaluates the derivative of the dispersion polynomial from the reference
        datamodel for a given order along the dispersion axis.

        Parameters
        ----------
        order : str
            Order name like '+1', '0', '-1'

        x0, y0 : float or array-like
            Detector coordinates in the direct image

        t : float or array-like
            Evaluation point(s) of the independent trace variable

        dt : float
            Delta t for finite difference.

        Returns
        -------
        ddispl : float or array-like
            Value(s) of the derivative of the dispersion polynomial.

        """
        io = self.orders.index(order)
        v0 = self._eval_model(self.displ[io], x0, y0, t)
        v1 = self._eval_model(self.displ[io], x0, y0, t + dt)
        return (v1 - v0) / dt

    def INVDISPX(self, order, x0, y0, dx, t0=np.linspace(-1, 2, 128), from_root=False):
        """
        Inverse DISPX

        Evaluates the inverse of the dispersion polynomial from the reference
        datamodel for a given order along the x-axis.

        Parameters
        ----------
        order : str
            Order name like '+1', '0', '-1'

        x0, y0 : float or array-like
            Detector coordinates in the direct image

        dx : float
            X coordinate where to interpolate the trace

        t0 : array-like
            1D evaluation grid for the inverse.

        from_root : bool
            Use polynomial roots to find the inverse.

        Returns
        -------
        t : float
            Independent variable value along the trace.

        """
        if from_root:
            func = self._root_inverse_model
        else:
            func = self._inverse_model

        io = self.orders.index(order)
        t = func(self.dispx[io], x0, y0, dx, t0=t0)
        return t

    def INVDISPY(self, order, x0, y0, dx, t0=np.linspace(-1, 2, 128), from_root=False):
        """
        Inverse DISPY

        Evaluates the inverse of the dispersion polynomial from the reference
        datamodel for a given order along the y-axis.

        Parameters
        ----------
        order : str
            Order name like '+1', '0', '-1'

        x0, y0 : float or array-like
            Detector coordinates in the direct image

        dx : float, array-like
            Y coordinate where to interpolate the trace
            
        t0 : array-like
            1D evaluation grid for the inverse interpolation.

        from_root : bool
            Use polynomial roots to find the inverse.

        Returns
        -------
        t : like ``dx``
            Independent variable value along the trace.

        """
        if from_root:
            func = self._root_inverse_model
        else:
            func = self._inverse_model

        io = self.orders.index(order)
        t = func(self.dispy[io], x0, y0, dx, t0=t0)
        return t

    def INVDISPL(self, order, x0, y0, dx, t0=np.linspace(-1, 2, 128), from_root=False):
        """
        Inverse DISPL

        Evaluates the inverse of the wavelength polynomial from the reference
        datamodel for a given order, reference position, and dispersed position
        along the dispersion axis.

        Parameters
        ----------
        order : str
            Order name like '+1', '0', '-1'

        x0, y0 : float
            Detector coordinates in the direct image

        dx : float, array-like
            Wavelengths where to interpolate the trace
            
        t0 : array-like
            1D evaluation grid for the inverse interpolation.

        from_root : bool
            Use polynomial roots to find the inverse.

        Returns
        -------
        t : like ``dx``
            Independent variable value along the trace.

        """
        if from_root:
            func = self._root_inverse_model
        else:
            func = self._inverse_model

        io = self.orders.index(order)
        t = func(self.displ[io], x0, y0, dx, t0=t0)
        return t

    def _eval_model(self, model, x0, y0, t, get_coeffs=False):
        """
        General function for evaluating model polynomials.

        Parameters
        ----------
        model : model object
            Example `astropy.modeling.Polynomial2D`.

        x0, y0 : float or array-like
            Detector coordinates in the direct image

        t : float or array-like
            Independent variable along the trace

        get_coeffs : bool
            Return polynomial coefficients instead of the evaluated value.

        Returns
        -------
        if get_coeffs:
            value : list
                Polynomial coefficients
        else:
            value : float
                Evaluated value(s) of the polynomial model.

        """
        import numpy as np

        if hasattr(model, "n_inputs"):
            # model is a Polynomial
            if model.n_inputs == 1:
                if get_coeffs:
                    value = model.parameters[::-1]
                else:
                    value = model(t)
            else:
                value = model(x0, y0)

        elif len(model) == 1:
            # model is a single-element list, probably Polynomial1D
            if model[0].n_inputs == 1:
                if get_coeffs:
                    value = model[0].parameters[::-1]
                else:
                    value = model[0](t)
            else:
                value = model[0](x0, y0)

        else:
            # model is a list
            _c = []
            for m in model:
                if m.n_inputs == 1:
                    _c.append(m(t))
                else:
                    _c.append(m(x0, y0))

            if get_coeffs:
                value = _c
            else:
                value = np.polynomial.Polynomial(_c)(t)

        return value

    def _root_inverse_model(self, model, x0, y0, dx, **kwargs):
        """
        Calculate roots of the polynomial model

        Parameters
        ----------
        model : model object
            Example `astropy.modeling.Polynomial2D`.

        x0, y0 : float or array-like
            Detector coordinates in the direct image

        dx : float
            Value to shift the constant term of the polynomial model by.

        Returns
        -------
        value : float
            Coefficients of the inverse polynomial model.

        """
        coeffs = self._eval_model(model, x0, y0, 0, get_coeffs=True)
        if hasattr(coeffs, "__len__"):
            coeffs[0] -= dx
            value = np.polynomial.Polynomial(coeffs).roots()[-1]
        else:
            value = coeffs

        return value

    def _inverse_model(self, model, x0, y0, dx, t0=np.linspace(-1, 2, 128)):
        """
        Inverse values interpolated from the forward model

        Parameters
        ----------
        model : model object
            Example `astropy.modeling.Polynomial2D`.

        x0, y0 : float or array-like
            Detector coordinates in the direct image

        dx : float, array-like
            Parameter value of the model

        t0 : array-like
            Evaluation grid for the polynomial model. Used for 1D models.

        Returns
        -------
        t : float
            Independent variable along the trace where ``t = model(x0, y0, t0)``
            
        """
        values = self._eval_model(model, x0, y0, t0)

        so = np.argsort(values)
        t = np.interp(dx, values[so], t0[so])
        return t

    def get_photom(
        self,
        xyt=(1024, 1024, 0.5),
        date=None,
        photom_file=None,
        verbose=False,
        **kwargs,
    ):
        """
        Load photom reference from CRDS and scale to grismconf / aXe convention

        Parameters
        ----------
        xyt : (float, float, float)
            Coordinate (x0, y0, t) where to evaluate the grism dispersion DLDP

        date : str, None
            Observation date in ISO format, e.g., '2023-01-01 00:00:00'.  If not
            specified, defaults to "now"

        photom_file : str
            Explicit filename of a CRDS ``photom`` reference file

        verbose : bool
            Print status message

        Returns
        -------
        SENS_data : dict
            Dict of ``{'order': (wave, sens)}``.  Also sets ``SENS_data``,
            ``SENS_dldp`` and ``SENS_xyt`` attributes.

        """
        import astropy.time
        import astropy.table
        import astropy.units as u
        import crds
        import jwst.datamodels

        if photom_file is None:
            cpars = self.crds_parameters

            if self.instrument == "NIRISS":
                cpars["meta.instrument.detector"] = "NIS"
                cpars["meta.exposure.type"] = "NIS_WFSS"
            else:
                cpars["meta.instrument.detector"] = f"NRC{self.module}LONG"
                cpars["meta.exposure.type"] = "NRC_WFSS"

            if date is None:
                date = astropy.time.Time.now().iso

            cpars["meta.observation.date"] = date.split()[0]
            cpars["meta.observation.time"] = date.split()[1]

            refs = crds.getreferences(cpars, reftypes=("photom",))

            if verbose:
                msg = f"Read photometry reference {refs['photom']} (date = '{date}')"
                print(msg)

            photom_file = refs["photom"]

        if self.instrument == "NIRCAM":
            ph = jwst.datamodels.NrcWfssPhotomModel(photom_file)
        else:
            ph = jwst.datamodels.NisWfssPhotomModel(photom_file)

        phot = astropy.table.Table(ph.phot_table)

        pixel_area = ph.meta.photometry.pixelarea_steradians

        self.sens_ref_file = refs["photom"]
        self.SENS_xyt = xyt
        self.SENS_dldp = {}
        self.SENS_data = {}

        for i, order in enumerate(self.orders):
            ix = phot["filter"] == self.filter
            ix &= phot["pupil"] == self.pupil
            ix &= phot["order"] == self.dm_orders[i]
            if ix.sum() == 0:
                msg = f"Order {order} = {self.dm_orders[i]} not found in "
                msg += f"{refs['photom']} for {self.filter} {self.pupil}"
                utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
                continue

            row = phot[ix]

            wave = np.squeeze(row["wavelength"].data)
            sens_fnu = np.squeeze(row["relresponse"].data)

            sens_fnu *= pixel_area * row["photmjsr"]

            # Dispersion, DLAM/DPIX
            dl = self.DDISPL(order, *xyt)
            dx = self.DDISPX(order, *xyt)
            dy = self.DDISPY(order, *xyt)
            dldp = dl / np.sqrt(dx ** 2 + dy ** 2)

            self.SENS_dldp[order] = dldp

            mask = (wave > 0) & (sens_fnu > 0)

            _flam_unit = u.erg / u.second / u.cm ** 2 / u.micron
            sens_flam = (sens_fnu[mask] * dldp * u.megaJansky).to(
                _flam_unit, equivalencies=u.spectral_density(wave[mask] * u.micron)
            )

            self.SENS_data[order] = [wave[mask], 1.0 / sens_flam.value]

        return self.SENS_data

    def load_new_sensitivity_curve(self, verbose=True, **kwargs):
        """
        Replace +1 NIRCam sensitivity curves with Nov 10, 2023 updates

        Files generated with the calibration data of P330E from program
        CAL-1538 (K. Gordon)

        Download the FITS files from the link below and put them in
        ``$GRIZLI/CONF/GRISM_NIRCAM/``.

        https://s3.amazonaws.com/grizli-v2/JWSTGrism/NircamSensitivity/index.html

        Parameters
        ----------
        verbose : bool
            Print messages to the terminal.
        """

        path = os.path.join(GRIZLI_PATH, "CONF", "GRISM_NIRCAM")
        meta = self.crds_parameters
        if meta["meta.instrument.name"] != "NIRCAM":
            msg = "load_new_sensitivity_curve: only defined for NIRCAM ({0})"
            utils.log_comment(
                utils.LOGFILE,
                msg.format(meta["meta.instrument.name"]),
                verbose=verbose
            )
            return None

        sens_base = "nircam_wfss_sensitivity_{filter}_{pupil}_{module}.10nov23.fits"
        sens_file = sens_base.format(
            filter=meta["meta.instrument.filter"],
            pupil=meta["meta.instrument.pupil"],
            module=meta["meta.instrument.module"],
        )

        sens_file = os.path.join(path, sens_file)

        if os.path.exists(sens_file):
            msg = "grismconf.CRDSGrismConf: replace sensitivity curve with "
            msg += f"{sens_file}"
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

            si = utils.read_catalog(sens_file)

            self.SENS_data["+1"] = [si["WAVELENGTH"] / 1.0e4, si["SENSITIVITY"]]
