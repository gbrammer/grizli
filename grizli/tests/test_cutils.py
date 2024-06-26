import unittest

import numpy as np

from grizli import utils
from grizli.utils_c import interp
from grizli.utils_numba import interp as interp_numba

from grizli.utils_c import disperse
from grizli.utils_numba import disperse as disperse_numba


def test_cinterp():
    """
    Simple interpolation
    """
    xarr = np.array([0.0,1.0,2.0])
    yarr = np.array([0.0,1.0,0.0])
    for base_module in interp, interp_numba:
        result = base_module.interp_c(np.array([0.5]), xarr, yarr)
        assert np.allclose(result, 0.5)


def test_cinterp_conserve():
    """
    Linear interpolation conserving the integral 
    """
    xarr = np.arange(1., 3., 0.0001)
    yarr = (np.abs(xarr-2.) <= 0.1)*1.
    
    np.random.seed(1)
    xlr = np.random.rand(10)*2+1
    xlr.sort()
    
    for base_module in interp, interp_numba:
        ylr = base_module.interp_conserve_c(xlr, xarr, yarr)
    
        assert np.allclose(np.trapz(ylr, xlr), np.trapz(yarr, xarr))
    
    ylr_slow = interp.interp_conserve(xlr, xarr, yarr)
    assert np.allclose(np.trapz(ylr_slow, xlr), np.trapz(yarr, xarr))


def test_disperse_c():
    """
    Grism dispersion function
    """
    for disperse_module in [disperse, disperse_numba]:
        # Array shapes from a NIRCam F444W example
        sh = (128, 128)
        sh_beam = (128, 1544)
        x0 = np.array([63, 63])
    
        modelf = np.zeros(sh_beam, dtype=np.float32).flatten()
        seg = np.zeros(sh, dtype=np.float32)
    
        id = 1.

        yy, xx = np.indices(sh)
        R = np.sqrt((xx-62)**2 + (yy-62)**2)
        Rmax = 10
        seg += R < Rmax

        nx = sh_beam[1] - sh[0]
        dxpix = np.arange(nx, dtype=int) + sh[0] // 2 - 1
        ytrace_beam = (np.arange(nx) - nx/2) * 0.5 / 200
        yfrac_beam = ytrace_beam - np.floor(ytrace_beam)
        dyc = np.cast[int](ytrace_beam+20)-20+1
        idx = np.arange(modelf.size, dtype=np.int64).reshape(sh_beam)
        flat_index = idx[dyc + x0[0], dxpix]
        sens_curve = yfrac_beam**0

        ########
        # single pixel
        thumb = seg*0.
        thumb[63,63] = 1.
    
        modelf *= 0.
    
        status = disperse_module.disperse_grism_object(
            thumb,
            seg,
            np.float32(id),
            flat_index,
            yfrac_beam.astype(np.float64),
            sens_curve.astype(np.float64),
            modelf, 
            x0,
            np.array(sh, dtype=np.int64),
            x0,
            np.array(sh_beam, dtype=np.int64),
        )                         
    
        model2d = modelf.reshape(sh_beam)
        model_trace = model2d[:,x0[1]:-(x0[1]+2)]
    
        # "left" edge of 2D model is zero
        assert np.allclose(model2d[:,:x0[1]], 0.)
    
        npix = (model_trace > 0).sum(axis=0)
        assert npix.max() >= 1
        assert npix.max() <= 2
    
        # Pixels across the trace should sum to one
        assert np.allclose(model_trace.sum(axis=0), 1.0)
    
        ########
        # A "source" with flux across the segment
        thumb = seg*1.
    
        modelf *= 0.
    
        status = disperse_module.disperse_grism_object(
            thumb,
            seg,
            np.float32(id),
            flat_index,
            yfrac_beam.astype(np.float64),
            sens_curve.astype(np.float64),
            modelf, 
            x0,
            np.array(sh, dtype=np.int64),
            x0,
            np.array(sh_beam, dtype=np.int64),
        )                         
    
        model2d = modelf.reshape(sh_beam)
        model_trace = model2d[:, x0[1] + Rmax : -(x0[1] + 2 + Rmax)]
    
        # Each location along the spectrum should see the "whole" thumb
        assert np.allclose(model_trace.sum(axis=0), thumb.sum())
    
        #######
        # Unresolved "emission line" in a single spectral bin should give the thumbnail
        line_spectrum = sens_curve * 0
        line_spectrum[sh_beam[1] // 2] = 1
        modelf *= 0.
    
        status = disperse_module.disperse_grism_object(
            thumb,
            seg,
            np.float32(id),
            flat_index,
            yfrac_beam.astype(np.float64),
            line_spectrum.astype(np.float64),
            modelf, 
            x0,
            np.array(sh, dtype=np.int64),
            x0,
            np.array(sh_beam, dtype=np.int64),
        )                         
    
        model2d = modelf.reshape(sh_beam)
        model_trace = model2d[:, x0[1] + Rmax : -(x0[1] + 2 + Rmax)]
    
        assert model_trace.sum() == thumb.sum()
    
    