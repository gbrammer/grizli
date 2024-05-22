import numpy as np
import astropy.units as u

KMS = u.km/u.s
FLAMBDA_CGS = u.erg/u.s/u.cm**2/u.angstrom
FNU_CGS = u.erg/u.s/u.cm**2/u.Hz

# Filter footprints
PLUS_FOOTPRINT = np.array([[0,1,0], [1,0,1], [0,1,0]]) > 0
CORNER_FOOTPRINT = (~PLUS_FOOTPRINT)
CORNER_FOOTPRINT[1,1] = False

JWST_DQ_FLAGS = [
    "DO_NOT_USE",
    "OTHER_BAD_PIXEL",
    "UNRELIABLE_SLOPE",
    "UNRELIABLE_BIAS",
    "NO_SAT_CHECK",
    "NO_GAIN_VALUE",
    "HOT",
    "WARM",
    "DEAD",
    "RC",
    "LOW_QE",
]

