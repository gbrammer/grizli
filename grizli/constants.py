import astropy.units as u

KMS = u.km/u.s
FLAMBDA_CGS = u.erg/u.s/u.cm**2/u.angstrom
FNU_CGS = u.erg/u.s/u.cm**2/u.Hz

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

