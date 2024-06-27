"""
update LW badpix table from C. Willott

- Add mask around edge of NIRCam detectors

https://github.com/gbrammer/grizli/issues/200
https://github.com/gbrammer/grizli/issues/233
"""

import os
import astropy.io.fits as pyfits
import numpy as np
from grizli import utils

bpix = {
    "along": np.zeros((2048, 2048), dtype=np.int16),
    "blong": np.zeros((2048, 2048), dtype=np.int16),
}

# Edge mask
nircam_edge = 8

for det in bpix:
    bpix[det][:nircam_edge, :] |= 1024
    bpix[det][-nircam_edge:, :] |= 1024
    bpix[det][:, :nircam_edge] |= 1024
    bpix[det][:, -nircam_edge:] |= 1024

willott_files = [
    "https://github.com/gbrammer/grizli/files/13878784/badpixlist_along_20240108.txt",
    "https://github.com/gbrammer/grizli/files/13878785/badpixlist_blong_20240108.txt",
    "https://github.com/user-attachments/files/15995936/badpixlist_along_20240626.txt",
    "https://github.com/user-attachments/files/15995938/badpixlist_blong_20240626.txt",
]

for file in willott_files:
    det = os.path.basename(file).split("_")[-2]
    new_data = utils.read_catalog(file, format="ascii")
    for x, y in zip(new_data["col1"], new_data["col2"]):
        bpix[det][y, x] |= 1

header = pyfits.Header()
header["INSTRUMENT"] = "NIRCAM"
header["TELESCOPE"] = "JWST"
header["NEDGE"] = (nircam_edge, "Number of masked pixels around edge")
header["EDGEVAL"] = (1024, "Edge DQ value")
header["BPVAL"] = (1, "Bad pixel DQ value")

for det in bpix:
    header["DETECTOR"] = ("NRC" + det).upper()
    out_file = f"nrc_badpix_240627_NRC{det.upper()}.fits.gz"
    print(f"NRC{det.upper()} N={(bpix[det] & 1 ).sum()}")
    pyfits.PrimaryHDU(header=header, data=bpix[det]).writeto(out_file, overwrite=True)

### Previous - add to earlier conservative bpix list
#
# old_list = ['nrc_badpix_231206_NRCALONG.fits.gz', 'nrc_badpix_231206_NRCBLONG.fits.gz']
# for new_file, old_file in zip(new_list, old_list):
#     det = os.path.basename(new_file).split('_')[1]
#
#     new_data = utils.read_catalog(new_file, format='ascii')
#
#     with pyfits.open(old_file) as im:
#         orig_bad = im[0].data.sum()
#
#         for x, y in zip(new_data['col1'], new_data['col2']):
#             im[0].data[y,x] |= 1
#
#         new_bad = im[0].data.sum()
#
#         msg = f'{det} prev = {orig_bad.sum()}  new = {new_bad.sum()}'
#         print(msg)
#
#     out_file = f'nrc_badpix_240112_NRC{det.upper()}.fits.gz'
#     im.writeto(out_file, overwrite=True)
