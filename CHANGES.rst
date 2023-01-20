0.1 (test)
----------

- Bleeding edge development version

0.9.0
-----

- Bounded fits by default

0.10.0
------

- Fix bug in 1D wave

0.11.0
------

- Refactored parameter files

0.13.0
------

- Add AWS scripts

1.0
---

- First production version

1.1
----------

- Better PEP8 compatibility throughout
- Fix bug on SEP `tot_corr` total correction that wasn't using the correct   
  pixel scale
- Bug fixes

1.1.1
-----

- Bug for tag version

1.3
---
Tagged version

1.6.0-68-g1096b89
-----------------
- Increase precision on ra/dec in reference catalogs.  Before was sometimes
  limited to 5 decimal places, which is essentially the size of a NIRCam SW
  pixel.
  
1.6.0-70-g618329f
-----------------
- decrease binary dilation of NIRCam bad pixel mask to one pixel

1.6.0-73-ge84d650
-----------------
- Remove writing a few temporary files to /tmp
- Close out some dangling pyfits.open objects

1.7.1
-----
- Modernize setup

1.7.3
-----
- Replace `descartes` dependency with updated `sregion`

1.7.4
-----
- fix bug in versioning

1.7.6
-----
- Updated NIRCam LW bad pixel masks