.. raw:: html

   <!---
   Hello there! If you want to contribute to Grizli for dummies, please follow the
   following format when adding a chapter to the book:

   1. Below the heading of each chapter, the name of the author who wrote that
   chapter is stated. It's more helpful if you can make your name a link to your
   GitHub page or personal website, such that if a reader is confused about
   something written in the chapter, they know how to contact you for further
   questions.

   2. Underneath the Author name for each chapter, please state the Grizli version
   you used for the task explained in that chapter (if relevant). Reasons for doing
   this are explained in Chapter 1 of Grizli for dummies.

   3. Place your footnotes at the end of each chapter.

   4. Put a navigation bar at the end of your chapter (see existing examples).

   If you follow the above steps, it will really make the editing process easier
   for me -- thank you!

   Jasleen Matharu 11th June 2020
   -->

Grizli for dummies
==================

**I was scared I’d forget these details, so I put them in a book.**

Author: `Jasleen Matharu <https://github.com/jkmatharu>`__

**If you would like to contribute to Grizli for dummies, please read the
commented section before the reStructuredText document begins to understand the
layout Grizli for dummies follows.**


Preface
=======

I never chose to write this book, it chose me. During my PhD and now
during my first Postdoc, I have been forced to learn and grasp many
intricate details regarding the usage of the *Grism redshift and line
analysis software* (Grizli) written by Gabriel Brammer [#]_ . The official
documentation and software can be found
`here <https://grizli.readthedocs.io/en/master/>`__.

Out of sheer fear that I would forget all the details I have been forced
to learn, or that I will waste away many hours frantically flipping
through my notebooks, emails and slack channels trying to remember how I
did something, I’ve chosen to put all my notes in this book.

The details in this book are not designed to help with the basics. This
book was written selfishly for myself, with the added bonus that if
there is anyone else out there that just can’t seem to figure something
out about Grizli from the official documentation or source code, they
might get lucky and find the solution here. Otherwise, the struggle is
real.


| *Jasleen Matharu*
| *4th May 2020*
| *during the COVID-19 Pandemic* [#]_ 

.. [#] I still haven’t met him in person.
.. [#] No, I did not decide to write a book because I had lots of free time
  on my hands.

.. _fake-cont:

--------------


.. contents::


.. _this-book:

**How to get the most out of this book**
========================================

This book follows a particular format to help you get the most out of
the information presented. **Below the heading of each chapter, the name
of the author who wrote that chapter will be stated**. This is so that
in case you are confused about anything in that chapter, you know who to
contact for queries or further questions. In some cases, the author’s
name will be a link that will either take you to their GitHub page or a
website of theirs with their contact details. Otherwise, we’re all
relatively famous, that I’m sure you can google or NASA ADS us and
you’ll find the most up-to-date email address for us, or physical
address to send your telegram by pigeon.

**Underneath the author, if relevant, the version of Grizli that was
used for that chapter will be stated**. This is particularly important,
because there are differences between different versions of Grizli,
which means Grizli may not behave the same way for the same task in
different versions. If you’re following a task outlined in this book and
you can’t quite figure out why it’s not working out for you, it might be
worth comparing your version of Grizli to the one used for that chapter
and check whether perhaps an update or downgrade will solve your problem
(I would recommend a downgrade as a last resort though).

--------------

:ref:`Table of Contents<fake-cont>` `|` :ref:`Next Chapter<installing-grizli>` >

--------------

.. _installing-grizli:

**Installing Grizli**
=====================

Author: `Jasleen Matharu <https://github.com/jkmatharu>`__

As you have probably seen from the official `installation
page <https://grizli.readthedocs.io/en/master/grizli/install.html>`__,
there is only one way to install Grizli: using the `conda`
environment. Don’t try to do it any other way if you want to ensure an
environment within which Grizli will work harmoniously. Remember, Grizli
is designed to work *within* the `astroconda` environment, which
itself is a `conda` environment within `anaconda`\ [#]_ .

Updating Grizli
---------------

You can update Grizli using pip [#]_ :

.. code:: bash

		$ pip install git+https://github.com/gbrammer/grizli.git

If that doesn’t work, a wise person [#]_ told me to:

1. Clone the environment to a local location.

2. Update as necessary with `git pull`.

3. Run `pip install` in the repository.

The above approach seems to behave better with versioning, and you may
want to clean out any earlier installations of the Grizli module from
your `site-packages` directory or wherever the module is getting
placed by `setup.py`. To find out where Grizli is installed on your
computer, in `python` you can do:

.. code:: python

		>>> import grizli
		>>> print('grizli location: {0}'.format(grizli.__file__))
		grizli location: /Users/gbrammer/miniconda3/envs/grizli-dev/lib/python3.6/site-packages/grizli/__init__.py

You may also need to re-do:

.. code:: python

       >>> from grizli import utils
       >>> utils.symlink_templates()

to get any new redshift fit templates that have been added to the
repository.

Re-installing Grizli
--------------------

Sometimes, something might get really screwed up on your computer that
Grizli just won’t work. You don’t know why, but before you pull every
single strand of hair out of your scalp, you get software rage and
decide you want to wipe Grizli out of existence.

For me, to accomplish this I had to remove Grizli and the `grizli-dev`
environment and re-install from scratch using the `conda` environment
method.

Deleting the Grizli environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Within the `astroconda` environment, I ran:

.. code:: bash

		$ conda env remove --name grizli-dev

which deletes the `grizli-dev` environment and everything in it.

--------------

.. [#] Environment-ception.
.. [#] As spoken by the Grizli God himself, Gabe Brammer.
.. [#] You guessed it, it was the Grizli God himself, Gabe Brammer.

--------------

< :ref:`Previous Chapter<installing-grizli>` `|` :ref:`Table of Contents<fake-cont>` `|` :ref:`Next Chapter<changes>` >

--------------

.. _changes:

**Changes between Grizli versions**
===================================

Author: `Jasleen Matharu <https://github.com/jkmatharu>`__

Version 0.9 versus 1.0
----------------------

Improvement in the grism/photometry scaling algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you happen to have processed some grism data including photometry [#]_ 
with Grizli version 0.9 and then 1.0, you may have noticed that your 1.0
extractions look more reliable. The one-dimensional model spectrum seems
to follow the data much better in your `full.png` files.

Let’s pretend you absolutely need to reproduce the 0.9 version fit for
whatever reason. You try to really constrain the redshift window around
the 0.9 version’s determined grism redshift. Nope. Still a much better
fit when you compare your new and old `full.png` files for the same
galaxy. What the hell is going on?

Turns out, the grism/photometry scaling got a serious upgrade, giving
you better quality fits whether you like it or not. In the words of Gabe
Brammer himself:

*"Before I was trying to fit the templates to the spectrum and
photometry and calculate a scaling based on that. The problem was that
the comparison had to be done at about the correct redshift, otherwise
lines being in the wrong place would compromise the fit. The new method
fits a more flexible spline function to the spectrum and tries to
integrate the broad-band flux density of the available filters that
overlap the fit, which it compares to the observed photometry. You still
need at least one filter that overlaps the available spectrum more or
less completely. One way around that could be defining an interpolated
filter in the photometric catalog derived from the photo-z fit. Say,
filling F140W with the template value for objects where it is otherwise
missing."*

--------------

.. [#] For example, you set `scale_photometry=1` when running the ``grizli.fitting.run_all`` function.

--------------

< :ref:`Previous Chapter<changes>` `|` :ref:`Table of Contents<fake-cont>` `|` :ref:`Next Chapter<accessing>` >

--------------

.. _accessing:

**Accessing the public database of reduced grism data**
=======================================================

| Author: `Jasleen Matharu <https://github.com/jkmatharu>`__
| Grizli version: `1.0-76-g71853af`

The database of reduced public HST grism data can be accessed with the
following information in `python` [#]_ :

.. code:: python

	>>> from grizli.aws import db
	>>> config = {'hostname':'grizdbinstance.c3prl6czsxrm.us-east-1.rds.amazonaws.com',
	      'username':'****',
	      'password':'****',
	      'database':'****',
	      'port':5432}
	>>> engine = db.get_db_engine(config=config)

--------------

.. [#] You didn’t honestly think I was going to publicise the login details, did you? If you require access, you need to ask Gabe Brammer nicely.

--------------

< :ref:`Previous Chapter<accessing>` `|` :ref:`Table of Contents<fake-cont>` `|` :ref:`Next Chapter<dim-thumbs>` >

--------------

.. _dim-thumbs:

**Creating thumbnails that are not the standard 80 x 80 pixels in** `full.fits`
===============================================================================

| Author: `Jasleen Matharu <https://github.com/jkmatharu>`__
| Grizli version: `1.0-76-g71853af`

In this chapter, I will walk you through how to create thumbnails in
your `full.fits` files with the dimensions of your choice.

If you already have existing `beams.fits` files you’ve generated, you
do not need to recreate them for this task, unless your beams aren’t
tall enough. For reference, I successfully created 189 x 189 pixel
thumbnails from existing beams that were used to create the standard 80
x 80 thumbnails in `full.fits`. What you will need is:

-  To load and initiate the relevant line templates for fitting the line
   fluxes:

   .. code:: python

		>>> templ0 = grizli.utils.load_templates(fwhm=1200, line_complexes=True,
		            stars=False, full_line_list=None,  continuum_list=None,
		            fsps_templates=True)

		>>> templ1 = grizli.utils.load_templates(fwhm=1200, line_complexes=False, 
			     stars=False, full_line_list=None, continuum_list=None,
		             fsps_templates=True)

-  **If you’re including photometry in your fit, do the following steps
   before the above**:

   1. Install `eazy-py <https://github.com/gbrammer/eazy-py>`__ (and
      import it in your `python` script with the line
      ``import eazy``), with the following parameters [#]_ defined in your
      `python` script:

      .. code:: python

                 params = {}
                 params['Z_STEP'] = 0.002
                 params['Z_MAX'] = 4
                 params['TEMPLATES_FILE'] = 'templates/fsps_full/tweak_fsps_QSF_12_v3.param'
                 params['PRIOR_FILTER'] = 205
                 params['MW_EBV'] = {'aegis':0.0066, 'cosmos':0.0148, 'goodss':0.0069, \
                                 'uds':0.0195, 'goodsn':0.0103}['goodsn']

   2. Acquire the `.translate` files for your field.

   3. Define the following parameters [#]_ for your field:

      .. code:: python

                 params['CATALOG_FILE'] = my_photometric_catalogue.cat
                 params['MAIN_OUTPUT_FILE'] = '{0}_3dhst.{1}.eazypy'.format('goodss', 'v4.4')

   4. Create a symlink to your `templates` directory with the
      following lines of `python` code:

      .. code:: python

                 import os
                 eazy.symlink_eazy_inputs(path=os.path.dirname(eazy.__file__)+'/data')

   5. Run the following line of `python` code:

      .. code:: python

                 ez = eazy.photoz.PhotoZ(param_file=None, translate_file=translate_file,
                         zeropoint_file=None, params=params, load_prior=True, load_products=False)

   6. **Then, after loading and initiating your line templates as shown
      in the first bullet point, run**:

      .. code:: python

                 from grizli.pipeline import photoz
                 ep = photoz.EazyPhot(ez, grizli_templates=templ0, zgrid=ez.zgrid)

.. _set-dimensions:

Setting the thumbnail dimensions
--------------------------------

The next line of code I’m going to show you is **the** line of the code.
The line of code that will allow you to fiddle with the properties of
your output thumbnails in `full.fits`. The default setting leads to
thumbnails in `full.fits` with a pixel scale of 0.1" and dimensions of
80 x 80 pixels:

.. code:: python

       pline = {'kernel': 'point', 'pixfrac': 0.2, 'pixscale': 0.1, 'size': 8, 'wcs': None}

Now, for different thumbnail dimensions, all you need to do is change
the value of `size`. With `pixscale=0.1`, an 8" x 8" thumbnail is 80
x 80 pixels. So, for example, if I wanted thumbnails with dimensions 189
x 189 pixels, I would set `size=18.9`.

Running your new fits with Grizli
---------------------------------

If you’re including photometry, then you must first do:

Otherwise...

--------------

.. [#] The values shown for the parameters are just examples. They may not
  be relevant to your particular data.
.. [#] The values shown for the parameters are just examples. They may not
  be relevant to your particular data.

--------------

< :ref:`Previous Chapter<dim-thumbs>` `|` :ref:`Table of Contents<fake-cont>` `|` :ref:`Next Chapter<reliable-thumbs>` >

--------------

.. _reliable-thumbs:

**Creating reliable direct image thumbnails**
=============================================

| Author: `Jasleen Matharu <https://github.com/jkmatharu>`__
| Grizli version: `1.0-76-g71853af` and `1.0.dev1458`


.. _full-fits-files:

The `full.fits` files
-----------------------

When one has run Grizli from end-to-end, perhaps following the
`Grizli-Pipeline <https://github.com/gbrammer/grizli/blob/master/examples/Grizli-Pipeline.ipynb>`__
notebook, you will find that you will have `root_id.full.fits` files
in your `root/Extractions/` folder. These contain thumbnails of the
direct images, emission line maps and associated contamination, weight [#]_ ,
PSFs and segmentation maps for the source in the field = `root` with
Object ID = `id`. These have been designed to work with
`GALFIT <https://users.obs.carnegiescience.edu/peng/work/galfit/galfit.html>`__.

.. _direct-image-full:

Direct image thumbnails in `full.fits`
----------------------------------------

Note, the direct image thumbnails in `full.fits` are in units of
electrons per second, but the emission line map thumbnails are in units
of 10\ :sup:`-17` ergs s\ :sup:`-1` cm\ :sup:`-2`. To convert the direct image thumbnails to the
same units as the emission line maps, you need the relevant `PHOTPLAM`
and `PHOTFLAM` values. These can be found as keywords in the header of
the direct image thumbnail extension (`DSCI`) in `full.fits`. If
not, this `StScI
website <https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/ir-photometric-calibration>`__
tabulates the values for the relevant HST filters.

**If you are conducting a study where you need to directly compare the
direct image thumbnails to the emission line map thumbnails, you cannot
use the direct image thumbnails in the** `root_id.full.fits` **files**.
This is because the direct images have been “blotted” [#]_ from the full
mosaic without accounting for the correct variance of the parent image.
The most reliable direct images can be generated by:

"*drizzling them from the original direct image FLTs to the same WCS and
with the same drizzle parameters used to generate the line map. The*
``grizli.aws.aws_drizzler.drizzle_images`` *function can help with
this."* [#]_ 

The above is not as straightforward as the author of this chapter
thought.

Generating reliable direct image thumbnails
-------------------------------------------

.. _with-phot:

Generating direct image thumbnails when your `_phot.fits` file is generated with Grizli
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To accomplish this monumental task, you will need to run the
``auto_script.make_rgb_thumbnails`` function in the `root/Prep/`
directory and you will need the following files in your `root/Prep/`
directory for it to work:

-  The necessary [#]_ `flt.fits` [#]_ files in the `root/Prep/` directory.
   **If you are not sure about this, please check how you queried the
   HST archive when doing your Grizli extractions. For the most reliable
   direct image thumbnails, you need ALL the available** `flt.fits`
   **files available for your field, not necessarily those pertaining to
   your proposal ID (especially for well-studied fields such as those in
   3D-HST/CANDELS). If you know you’ve added new** `flt.fits` **files
   since doing your Grizli run, you need to generate a new**
   `root_groups.npy` **file — read** :ref:`this section <create-groups-file>` **NOW.**

-  The `root_phot.fits` file in the `root/Prep/` directory.

-  The `root_visits.npy` file in the `root/Prep/` directory.

-  The `root-ir_seg.fits` file to be in your `root/Prep/` directory
   (if you want a corresponding segmentation map thumbnail to be
   generated).

Reliable direct image thumbnails can be created with the function
``auto_script.make_rgb_thumbnails``. An example of its usage can be seen
in `In [40]:` of the
`Grizli-Pipeline <https://github.com/gbrammer/grizli/blob/master/examples/Grizli-Pipeline.ipynb>`__
notebook. For a given field (or `root`), you will need to run this
function in the `root/Prep/` directory. If you set the keyword
`use_line_wcs = True`, the function will look in `root/Extractions/`
for the `full.fits` files associated with the object IDs you request
and match the WCS and drizzle parameters of the thumbnails to those of
the `LINE` extensions. Also, set the keyword `skip = False` if the
function doesn’t do anything, since `skip = True` will skip over objects
where a `root_id.thumb.fits` file already exists. The
`root_id.thumb.fits` files will be saved in the `root/Prep/`
directory.

For example, to make a single thumbnail for one of the objects in the
`Grizli-Pipeline <https://github.com/gbrammer/grizli/blob/master/examples/Grizli-Pipeline.ipynb>`__
demo, run [#]_:

.. code:: python

	auto_script.make_rgb_thumbnails(root='j033216m2743', ids=[424], use_line_wcs=True) 
 

However, the story does not end there.

Corresponding segmentation map thumbnails
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You may suddenly realise you need corresponding segmentation maps for
your newly-generated direct image thumbnails [#]_ . Have no fear, you can
generate them when you run ``auto_script.make_rgb_thumbnails`` as
explained above, but you need to set the keyword
`make_segmentation_figure=True`. For a segmentation map to be
successfully generated, you need the `root-ir_seg.fits` file to be in
your `root/Prep/` directory.

Other important things to note
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  By default, the `min_filters` keyword is set to `2`. Sometimes,
   you only have imaging for the object in one filter. So if you want
   ``auto_script.make_rgb_thumbnails`` to work in that instance, you’ll
   need to explicitly set `min_filters = 1`.

.. _without-phot:

Generating direct image thumbnails when your photometric catalog is external to Grizli
---------------------------------------------------------------------------------------

To accomplish this task, you will need to run the
``grizli.aws.aws_drizzler.drizzle_images`` in your `root/Prep/`
directory and you will need the following files for it to work:

-  The necessary [#]_ `flt.fits` [#]_ files in the `root/Prep/` directory.

-  The `_groups.npy` file in your `root/Prep/` directory.

-  The segmentation map for your field in the `root/Prep/` directory
   (if you want a corresponding segmentation map thumbnail).

-  The photometric catalog for your field, **with the Object ID column
   named as** `'number'` [#]_ (if you want a corresponding segmentation
   map thumbnail).

The method to create reliable direct image thumbnails outlined in :ref:`the
previous sub-section <with-phot>` will only work if you used a
photometric catalog that was generated by Grizli (a `root_phot.fits`
file in your `root/Prep/` directory) throughout your reduction
process. If this is not the case, then you my friend, are in a bit of a
pickle [#]_ .

No you’re not. You have another option. In certain cases, you will not
need Grizli to generate a photometric catalog, because you’re working on
a well-studied field which already has a much more complete, external
photometric catalog. You may think “Aw, heck. Let me just use Grizli to
create it anyway." **No. Stop.** For well-studied fields such as those
part of CANDELS and/or 3D-HST – or any other field that has obtained HST
imaging external to grism programs – this may be problematic. It all
depends on how you queried the HST archive when you ran Grizli on your
dataset (look at the section”Query the HST archive" on `In [5]:` of
the
`Grizli-Pipeline <https://github.com/gbrammer/grizli/blob/master/examples/Grizli-Pipeline.ipynb>`__
notebook.). Did you just extract the data based on your Proposal ID? Did
you use the overlap query and if you did, did you make sure you obtained
ALL the possible relevant imaging for your objects of interest? If you
did not query the HST archive for ALL the relevant HST imaging for your
targets in existence, then the mosaics Grizli will construct from these
– on which Grizli runs SExtractor to generate its `root_phot.fits`
file – will be incomplete. You need to query the HST archive again,
making sure to download ALL the necessary `flt.fits` files
corresponding to the filter you want the direct image to be in. Then,
you can either:

1. Use Grizli to generate a new `root_phot.fits` file, or

2. Use an existing photometric catalog (if it exists).

Well don’t just stare at me, hoping I’ll make the decision for you. I’m
now going to explain how you can generate reliable direct image
thumbnails using an existing photometric catalog, assuming you have now
downloaded all the relevant `flt.fits` files you need **and have
generated your** `_groups.npy` **file. If not, go read** :ref:`this section <create-groups-file>` **now.** You can join me back here
afterwards.

When you have an existing photometric catalog, it is best to by-pass the
whole process of constructing the `root_phot.fits` file with Grizli
and run the ``grizli.aws.aws_drizzler.drizzle_images``\ [#]_ function by
hand.

So, "how do I run this function?!", I hear you scream. Below I show you
how I call the function:

.. code:: python

   from grizli.aws import aws_drizzler

   new_thumbnail=aws_drizzler.drizzle_images(label=label_name, ra=RA, dec=DEC, master=field,
                   single_output=True, make_segmentation_figure=False, pixscale=0.1,
                   pixfrac=0.2, size=18.9, filters=['f105w'], remove=False, include_ir_psf=True)

-  `label_name` is the name of the output files you want. For me it
   was the `field` name followed by the Object ID number. e.g.
   `‘ERSPRIME_42362’`. But you can set this to whatever you fancy.

-  `field` is just the field name, for me it was `‘ERSPRIME’`.
   Again, as far as I can see, the user can set this to whatever they
   want.

-  No idea what `single_output` is [#]_ .

-  Now, it may seem strange to you that I set
   `make_segmentation_figure = False`. I want to generate segmentation
   map thumbnails, but when I set this to `True`, my segmentation map
   thumbnails were not generated. This is because Grizli tries to find
   the segmentation map in the cloud and not the local directory. I
   explain in :ref:`this subsection <without-phot-seg>` how to
   generate the segmentation map thumbnail when your segmentation map is
   in your local directory.

-  The `pixscale`, `pixfrac` and `size` arguments are the ones you
   need to be careful about here. In the instance where you have a
   photometric catalog generated by Grizli (see :ref:`this
   section <with-phot>`), these arguments were taken care of for
   you because you ran that function on the `full.fits` files and
   could just set the argument `use_line_wcs = True`. The function
   would then just use the drizzle parameters of the `LINE` extensions
   in `full.fits` and generate direct image thumbnails with these
   drizzle parameters. Not here. **Here you need to make sure you are
   setting the correct drizzle parameters**. If you are not sure what
   these are, you should look back at (or find out) the value of these
   parameters when you generated your `full.fits` files (for an
   example, see :ref:`this section <set-dimensions>`). Alternatively, you
   should be able to find `PIXFRAC` and `PIXASEC` keywords in the
   headers of almost all the extensions in `full.fits`. Similarly to
   get the size, just multiply the value for `NAXIS1` in the header by
   the `PIXASEC`.

-  You can specify which `filters` you want direct images for. If you
   don’t specify this, the function will generate direct image
   thumbnails in all filters available for that object, which means you
   need to make sure ALL the `flt.fits` file for that object/field are
   present in your `root/Prep/` directory. Otherwise, you will only
   need the ones corresponding to the filter you specify.

-  If `remove = True`, the function will delete the `flt.fits` files
   it uses after it has run.

-  If you would like a corresponding PSF thumbnail, you should set
   `include_ir_psf = True`. [#]_ 

.. _without-phot-seg:

Corresponding segmentation map thumbnails
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As mentioned in the :ref:`above section <without-phot>`, setting
`make_segmentation_figure = True` when running the function
``grizli.aws.aws_drizzler.drizzle_images`` did not generate a
segmentation map thumbnail for me. To generate my segmentation map
thumbnails, I ran the function
``grizli.aws.aws_drizzler.segmentation_figure`` *after* I ran
``grizli.aws.aws_drizzler.drizzle_images``, like so:

.. code:: python

       segmap = aws_drizzler.segmentation_figure(label_name, cat_phot, seg_file)

-  `cat_phot` is your photometric catalog. Remember, **for your
   segmentation map thumbnail to be generated, the Object ID column
   needs to have the title** `number` [#]_ .

-  `seg_file` is the filename of your segmentation map `.fits` file.
   I put this file in my `root/Prep/` directory.

Tips
^^^^

| For me, after generating the relevant files, the functions
  ``grizli.aws.aws_drizzler.drizzle_images`` and
| ``grizli.aws.aws_drizzler.segmentation_figure`` would sometimes break.
  This breaking was unrelated to the generation of the relevant
  thumbnails. So to ensure the functions ran on my entire sample in my
  code, I used the python ``try`` and ``except`` conditions like so:

::

       flag=False
       try:
           new_thumbnail = aws_drizzler.drizzle_images(label=label_name, ra=RA, dec=DEC,
                                        master=field, single_output=True,
                                        make_segmentation_figure=False,
                                        pixscale=0.1, pixfrac=0.2, size=18.9, 
                                        filters=['f105w'],
                                        remove=False, include_ir_psf=True)

       except:
           flag=True

       flag=False

       try:
           segmap = aws_drizzler.segmentation_figure(label_name, cat_phot, seg_file)
       except:
           flag=True



.. _create-groups-file:

Creating your own `_groups.npy` file
--------------------------------------

If you are working on a well-studied field, like, I don’t know, maybe
one of the 3D-HST/CANDELS fields [#]_ , you may need to generate a new
`_groups.npy` file to obtain the most reliable direct image
thumbnails. This all depends on how you queried the HST archive for your
Grizli run (look at the section "Query the HST archive" on `In [5]:`
of the
`Grizli-Pipeline <https://github.com/gbrammer/grizli/blob/master/examples/Grizli-Pipeline.ipynb>`__
notebook.). Did you just extract the data based on your Proposal ID? Did
you use the overlap query and if you did, did you make sure you obtained
ALL the possible relevant imaging for your objects of interest? **The
instructions in** :ref:`this chapter <with-phot>` **implicitly
assume that if your** `_phot.fits` **file was generated with Grizli, it
was generated using all the HST imaging available for that field in that
filter.** This may not be the case, so I implore you, for what feels
like the millionth time, to go back and check you have all the necessary
`_flt.fits` files in existence for the filter within which you want to
create reliable direct image postage stamps. If you are using the method
outlined in :ref:`this chapter <with-phot>` to create your reliable
direct image postage stamps, as far as I am aware, the `_groups.npy`
can be used interchangeably with the `_visits.npy` file. So if you
have to generate a new `_groups.npy` file (as is about to be
explained), you should be able to use it instead of the `_visits.npy`
file. Just make sure you get rid of the old file, or move it into a
different directory.

Once you have downloaded all the necessary `_flt.fits` files, the
`python` function below [#]_ will generate your new `_groups.npy` in
the local directory, with an example at the end of how to call it:

::

   import os
   import numpy as np

   field = 'my_beautiful_fieldname'

   def make_local_groups(path_to_flt='./', verbose=True, output_file='local_filter_groups.npy'):
       """
       Make a "groups" dictionary with lists of FLT exposures separated by
       filter.
       """
       import glob

       import astropy.io.fits as pyfits
       import astropy.wcs as pywcs

       from shapely.geometry import Polygon

       from grizli import utils

       # FLT files
       files = glob.glob(os.path.join(path_to_flt, '*fl[tc].fits'))
       files.sort()

       groups = {}
       for file in files:

           im = pyfits.open(file)

           # THE FOLLOWING LINE NEEDS TO HAVE .LOWER() AT THE END OTHERWISE
           # THE RESULTING FILE WON'T WORK
           filt = utils.get_hst_filter(im[0].header).lower()

           # UVIS
           if ('_flc' in file) & os.path.basename(file).startswith('i'):
               filt += 'U'

           if filt not in groups:
               groups[filt] = {}
               groups[filt]['filter'] = filt
               groups[filt]['files'] = []
               groups[filt]['footprints'] = []
               groups[filt]['awspath'] = []

           fpi = None
           for i in [1,2]:
               if ('SCI',i) in im:
                   wcs = pywcs.WCS(im['SCI',i].header, fobj=im)
                   if fpi is None:
                       fpi = Polygon(wcs.calc_footprint())
                   else:
                       fpi = fpi.union(Polygon(wcs.calc_footprint()))

           groups[filt]['files'].append(file)
           groups[filt]['footprints'].append(fpi)
           groups[filt]['awspath'].append(None)

           if verbose:
               cosd = np.cos(wcs.wcs.crval[1]/180*np.pi)
               print('{0} {1:>7} {2:.1f}'.format(file, filt, fpi.area*cosd*3600))

       if output_file is not None:
           np.save(output_file, [groups])

       return groups


   new_group_file = make_local_groups(path_to_flt='', verbose=True, 
                                      output_file=field+'_filter_groups.npy')

Obviously change the default field name otherwise you’re going to look
like a right idiot.

--------------

.. [#] The `DWHT` and `LINEWHT` extensions are indeed inverse variance
  maps, where σ = 1 / √weight. σ can be used as a sigma image with
  GALFIT.
.. [#] Going from the *undistorted* mosaic to a distorted mosaic is
  “blotting”. Going in the opposite direction is “drizzling”. The
  individual images that get spat out of the Telescope are drizzled to
  some tangent point, leading to an undistorted mosaic. In
  `full.fits`, the `DSCI` image you see has been taken from the
  undistorted mosaic and put back into a distorted frame. So basically,
  the pixel positions (and probably the pixel values) in the `DSCI`
  `full.fits` extension are not reliable. Still don’t understand? Well
  don’t shoot the messenger.
.. [#] As spoken by the Grizli God himself, Gabe Brammer.
.. [#] At least the ones corresponding to the filter for which you want
  direct image thumbnail for. Note, in older (before ~May 2020) versions
  of Grizli, you would have needed ALL the `flt.fits` files for a
  particular field, otherwise the code would break.
.. [#] These files contain images of each HST pointing/exposure.
.. [#] As spoken by the Grizli God himself, Gabe Brammer.
.. [#] This most definitely was not me.
.. [#] You only need the `flt.fits` files corresponding to the filter you
  want the direct image to be in.
.. [#] These files contain images of each HST pointing/exposure.
.. [#] Otherwise the segmentation map thumbnail will not be generated.
  It’s just the way of the code, deal with it.
.. [#] No, not a `python` pickle.
.. [#] So that’s what Gabe meant in :ref:`this
  section <direct-image-full>`!
.. [#] A reminder that this book wasn’t written by people who wrote
  Grizli.
.. [#] If a PSF thumbnail is not generated, check you have the relevant
  PSF files in your `grizli/CONF` directory and can open them. For
  example, when generating F105W reliable direct image thumbnails, I
  needed to be able to open the file `PSFSTD_WFC3IR_F105W.fits`. Mine
  for some reason was corrupt :( .
.. [#] Otherwise the segmentation map thumbnail will not be generated.
  It’s just the way of the code, deal with it.
.. [#] This most definitely did not happen to me.
.. [#] As generously given to me (and then adapted by me) by our Grizli
  God, Gabe Brammer.

--------------

< :ref:`Previous Chapter<reliable-thumbs>` `|` :ref:`Table of Contents<fake-cont>` `|` :ref:`Next Chapter<emission-maps>` >

--------------

.. _emission-maps:

**Notes about emission line map thumbnails**
============================================

| Author: `Jasleen Matharu <https://github.com/jkmatharu>`__
| Grizli version: `1.0-76-g71853af`

-  Pixel values are in units of 10\ :sup:`-17` ergs s\ :sup:`-1` cm\ :sup:`-2`.

-  You do not need to apply the associated contamination maps to them –
   the `CONTAM` maps just show you where the contamination is. The
   contamination has already been removed [#]_ from the `LINE` extensions.

--------------

.. [#] If there is residual contamination left in the `LINE` extension, this means Grizli failed to remove it. You may have to apply your own contamination removal techniques or if possible, see if you can use the associated `CONTAM` map to mask the problematic regions.

--------------

< :ref:`Previous Chapter<emission-maps>` `|` :ref:`Table of Contents<fake-cont>` `|` :ref:`Next Chapter<fluxes-widths>` >

--------------


.. _fluxes-widths:


**Emission line fluxes and equivalent widths**
==============================================

| Author: `Jasleen Matharu <https://github.com/jkmatharu>`__
| Grizli version: `1.0-76-g71853af`

In older versions of Grizli, Grizli used to spit out a catalog with line flux *and* equivalent width measurements. A description of the keywords for equivalent widths in such catalogs can be found in :ref:`this chapter <output-catalog>`. In newer versions, such as the one used for this chapter (yep, why don't you glance back up near the title of this chapter to check), only line fluxes and their errors are stated in the output catalog. The equivalent widths seem to have vanished. Gah!

No they haven't. The rest-frame equivalent widths (including their 16th, 50th and 84th percentiles) along with their errors can be found in the header of the `COVAR` extension in `full.fits` (see :ref:`this section <full-fits-files>` for more on `full.fits` files). You will also find *all* the line fluxes and their errors there too for that particular source. Emission line fluxes and their errors can also be found in the headers of their corresponding `LINE` extensions in `full.fits`.

For the same dataset, I have found that the same line has the same number in the `COVAR` header of `full.fits` for every object in that dataset. Check this is true for you of course, before you take my word for it.

--------------

< :ref:`Previous Chapter<fluxes-widths>` `|` :ref:`Table of Contents<fake-cont>` `|` :ref:`Next Chapter<output-catalog>` >

--------------

.. _output-catalog:

**The output Grizli catalogue** [#]_ 
====================================

Author: `Jasleen Matharu <https://github.com/jkmatharu>`__

-  `ew50_Ha` is the median of the Hα equivalent width Probability
   Density Function (PDF).

-  `ewhw_Ha` is the "half-width", so something like the 1σ uncertainty
   on `ew50_Ha`.


Grizli does not fit for resolved lines in the grism spectra, so there is
no parameter for the velocity line width. For all but broad-line AGN
(approximately ≥ 1000 km s\ :sup:`-1`), the lines are unresolved [#]_ .

--------------

.. [#] Yes, I am British. The word 'catalogue' does not end at the 'g', obviously \*eye roll\*.  
.. [#] All of the above, as said by the Grizli God himself, Gabe Brammer.

--------------

< :ref:`Previous Chapter<output-catalog>` `|` :ref:`Table of Contents<fake-cont>` `|`

--------------
