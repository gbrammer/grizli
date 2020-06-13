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

3. Since markdown sucks at automatically organising your footnotes for you,
treat each chapter like its own document -- start with footnote numnber 1 when
you start writing your chapter. Place your footnotes at the end of each chapter.
When you start a new chapter, start from footnote number 1 again. See some
existing chapters that are consecutive and have footnotes in each to see what I
mean.

4. Don't forget to add your chapter (and sub-sections) to the Table of contents.

If you follow the above steps, it will really make the editing process easier
for me -- thank you!

Jasleen Matharu 11th June 2020
-->

Grizli for dummies
=======
***I was scared I'd forget these details, so I put them in a book***

Authors: [Jasleen Matharu](https://github.com/jkmatharu), [Vicente Estrada-Carpenter](https://github.com/Vince-ec)

**If you would like to contribute to Grizli for dummies, please read the commented section before the Markdown document begins to understand the layout Grizli for dummies follows.**

---

Preface
=======

I never chose to write this book, it chose me. During my PhD and now
during my first Postdoc, I have been forced to learn and grasp many
intricate details regarding the usage of the *Grism redshift and line
analysis software* (Grizli) written by Gabriel Brammer<a href="#gabe" id="gabe_1"><sup>1</sup></a>. The official
documentation and software can be found [here](https://grizli.readthedocs.io/en/master/).

Out of sheer fear that I would forget all the details I have been forced
to learn, or that I will waste away many hours frantically flipping
through my notebooks, emails and slack channels trying to remember how I
did something, I've chosen to put all my notes in this book.

The details in this book are not designed to help with the basics. This
book was written selfishly for myself, with the added bonus that if
there is anyone else out there that just can't seem to figure something
out about Grizli from the documentation or source code, they might get
lucky and find the solution here. Otherwise, the struggle is real.


*Jasleen Matharu\
4th May 2020\
during the COVID-19 Pandemic<a href="#covid" id="covid_1"><sup>2</sup></a>*

<a id="gabe" href="#gabe_1"><sup>1</sup></a> I still haven't met him in person.   
<a id="covid" href="#covid_1"><sup>2</sup></a> No, I did not decide to write a book because I had lots of free
    time on my hands.

---

# Table of contents
1. [How to get the most out of this book](#get_from_book)
2. [Installing Grizli](#installing_grizli)
    1. [Updating Grizli](#updating_grizli)
    2. [Re-installing Grizli](#reinstalling_grizli)
        1. [Deleting the Grizli environment](#deleting_the_grizli_environment)
3. [Changes between Grizli versions](#grizli_versions)
    1. [Version 0.9 versus 1.0](#version_0.9_versus_1.0)
        1. [Improvement in the grism/photometry scaling algorithm](#scaling_algorithm)
4. [Accessing the public database of reduced grism data](#public_database)

---

<a name="get_from_book"></a>

How to get the most out of this book
====================================

This book follows a particular format to help you get the most out of
the information presented. **Below the heading of each chapter, the name
of the author who wrote that chapter will be stated**. This is so that
in case you are confused about anything in that chapter, you know who to
contact for queries or further questions. We're all relatively famous,
that I'm sure you can google or NASA ADS us and you'll find the most
up-to-date email address for us, or physical address to send your
telegram by pigeon.

**Underneath the author, if relevant, the version of Grizli that was
used for that chapter will be stated in small font**. This is
particularly important, because there are differences between different
versions of Grizli, which means Grizli may not behave the same way for
the same task in different versions. If you're following a task outlined
in this book and you can't quite figure out why it's not working out for
you, it might be worth comparing your version of Grizli to the one used
for that chapter and check whether perhaps an update or downgrade will
solve your problem (I would recommend a downgrade as a last resort
though).

<a name="installing_grizli"></a>

Installing Grizli
=================

Author: [Jasleen Matharu](https://github.com/jkmatharu)

As you have probably seen from the official [installation page](https://grizli.readthedocs.io/en/master/grizli/install.html), there is
only one way to install Grizli: using the `conda` environment. Don't try to do
it any other way if you want to ensure an environment within which Grizli will
work harmoniously. Remember, Grizli is designed to work *within* the
`astroconda` environment, which itself is a `conda` environment within
`anaconda`<a href="#anaconda" id="anaconda_1"><sup>1</sup></a>.


<a name="updating_grizli"></a>

Updating Grizli
---------------

You can update Grizli using pip<a href="#pip" id="pip_1"><sup>2</sup></a>:

`pip install git+https://github.com/gbrammer/grizli.git`

If that doesn't work, a wise person<a href="#wise" id="wise_1"><sup>3</sup></a> told me to:

1.  Clone the environment to a local location.

2.  Update as necessary with `git pull`.

3.  Run `pip install` in the repository.

The above approach seems to behave better with versioning, and you may
want to clean out any earlier installations of the Grizli module from
your `site-packages` directory or wherever the module is getting placed
by `setup.py`. To find out where Grizli is installed on your computer,
in `python` you can do:

        >> import grizli
        >> print(`grizli location: {0}'.format(grizli.__file__))
        /Users/gbrammer/miniconda3/envs/grizli-dev/lib/python3.6/site-packages/grizli/__init__.py

You may also need to re-do:

        from grizli import utils
        utils.symlink_templates()

to get any new redshift fit templates that have been added to the
repository.

<a name="reinstalling_grizli"></a>

Re-installing Grizli
--------------------

Sometimes, something might get really screwed up on your computer that
Grizli just won't work. You don't know why, but before you pull every
single strand of hair out of your scalp, you get software rage and
decide you want to wipe Grizli out of existence.

For me, to accomplish this I had to remove Grizli and the `grizli-dev`
environment and re-install from scratch using the `conda` environment
method.

<a name="deleting_the_grizli_environment"></a>

### Deleting the Grizli environment

Within the `astroconda` environment, I ran:

        conda env remove --name grizli-dev

which deletes the `grizli-dev` environment and everything in it.


<a id="anaconda" href="#anaconda_1"><sup>1</sup></a> Environment-ception.  
<a id="pip" href="#pip_1"><sup>2</sup></a> As spoken by the Grizli God himself, Gabe Brammer.  
<a id="wise" href="#wise_1"><sup>3</sup></a> You guessed it, it was the Grizli God himself, Gabe Brammer.  

<a name="changes_between_grizli_versions"></a>

Changes between Grizli versions
===============================

Author: [Jasleen Matharu](https://github.com/jkmatharu)

<a name="version_0.9_versus_1.0"></a>

Version 0.9 versus 1.0
----------------------

<a name="scaling_algorithm"></a>

### Improvement in the grism/photometry scaling algorithm

If you happen to have processed some grism data including photometry<a href="#inc_photometry" id="inc_photometry_1"><sup>1</sup></a>
with Grizli version 0.9 and then 1.0, you may have noticed that your 1.0
extractions look more reliable. The one-dimensional model spectrum seems
to follow the data much better in your `full.png` files.

Let's pretend you absolutely need to reproduce the 0.9 version fit for
whatever reason. You try to really constrain the redshift window around
the 0.9 version's determined grism redshift. Nope. Still a much better
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
missing.\"*

<a id="inc_photometry" href="#inc_photometry_1"><sup>1</sup></a> For example, you set `scale_photometry=1` when running the `grizli.fitting.run_all` function.

<a name="public_database"></a>

Accessing the public database of reduced grism data
===================================================

Author: [Jasleen Matharu](https://github.com/jkmatharu)  
Grizli version: `1.0-76-g71853af`

The database of reduced public HST grism data can be accessed with the
following information in `python`<a href="#database" id="database_1"><sup>1</sup></a>:

        from grizli.aws import db

        config = {`hostname':`grizdbinstance.c3prl6czsxrm.us-east-1.rds.amazonaws.com',
              `username':`****',
              `password':`****',
              `database':`****',
              `port':5432}

    engine = db.get_db_engine(config=config)

<a id="database" href="#database_1"><sup>1</sup></a> You didn't honestly think I was going to publicise the login details, did you? If you require access, you need to ask Gabe Brammer nicely. Perhaps throw in a "Please Sir" too.

Creating thumbnails that are not the standard 80 x 80 pixels in `full.fits`
================================================================================

Author: [Jasleen Matharu](https://github.com/jkmatharu)  
Grizli version: `1.0-76-g71853af`

In this chapter, I will walk you through how to create thumbnails in
your `full.fits` files with the dimensions of your choice.

If you already have existing `beams.fits` files you've generated, you do
not need to recreate them for this task, unless your beams aren't tall
enough. For reference, I successfully created 189 x 189 pixel
thumbnails from existing beams that were used to create the standard
80 x 80 thumbnails in `full.fits`. What you will need is:

-   To load and initiate the relevant line templates for fitting the
    line fluxes:

            templ0 = grizli.utils.load_templates(fwhm=1200, line_complexes=True,
                        stars=False, full_line_list=None,  continuum_list=None,
                        fsps_templates=True)

            templ1 = grizli.utils.load_templates(fwhm=1200, line_complexes=False, stars=False,
                                             full_line_list=None, continuum_list=None,
                                             fsps_templates=True)


-   **If you're including photometry in your fit, do the following steps
    before the above**:

    1.  Install [`eazy-py`](https://github.com/gbrammer/eazy-py) (and
        import it in your `python` script with the line `import eazy`),
        with the following parameters<a href="#eazy" id="eazy_1"><sup>1</sup></a> defined in your `python`
        script:

                    params = {}
                    params[`Z_STEP'] = 0.002
                    params[`Z_MAX'] = 4
                    params[`TEMPLATES_FILE'] = `templates/fsps_full/tweak_fsps_QSF_12_v3.param'
                    params[`PRIOR_FILTER'] = 205
                    params[`MW_EBV'] = {`aegis':0.0066, `cosmos':0.0148, `goodss':0.0069, \
                                    `uds':0.0195, `goodsn':0.0103}[`goodsn']


    2.  Acquire the `.translate` files for your field.

    3.  Define the following parameters<a href="#photcat" id="photcat_1"><sup>2</sup></a> for your field:

                    params[`CATALOG_FILE'] = my_photometric_catalogue.cat
                    params[`MAIN_OUTPUT_FILE'] = `{0}_3dhst.{1}.eazypy'.format(`goodss', `v4.4')


    4.  Create a symlink to your `templates` directory with the
        following lines of `python` code:

                    import os
                    eazy.symlink_eazy_inputs(path=os.path.dirname(eazy.__file__)+`/data')


    5.  Run the following line of `python` code:

                    ez = eazy.photoz.PhotoZ(param_file=None, translate_file=translate_file,
                            zeropoint_file=None, params=params, load_prior=True, load_products=False)


    6.  **Then, after loading and initiating your line templates as
        shown in the first bullet point, run**:

                    from grizli.pipeline import photoz
                    ep = photoz.EazyPhot(ez, grizli_templates=templ0, zgrid=ez.zgrid)


Setting the thumbnail dimensions {#thumbnail_dimensions}
--------------------------------

The next line of code I'm going to show you is **the** line of the code.
The line of code that will allow you to fiddle with the properties of
your output thumbnails in `full.fits`. The default setting leads to
thumbnails in `full.fits` with a pixel scale of $0.1"$ and dimensions of
$80\times80$ pixels:

        pline = {`kernel': `point', `pixfrac': 0.2, `pixscale': 0.1, `size': 8, `wcs': None}

Now, for different thumbnail dimensions, all you need to do is change
the value of `size`. With `pixscale=0.1`, an $8"\times8"$ thumbnail is
$80\times80$ pixels. So, for example, if I wanted thumbnails with
dimensions $189\times189$ pixels, I would set `size=18.9`.

Running your new fits with Grizli
---------------------------------

If you're including photometry, then you must first do:

Otherwise\...

<a id="eazy" href="#eazy_1"><sup>1</sup></a> The values shown for the parameters are just examples. They may not be relevant to your particular data.  
<a id="photcat" href="#photcat_1"><sup>2</sup></a> The values shown for the parameters are just examples. They may not be relevant to your particular data.


Creating reliable direct image thumbnails
=========================================

0pt [Jasleen Matharu]{.smallcaps}

0pt Grizli version: `1.0-76-g71853afand 1.0.dev1458`

The `full.fits` files
---------------------

When one has run Grizli from end-to-end, perhaps following the
[Grizli-Pipeline](#https://github.com/gbrammer/grizli/blob/master/examples/Grizli-Pipeline.ipynb)
notebook, you will find that you will have `root_id.full.fits` files in
your `root/Extractions/` folder. These contain thumbnails of the direct
images, emission line maps and associated contamination, weight[^10],
PSFs and segmentation maps for the source in the field = `root` with
Object ID = `id`. These have been designed to work with
[GALFIT](#https://users.obs.carnegiescience.edu/peng/work/galfit/galfit.html).

Direct image thumbnails in `full.fits` {#direct_full.fits}
--------------------------------------

Note, the direct image thumbnails in `full.fits` are in units of
electrons per second, but the emission line map thumbnails are in units
of $\times10^{-17}$ ergs s$^{-1}$ cm$^{-2}$. To convert the direct image
thumbnails to the same units as the emission line maps, you need the
relevant `PHOTPLAM` and `PHOTFLAM` values. These can be found as
keywords in the header of the direct image thumbnail extension in
`full.fits`. If not, this [StScI
website](#https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/ir-photometric-calibration)
tabulates the values for the relevant HST filters.

If you are conducting a study where you need to *directly* compare the
direct image thumbnails to the emission line map thumbnails, you cannot
use the direct image thumbnails in the `root_id.full.fits` files. This
is because the direct images have been "blotted\"[^11] from the full
mosaic without accounting for the correct variance of the parent image.
The most reliable direct images can be generated by:

"*drizzling them from the original direct image FLTs to the same WCS and
with the same drizzle parameters used to generate the line map. The*
`grizli.aws.aws_drizzler.drizzle_images` *function can help with
this.[^12]\"*

The above is not as straightforward as the author of this chapter
thought.

Generating reliable direct image thumbnails
-------------------------------------------

### Generating direct image thumbnails when your `_phot.fits` file is generated with Grizli {#thumbnails_rgb}

To accomplish this monumental task, you will need to run the
`auto_script.make_rgb_thumbnails` function in the `root/Prep/` directory
and you will need the following files in your `root/Prep/` directory for
it to work:

-   The necessary[^13] `flt.fits`[^14] files in the `root/Prep/`
    directory. **If you are not sure about this, please check how you
    queried the HST archive when doing your Grizli extractions. For the
    most reliable direct image thumbnails, you need ALL the available
    `flt.fits` files available for your field, not necessarily those
    pertaining to your proposal ID (especially for well-studied fields
    such as those in 3D-HST/CANDELS). If you know you've added new
    `flt.fits` files since doing your Grizli run, you need to generate a
    new `root_groups.npy` file -- go read
    Section [6.3.3](#groups_file){reference-type="ref"
    reference="groups_file"} NOW.**

-   The `root_phot.fits` file in the `root/Prep/` directory.

-   The `root_visits.npy` file in the `root/Prep/` directory.

-   The `root-ir_seg.fits` file to be in your `root/Prep/` directory (if
    you want a corresponding segmentation map thumbnail to be
    generated).

Reliable direct image thumbnails can be created with the function
`auto_script.make_rgb_thumbnails`. An example of its usage can be seen
in `In [40]:` of the
[Grizli-Pipeline](#https://github.com/gbrammer/grizli/blob/master/examples/Grizli-Pipeline.ipynb)
notebook. For a given field (or `root`), you will need to run this
function in the `root/Prep/` directory. If you set the keyword
`use_line_wcs=True`, the function will look in `root/Extractions/` for
the `full.fits` files associated with the object IDs you request and
match the WCS and drizzle parameters of the thumbnails to those of the
`LINE` extensions. Also, set the keyword `skip=False` if the function
doesn't do anything, since `skip=True` will skip over objects where a
`root_id.thumb.fits` file already exists. The `root_id.thumb.fits` files
will be saved in the `root/Prep/` directory.

For example, to make a single thumbnail for one of the objects in the
[Grizli-Pipeline](#https://github.com/gbrammer/grizli/blob/master/examples/Grizli-Pipeline.ipynb)
demo, run:

` auto_script.make_rgb_thumbnails(root=‘j033216m2743’, ids=[424], use_line_wcs=True)`[^15]

However, the story does not end there.

#### Corresponding segmentation map thumbnails

You may suddenly realise you need corresponding segmentation maps for
your newly-generated direct image thumbnails.[^16] Have no fear, you can
generate them when you run `auto_script.make_rgb_thumbnails` as
explained above, but you need to set the keyword
`make_segmentation_figure=True`. For a segmentation map to be
successfully generated, you need the `root-ir_seg.fits` file to be in
your `root/Prep/` directory.

#### Other important things to note

-   By default, the `min_filters` keyword is set to `2`. Sometimes, you
    only have imaging for the object in one filter. So if you want
    `auto_script.make_rgb_thumbnails` to work in that instance, you'll
    need to explicitly set `min_filters=1`.

### Generating direct image thumbnails when your photometric catalog is external to Grizli {#thumbnails_external_photcat}

To accomplish this task, you will need to run the
`grizli.aws.aws_drizzler.drizzle_images` in your `root/Prep/` directory
and you will need the following files for it to work:

-   The necessary[^17] `flt.fits`[^18] files in the `root/Prep/`
    directory.

-   The `_groups.npy` file in your `root/Prep/` directory.

-   The segmentation map for your field in the `root/Prep/` directory
    (if you want a corresponding segmentation map thumbnail).

-   The photometric catalog for your field, **with the Object ID column
    named as `‘number’`**[^19] (if you want a corresponding segmentation
    map thumbnail).

The method to create reliable direct image thumbnails outlined in
Section [6.3.1](#thumbnails_rgb){reference-type="ref"
reference="thumbnails_rgb"} will only work if you used a photometric
catalog that was generated by Grizli (a `root_phot.fits` file in your
`root/Prep/` directory.) throughout your reduction process. If this is
not the case, then you my friend, are in a bit of a pickle[^20].

No you're not. You have another option. In certain cases, you will not
need Grizli to generate a photometric catalog, because you're working on
a well-studied field which already has a much more complete, external
photometric catalog. You may think "Aw, heck. Let me just use Grizli to
create it anyway.\" **No. Stop.** For well-studied fields such as those
part of CANDELS and/or 3D-HST -- or any other field that has obtained
HST imaging external to grism programs -- this may be problematic. It
all depends on how you queried the HST archive when you ran Grizli on
your dataset (look at the section "Query the HST archive\" on `In [5]:`
of the
[Grizli-Pipeline](#https://github.com/gbrammer/grizli/blob/master/examples/Grizli-Pipeline.ipynb)
notebook.). Did you just extract the data based on your Proposal ID? Did
you use the overlap query and if you did, did you make sure you obtained
ALL the possible relevant imaging for your objects of interest? If you
did not query the HST archive for ALL the relevant HST imaging for your
targets in existence, then the mosaics Grizli will construct from these
-- on which Grizli runs SExtractor to generate its `root_phot.fits` file
-- will be incomplete. You now have two options. You could:

1.  Query the HST archive again, make sure you download ALL the
    necessary `flt.fits` files corresponding to the filter you want the
    direct image to be in and use Grizli to generate a new
    `root_phot.fits` file.

2.  Query the HST archive again, make sure you download ALL the
    necessary `flt.fits` files corresponding to the filter you want the
    direct image to be in and use an existing photometric catalog (if it
    exists).

I'm now going to explain how you can generate reliable direct image
thumbnails using an existing photometric catalog, assuming you have now
downloaded all the relevant `flt.fits` files you need **and have
generated your \_groups.npy file. If not, go read
Section [6.3.3](#groups_file){reference-type="ref"
reference="groups_file"} NOW.**

In such a situation, it is best to by-pass the whole process of
constructing the `root_phot.fits` file with Grizli and run the
`grizli.aws.aws_drizzler.drizzle_images`[^21] function by hand.

So, "how do I run this function?!\", I hear you scream. Below I show you
how I call the function:

    from grizli.aws import aws_drizzler

    new_thumbnail=aws_drizzler.drizzle_images(label=label_name, ra=RA, dec=DEC, master=field,
                    single_output=True, make_segmentation_figure=False, pixscale=0.1,
                    pixfrac=0.2, size=18.9, filters=[`f105w'], remove=False, include_ir_psf=True)

-   `label_name` is the name of the output files you want. For me it was
    the `field` name followed by the Object ID number. e.g.
    `‘ERSPRIME_42362’`. But you can set this to whatever you fancy.

-   `field` is just the field name, for me it was `‘ERSPRIME’`. Again,
    as far as I can see, the user can set this to whatever they want.

-   No idea what `single_output` is[^22].

-   Now, it may seem strange to you that I set
    `make_segmentation_figure=False`. I want to generate segmentation
    map thumbnails, but when I set this to `True`, my segmentation map
    thumbnails were not generated. This is because Grizli tries to find
    the segmentation map in the cloud and not the local directory. I
    explain in an upcoming subsubsubsection how to generate the
    segmentation map thumbnail when your segmentation map is in your
    local directory.

-   The `pixscale`, `pixfrac` and `size` arguments are the ones you need
    to be careful about here. In the instance where you have a
    photometric catalog generated by Grizli
    (Section [6.3.1](#thumbnails_rgb){reference-type="ref"
    reference="thumbnails_rgb"}), these arguments were taken care of for
    you because you ran that function on the `full.fits` files and could
    just set the argument `use_line_wcs=True`. The function would then
    just use the drizzle parameters of the `LINE` extensions in
    `full.fits` and generate direct image thumbnails with these drizzle
    parameters. Not here. **Here you need to make sure you are setting
    the correct drizzle parameters**. If you are not sure what these
    are, you should look back at (or find out) the value of these
    parameters when you generated your `full.fits` files (see
    Section [5.1](#thumbnail_dimensions){reference-type="ref"
    reference="thumbnail_dimensions"} for example). Alternatively, you
    should be able to find `PIXFRAC` and `PIXASEC` keywords in the
    headers of almost all the extensions in `full.fits`. Similarly to
    get the size, just multiply the value for `NAXIS1` in the header by
    the `PIXASEC`.

-   You can specify which `filters` you want direct images for. If you
    don't specify this, the function will generate direct image
    thumbnails in all filters available for that object, which means you
    need to make sure ALL the `flt.fits` file for that object/field are
    present in your `root/Prep/` directory. Otherwise, you will only
    need the ones corresponding to the filter you specify.

-   If `remove=True`, the function will delete the `flt.fits` files it
    uses after it has run.

-   If you would like a corresponding PSF thumbnail, you should set
    `include_ir_psf=True`.

#### Corresponding segmentation map thumbnails

As mentioned above in
Section [6.3.2](#thumbnails_external_photcat){reference-type="ref"
reference="thumbnails_external_photcat"}, setting
`make_segmentation_figure=True` when running the function
`grizli.aws.aws_drizzler.drizzle_images` did not generate a segmentation
map thumbnail for me. To generate my segmentation map thumbnails, I ran
the function `grizli.aws.aws_drizzler.segmentation_figure` *after* I ran
`grizli.aws.aws_drizzler.drizzle_images`, like so:

        segmap=aws_drizzler.segmentation_figure(label_name, cat_phot, seg_file)

-   `cat_phot` is your photometric catalog. Remember, **for your
    segmentation map thumbnail to be generated, the Object ID column
    needs to have the title `‘number’`**[^23].

-   `seg_file` is the filename of your segmentation map `.fits` file. I
    put this file in my `root/Prep/` directory.

#### Tips

For me, after generating the relevant files, the functions
`grizli.aws.aws_drizzler.drizzle_images` and\
`grizli.aws.aws_drizzler.segmentation_figure` would sometimes break.
This breaking was unrelated to the generation of the relevant
thumbnails. So to ensure the functions ran on my entire sample in my
code, I used the python `try` and `except` conditions like so:

        flag=False
        try:

            new_thumbnail=aws_drizzler.drizzle_images(label=label_name, ra=RA, dec=DEC,
                                         master=field, single_output=True,
                                         make_segmentation_figure=False,
                                         pixscale=0.1, pixfrac=0.2, size=18.9, filters=['f105w'],
                                         remove=False, include_ir_psf=True)

        except:
            flag=True

        flag=False

        try:

            segmap=aws_drizzler.segmentation_figure(label_name, cat_phot, seg_file)

        except:
            flag=True

### Creating your own `_groups.npy` file {#groups_file}

If you are working on a well-studied field, like, I don't know, maybe
one of the 3D-HST/CANDELS fields[^24], you may need to generate a new
`_groups.npy` file to obtain the most reliable direct image thumbnails.
This all depends on how you queried the HST archive for your Grizli run
(look at the section "Query the HST archive\" on `In [5]:` of the
[Grizli-Pipeline](#https://github.com/gbrammer/grizli/blob/master/examples/Grizli-Pipeline.ipynb)
notebook.). Did you just extract the data based on your Proposal ID? Did
you use the overlap query and if you did, did you make sure you obtained
ALL the possible relevant imaging for your objects of interest? **The
instructions in Section [6.3.1](#thumbnails_rgb){reference-type="ref"
reference="thumbnails_rgb"} implicitly assume that if your `_phot.fits`
file was generated with Grizli, it was generated using all the HST
imaging available for that field in that filter.** This may not be the
case, so I implore you, for what feels like the millionth time, to go
back and check you have all the necessary `_flt.fits` files in existence
for the filter within which you want to create reliable direct image
postage stamps. If you are using the method outlined in
Section [6.3.1](#thumbnails_rgb){reference-type="ref"
reference="thumbnails_rgb"} to create your reliable direct image postage
stamps, as far as I am aware, the `_groups.npy` can be used
interchangeably with the `_visits.npy` file. So if you had to generate a
new `_groups.npy` file, you should be able to use it instead of the
`_visits.npy` file. Just make sure you get rid of the old file, or move
it into a different directory.

Once you have downloaded all the necessary `_flt.fits` files, the
`python` function below[^25] will generate your new `_groups.npy` in the
local directory, with an example at the end of how to call it:

    import os
    import numpy as np

    field="my_beautiful_fieldname"

    def make_local_groups(path_to_flt=`./', verbose=True, output_file=`local_filter_groups.npy'):
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
        files = glob.glob(os.path.join(path_to_flt, `*fl[tc].fits'))
        files.sort()

        groups = {}
        for file in files:

            im = pyfits.open(file)
            # THE FOLLOWING LINE NEEDS TO HAVE .LOWER() AT THE END OTHERWISE THE
            #RESULTING FILE WON'T WORK
            filt = utils.get_hst_filter(im[0].header).lower()

            # UVIS
            if (`_flc' in file) & os.path.basename(file).startswith(`i'):
                filt += `U'

            if filt not in groups:
                groups[filt] = {}
                groups[filt][`filter'] = filt
                groups[filt][`files'] = []
                groups[filt][`footprints'] = []
                groups[filt][`awspath'] = []

            fpi = None
            for i in [1,2]:
                if (`SCI',i) in im:
                    wcs = pywcs.WCS(im[`SCI',i].header, fobj=im)
                    if fpi is None:
                        fpi = Polygon(wcs.calc_footprint())
                    else:
                        fpi = fpi.union(Polygon(wcs.calc_footprint()))

            groups[filt][`files'].append(file)
            groups[filt][`footprints'].append(fpi)
            groups[filt][`awspath'].append(None)

            if verbose:
                cosd = np.cos(wcs.wcs.crval[1]/180*np.pi)
                print(`{0} {1:>7} {2:.1f}'.format(file, filt, fpi.area*cosd*3600))

        if output_file is not None:
            np.save(output_file, [groups])

        return groups



    new_group_file=make_local_groups(path_to_flt=`', verbose=True, output_file=field+`_filter_groups.npy')

Obviously change the default field name otherwise you're going to look
like a right idiot.

Notes about emission line map thumbnails
========================================

0pt [Jasleen Matharu]{.smallcaps}

0pt Grizli version: `1.0-76-g71853af`

-   Pixel values are in units of
    $\times10^{-17}$ ergs s$^{-1}$ cm$^{-2}$.

-   You do not need to apply the associated contamination maps to them
    -- the `CONTAM` maps just show you where the contamination is. The
    contamination has already been removed[^26] from the `LINE`
    extensions.

The output Grizli catalogue[^27]
================================

0pt [Jasleen Matharu]{.smallcaps}

-   `ew50_Ha` is the median of the H$\upalpha$ equivalent width
    Probability Density Function (PDF).

-   `ewhw_Ha` is the "half-width\", so something like the $1\sigma$
    uncertainty on `ew50_Ha`.

Grizli does not fit for resolved lines in the grism spectra, so there is
no parameter for the velocity line width. For all but broad-line AGN
($\gtrsim 1000$ km s$^{-1}$), the lines are unresolved[^28].

Working with beams
==================

0pt [Vicente Estrada-Carpenter]{.smallcaps}

-   Beams are the most fundamental product from the Grizli reductions,
    so working with these will give you the greatest control over your
    data

-   Due to the complex nature of grism contamination subtraction  10$\%$
    of galaxies will have residual contamination

-   Here I will explain how to clean and work with beams

The MultiBeam object
--------------------

The first thing you'll need to do is become familiar with the MultiBeam
object. This function can be read in from grizli.multifit. A galaxies
beams can be read in directly (with all the reduction info) using the
output BEAM.fits files. Using MultiBeam you can fit for redshift, fit
with templates, extract line and continuum fits, make plots, and in
general manipulate the beams.

working with multiple grisms
----------------------------

If a galaxy has multiple grism spectra then its best to split them up.
This is because each of the spectra will be scaled differently and
attempting to fit over both simultaneously doesn't work out well. By
working with them separately different nuisance parameters can be
applied to each spectra, generating better fits.

The easiest way to split the data is to first identify which beam
belongs to which instrument, this can be done with the MultiBeam. Once
you've created a list of each spectra you can then generate a MultiBeam
object for each spectra.

How to clean beams
------------------

The easiest way to identify residual contamination is to plot each of
the individual beams in 1D. Contamination will show itself as either a
fake emission line that only shows up in few beams (likely only in
ORIENT) or major offsets in the continuum (again only seen in a few
beams. Some beams have extremely large offsets in a few data points. If
this trend is only seen in one beam and not in an ORIENT set, then it
will likely average out and it isn't necessary to remove (unless there
only a few beams).

Once you've identified the contamination it can be removed by setting
the flux to 0 in the beams, or by just tossing the entire beam if the
contamination is too much to deal with. I've done this by using a
parameter file which identifies which beams are contaminated, whether
the beam needs to have the contamination masked out or whether it needs
to be omitted.

Forward modelling with Grizli
=============================

0pt [Vicente Estrada-Carpenter]{.smallcaps}

Fitting in 1D
-------------

Fitting in 2D
-------------




[^5]: You guessed it, it was the Grizli God himself, Gabe Brammer.

[^6]: For example, you set `scale_photometry=1` when running the
    `grizli.fitting.run_all` function.

[^7]: You didn't honestly think I was going to publicise the login
    details, did you?

[^8]: The values shown for the parameters are just examples. They may
    not be relevant to your particular data.

[^9]: The values shown for the parameters are just examples. They may
    not be relevant to your particular data.

[^10]: The `DWHT` and `LINEWHT` extensions are indeed inverse variance
    maps, where $\sigma = 1 / \sqrt{\mathrm{weight}}$. $\sigma$ can be
    used as a sigma image with GALFIT.

[^11]: Going from the *undistorted* mosaic to a distorted mosaic is
    "blotting\". Going in the opposite direction is "drizzling\". Still
    don't understand? Well don't shoot the messenger.

[^12]: As spoken by the Grizli God himself, Gabe Brammer.

[^13]: At least the ones corresponding to the filter for which you want
    direct image thumbnail for. Note, in older (before $\sim$May 2020)
    versions of Grizli, you would have needed ALL the `flt.fits` files
    for a particular field, otherwise the code would break.

[^14]: These files contain images of each HST pointing.

[^15]: As spoken by the Grizli God himself, Gabe Brammer.

[^16]: This most definitely was not me.

[^17]: You only need the `flt.fits` files corresponding to the filter
    you want the direct image to be in.

[^18]: These files contain images of each HST pointing.

[^19]: Otherwise the segmentation map thumbnail will not be generated.
    It's just the way of the code, deal with it.

[^20]: No, not a `python` pickle.

[^21]: So that's what Gabe meant in
    Section [6.2](#direct_full.fits){reference-type="ref"
    reference="direct_full.fits"} !

[^22]: A reminder that this book wasn't written by people who wrote
    Grizli.

[^23]: Otherwise the segmentation map thumbnail will not be generated.
    It's just the way of the code, deal with it.

[^24]: This most definitely did not happen to me.

[^25]: as generously given to me (and then adapted by me) by our Grizli
    God, Gabe Brammer.

[^26]: If there is residual contamination left in the `LINE` extension,
    this means Grizli failed to remove it. You may have to apply your
    own contamination removal techniques or if possible, see if you can
    use the associated `CONTAM` map to mask the problematic regions.

[^27]: Yes, I am British. The word 'catalogue' does not end at the 'g',
    obviously \*eye roll\*.

[^28]: All of the above, as said by the Grizli God himself, Gabe
    Brammer.
